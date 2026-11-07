"""Application use cases. Orchestrates domain logic through ports only."""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from triage.application.ports import (
    Classifier,
    Dashboard,
    Embedder,
    EscalationUnavailable,
    Escalator,
    ExampleStore,
    NullTelemetry,
    SemanticCache,
    Telemetry,
    TicketRepository,
    Transcriber,
    TranscriptionFailed,
    UrgencyFeatures,
    UrgencyModel,
)
from triage.domain.models import (
    Category,
    Complaint,
    DecidedBy,
    LabeledExample,
    Prediction,
    Sentiment,
    Ticket,
    TicketStatus,
    Urgency,
)
from triage.domain.normalization import detect_script, negation_signature, normalize
from triage.domain.pii import mask_pii
from triage.domain.routing import HUMAN_REVIEW_QUEUE, route


class UnknownModel(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TriagePolicy:
    encoder_threshold: float = 0.80
    llm_threshold: float = 0.70
    few_shot_examples: int = 5


class ClassifyComplaint:
    def __init__(
        self,
        classifiers: dict[str, Classifier],
        default_model: str,
        tickets: TicketRepository,
        policy: TriagePolicy | None = None,
        urgency_model: UrgencyModel | None = None,
        escalator: Escalator | None = None,
        embedder: Embedder | None = None,
        cache: SemanticCache | None = None,
        examples: ExampleStore | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._classifiers = classifiers
        self._default_model = default_model
        self._tickets = tickets
        self._policy = policy or TriagePolicy()
        self._urgency_model = urgency_model
        self._escalator = escalator
        self._embedder = embedder
        self._cache = cache
        self._examples = examples
        self._telemetry = telemetry or NullTelemetry()

    @property
    def available_models(self) -> list[str]:
        return list(self._classifiers)

    async def execute(self, complaint: Complaint, model: str | None = None) -> Ticket:
        started = time.perf_counter()
        model_name = model or self._default_model
        classifier = self._classifiers.get(model_name)
        if classifier is None:
            raise UnknownModel(model_name)

        ticket_id = complaint.ticket_id
        if complaint.external_id and (existing := await self._tickets.get(ticket_id)):
            self._telemetry.count("triage.idempotent_replays")
            return existing

        with self._telemetry.span("triage.classify", channel=complaint.channel, model=model_name):
            masked = mask_pii(complaint.text, complaint.customer_id)
            normalized = normalize(masked)
            prediction = await asyncio.to_thread(classifier.predict, normalized)
            decided_by = DecidedBy.BASELINE if model_name == "baseline" else DecidedBy.ENCODER
            review_reason = None

            if prediction.confidence < self._policy.encoder_threshold:
                prediction, decided_by, review_reason = await self._escalate(
                    masked, normalized, prediction, decided_by
                )

            prediction = await self._refine_urgency(prediction, complaint)
            needs_review = review_reason is not None
            ticket = Ticket(
                ticket_id=ticket_id,
                channel=complaint.channel,
                customer_id=complaint.customer_id,
                raw_text=complaint.text,
                masked_text=masked,
                normalized_text=normalized,
                script=detect_script(masked),
                prediction=prediction,
                decided_by=decided_by,
                target_queue=HUMAN_REVIEW_QUEUE if needs_review else route(prediction),
                status=TicketStatus.PENDING_REVIEW if needs_review else TicketStatus.ROUTED,
                latency_ms=(time.perf_counter() - started) * 1000,
                received_at=complaint.received_at,
                review_reason=review_reason,
            )
            stored = await self._tickets.save(ticket)

        self._telemetry.count(
            "triage.tickets", decided_by=stored.decided_by.value, queue=stored.target_queue
        )
        self._telemetry.record("triage.latency_ms", stored.latency_ms, model=model_name)
        self._telemetry.record("triage.confidence", prediction.confidence)
        return stored

    async def _escalate(
        self,
        masked: str,
        normalized: str,
        encoder_prediction: Prediction,
        encoder_decider: DecidedBy,
    ) -> tuple[Prediction, DecidedBy, str | None]:
        """Return the final prediction, who decided, and a review reason when a human must look."""
        if self._escalator is None:
            return encoder_prediction, encoder_decider, "low_confidence_no_llm"

        negation = negation_signature(normalized)
        version = self._escalator.prompt_version
        embedding: list[float] | None = None
        examples: list[LabeledExample] = []
        try:
            if self._embedder is not None:
                embedding = await asyncio.to_thread(self._embedder.embed, normalized)
            if embedding is not None and self._cache is not None:
                hit = await self._cache.lookup(embedding, negation, version)
                self._telemetry.count("triage.semantic_cache", result="hit" if hit else "miss")
                if hit is not None:
                    return hit.prediction, DecidedBy.CACHE, None
            if embedding is not None and self._examples is not None:
                examples = await self._examples.similar(embedding, self._policy.few_shot_examples)
        except Exception:
            # Cache and retrieval are optimizations: losing them must not lose the complaint.
            self._telemetry.count("triage.retrieval_errors")

        try:
            with self._telemetry.span("triage.escalate", few_shot=len(examples)):
                decision = await self._escalator.decide(masked, encoder_prediction, examples)
        except EscalationUnavailable:
            self._telemetry.count("triage.escalation_unavailable")
            return encoder_prediction, encoder_decider, "llm_unavailable"

        if decision.confidence < self._policy.llm_threshold:
            return decision, DecidedBy.LLM, "low_confidence_llm"

        if embedding is not None and self._cache is not None:
            try:
                await self._cache.store(masked, embedding, negation, version, decision)
            except Exception:
                self._telemetry.count("triage.cache_write_errors")
        return decision, DecidedBy.LLM, None

    async def _refine_urgency(self, prediction: Prediction, complaint: Complaint) -> Prediction:
        if self._urgency_model is None:
            return prediction
        repeats = 0
        if complaint.customer_id:
            repeats = await self._tickets.repeat_complaints(complaint.customer_id, hours=24)
        features = UrgencyFeatures(
            channel=complaint.channel,
            hour=complaint.received_at.astimezone(UTC).hour,
            repeat_complaints_24h=repeats,
        )
        urgency, confidence = await asyncio.to_thread(
            self._urgency_model.predict, prediction, features
        )
        return dataclasses.replace(prediction, urgency=urgency, urgency_confidence=confidence)


class IngestVoiceNote:
    def __init__(self, transcriber: Transcriber, classify: ClassifyComplaint) -> None:
        self._transcriber = transcriber
        self._classify = classify

    async def execute(
        self,
        audio: bytes,
        channel: str = "voice",
        external_id: str | None = None,
        customer_id: str | None = None,
    ) -> Ticket:
        text = await asyncio.to_thread(self._transcriber.transcribe, audio)
        if not text.strip():
            raise TranscriptionFailed("No speech found in the audio")
        return await self._classify.execute(
            Complaint(
                text=text,
                channel=channel,
                external_id=external_id,
                customer_id=customer_id,
                received_at=datetime.now(UTC),
            )
        )


class ReviewTicket:
    """Human decision. Overrides feed the few-shot store and the retraining set."""

    def __init__(
        self,
        tickets: TicketRepository,
        embedder: Embedder | None = None,
        examples: ExampleStore | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._tickets = tickets
        self._embedder = embedder
        self._examples = examples
        self._telemetry = telemetry or NullTelemetry()

    async def pending(self, limit: int = 50) -> list[Ticket]:
        return await self._tickets.pending_reviews(limit)

    async def execute(
        self,
        ticket_id: uuid.UUID,
        category: Category,
        urgency: Urgency,
        sentiment: Sentiment,
        reviewer: str,
    ) -> Ticket | None:
        current = await self._tickets.get(ticket_id)
        if current is None:
            return None
        decision = dataclasses.replace(
            current.prediction,
            category=category,
            category_confidence=1.0,
            urgency=urgency,
            urgency_confidence=1.0,
            sentiment=sentiment,
            sentiment_confidence=1.0,
        )
        updated = await self._tickets.apply_review(
            ticket_id, category, urgency, sentiment, route(decision), reviewer
        )
        if updated is None:
            return None
        overridden = (
            current.prediction.category is not category or current.prediction.urgency is not urgency
        )
        self._telemetry.count("triage.reviews", overridden=str(overridden).lower())
        if self._embedder is not None and self._examples is not None:
            embedding = await asyncio.to_thread(self._embedder.embed, current.normalized_text)
            await self._examples.add(
                LabeledExample(current.masked_text, category, urgency, sentiment), embedding
            )
        return updated


class GetDashboard:
    def __init__(self, tickets: TicketRepository) -> None:
        self._tickets = tickets

    async def execute(self, hours: int = 24) -> Dashboard:
        return await self._tickets.dashboard(hours)
