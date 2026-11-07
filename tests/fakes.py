"""In-memory port implementations. They also prove the core runs with no vendor code."""

from __future__ import annotations

import dataclasses
import uuid
from collections import Counter

from triage.application.ports import CachedDecision, Dashboard, EscalationUnavailable
from triage.domain.models import (
    Category,
    DecidedBy,
    LabeledExample,
    Prediction,
    Sentiment,
    Ticket,
    TicketStatus,
    Urgency,
)


def make_prediction(confidence: float = 0.95, category: Category = Category.OUTAGE) -> Prediction:
    return Prediction(
        category, confidence, Urgency.HIGH, confidence, Sentiment.NEGATIVE, confidence, "fake-v1"
    )


class FakeClassifier:
    version = "fake-v1"

    def __init__(self, prediction: Prediction | None = None) -> None:
        self.prediction = prediction or make_prediction()
        self.calls: list[str] = []

    def predict(self, normalized_text: str) -> Prediction:
        self.calls.append(normalized_text)
        return self.prediction


class FakeEscalator:
    prompt_version = "p1"

    def __init__(self, prediction: Prediction | None = None, fail: bool = False) -> None:
        self.prediction = prediction or make_prediction(0.9, Category.BILLING)
        self.fail = fail
        self.calls: list[tuple[str, list[LabeledExample]]] = []

    async def decide(self, masked_text, encoder_prediction, examples) -> Prediction:
        self.calls.append((masked_text, list(examples)))
        if self.fail:
            raise EscalationUnavailable("all providers down")
        return self.prediction


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class FakeCache:
    def __init__(self, hit: Prediction | None = None) -> None:
        self.hit = hit
        self.stored: list[tuple[str, str, str]] = []
        self.fail = False

    async def lookup(self, embedding, negation, prompt_version):
        if self.fail:
            raise ConnectionError("cache down")
        return CachedDecision(self.hit, 0.99) if self.hit else None

    async def store(self, masked_text, embedding, negation, prompt_version, prediction) -> None:
        self.stored.append((masked_text, negation, prompt_version))


class FakeExamples:
    def __init__(self, items: list[LabeledExample] | None = None) -> None:
        self.items = items or []

    async def similar(self, embedding, limit):
        return self.items[:limit]

    async def add(self, example, embedding) -> None:
        self.items.append(example)


class FakeUrgency:
    def __init__(self) -> None:
        self.features = None

    def predict(self, prediction, features):
        self.features = features
        return Urgency.NORMAL, 0.77


class FakeTranscriber:
    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8")


class FakeTickets:
    def __init__(self, repeats: int = 0) -> None:
        self.items: dict[uuid.UUID, Ticket] = {}
        self.repeats = repeats

    async def get(self, ticket_id):
        return self.items.get(ticket_id)

    async def save(self, ticket):
        return self.items.setdefault(ticket.ticket_id, ticket)

    async def repeat_complaints(self, customer_id, hours):
        return self.repeats

    async def pending_reviews(self, limit):
        pending = [t for t in self.items.values() if t.status is TicketStatus.PENDING_REVIEW]
        return pending[:limit]

    async def apply_review(self, ticket_id, category, urgency, sentiment, target_queue, reviewer):
        current = self.items.get(ticket_id)
        if current is None:
            return None
        updated = dataclasses.replace(
            current,
            prediction=dataclasses.replace(
                current.prediction, category=category, urgency=urgency, sentiment=sentiment
            ),
            decided_by=DecidedBy.HUMAN,
            target_queue=target_queue,
            status=TicketStatus.REVIEWED,
        )
        self.items[ticket_id] = updated
        return updated

    async def dashboard(self, hours):
        tickets = list(self.items.values())
        return Dashboard(
            total=len(tickets),
            by_category=dict(Counter(t.prediction.category.value for t in tickets)),
            by_urgency=dict(Counter(t.prediction.urgency.value for t in tickets)),
            by_sentiment=dict(Counter(t.prediction.sentiment.value for t in tickets)),
            by_decider=dict(Counter(t.decided_by.value for t in tickets)),
        )
