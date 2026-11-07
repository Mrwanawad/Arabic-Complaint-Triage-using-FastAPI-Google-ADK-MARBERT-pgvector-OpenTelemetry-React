"""Interfaces the application needs from the outside world. Adapters implement these."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

from triage.domain.models import (
    Category,
    LabeledExample,
    Prediction,
    Sentiment,
    Ticket,
    Urgency,
)


class EscalationUnavailable(Exception):
    """No LLM provider could answer: all failed, were over budget, or returned invalid output."""


class TranscriptionFailed(Exception):
    """The audio could not be decoded or contained no speech."""


@dataclass(frozen=True, slots=True)
class CachedDecision:
    prediction: Prediction
    similarity: float


@dataclass(frozen=True, slots=True)
class UrgencyFeatures:
    channel: str
    hour: int
    repeat_complaints_24h: int


@dataclass(frozen=True, slots=True)
class Dashboard:
    total: int
    by_category: dict[str, int]
    by_urgency: dict[str, int]
    by_sentiment: dict[str, int]
    by_decider: dict[str, int]
    by_hour: list[dict[str, Any]] = field(default_factory=list)
    pending_review: int = 0
    override_rate: float = 0.0


class Classifier(Protocol):
    version: str

    def predict(self, normalized_text: str) -> Prediction: ...


class UrgencyModel(Protocol):
    def predict(
        self, prediction: Prediction, features: UrgencyFeatures
    ) -> tuple[Urgency, float]: ...


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class Escalator(Protocol):
    prompt_version: str

    async def decide(
        self, masked_text: str, encoder_prediction: Prediction, examples: Sequence[LabeledExample]
    ) -> Prediction: ...


class Transcriber(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class SemanticCache(Protocol):
    async def lookup(
        self, embedding: list[float], negation: str, prompt_version: str
    ) -> CachedDecision | None: ...

    async def store(
        self,
        masked_text: str,
        embedding: list[float],
        negation: str,
        prompt_version: str,
        prediction: Prediction,
    ) -> None: ...


class ExampleStore(Protocol):
    async def similar(self, embedding: list[float], limit: int) -> list[LabeledExample]: ...

    async def add(self, example: LabeledExample, embedding: list[float]) -> None: ...


class TicketRepository(Protocol):
    async def get(self, ticket_id: uuid.UUID) -> Ticket | None: ...

    async def save(self, ticket: Ticket) -> Ticket:
        """Insert. On duplicate ticket_id, return the stored ticket and change nothing."""
        ...

    async def repeat_complaints(self, customer_id: str, hours: int) -> int: ...

    async def pending_reviews(self, limit: int) -> list[Ticket]: ...

    async def apply_review(
        self,
        ticket_id: uuid.UUID,
        category: Category,
        urgency: Urgency,
        sentiment: Sentiment,
        target_queue: str,
        reviewer: str,
    ) -> Ticket | None: ...

    async def dashboard(self, hours: int) -> Dashboard: ...


class Telemetry(Protocol):
    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Any]: ...

    def count(self, name: str, value: int = 1, **attributes: Any) -> None: ...

    def record(self, name: str, value: float, **attributes: Any) -> None: ...


class NullTelemetry:
    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Any]:
        return nullcontext()

    def count(self, name: str, value: int = 1, **attributes: Any) -> None:
        return None

    def record(self, name: str, value: float, **attributes: Any) -> None:
        return None
