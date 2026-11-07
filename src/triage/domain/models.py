"""Core entities. No framework imports allowed in this package."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

TICKET_NAMESPACE = uuid.UUID("6f1d3c52-9a4e-4b7a-8d21-3e5c7a9b1f00")


class Category(StrEnum):
    BILLING = "billing"
    OUTAGE = "outage"
    SPEED = "speed"
    INSTALLATION = "installation"
    OTHER = "other"


class Urgency(StrEnum):
    HIGH = "high"
    NORMAL = "normal"


class Sentiment(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class Script(StrEnum):
    ARABIC = "arabic"
    FRANCO = "franco"
    ENGLISH = "english"
    MIXED = "mixed"


class DecidedBy(StrEnum):
    ENCODER = "encoder"
    BASELINE = "baseline"
    LLM = "llm"
    CACHE = "cache"
    HUMAN = "human"


class TicketStatus(StrEnum):
    ROUTED = "routed"
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"


@dataclass(frozen=True, slots=True)
class Prediction:
    category: Category
    category_confidence: float
    urgency: Urgency
    urgency_confidence: float
    sentiment: Sentiment
    sentiment_confidence: float
    model_version: str

    @property
    def confidence(self) -> float:
        """Routing depends on category and urgency, so the weaker one gates the decision."""
        return min(self.category_confidence, self.urgency_confidence)


@dataclass(frozen=True, slots=True)
class LabeledExample:
    text: str
    category: Category
    urgency: Urgency
    sentiment: Sentiment


@dataclass(frozen=True, slots=True)
class Complaint:
    text: str
    channel: str = "api"
    external_id: str | None = None
    customer_id: str | None = None
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def ticket_id(self) -> uuid.UUID:
        """Deterministic when the channel supplies an id, which makes retries idempotent."""
        if self.external_id:
            return uuid.uuid5(TICKET_NAMESPACE, f"{self.channel}:{self.external_id}")
        return uuid.uuid4()


@dataclass(frozen=True, slots=True)
class Ticket:
    ticket_id: uuid.UUID
    channel: str
    customer_id: str | None
    raw_text: str
    masked_text: str
    normalized_text: str
    script: Script
    prediction: Prediction
    decided_by: DecidedBy
    target_queue: str
    status: TicketStatus
    latency_ms: float
    received_at: datetime
    review_reason: str | None = None
