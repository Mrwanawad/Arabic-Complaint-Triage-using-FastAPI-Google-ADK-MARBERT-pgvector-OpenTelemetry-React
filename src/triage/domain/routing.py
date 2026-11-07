"""Routing policy: maps a final decision to a support queue."""

from __future__ import annotations

from triage.domain.models import Category, Prediction, Urgency

HUMAN_REVIEW_QUEUE = "human_review"
OUTAGE_PRIORITY_QUEUE = "noc_priority"

_CATEGORY_QUEUES = {
    Category.BILLING: "billing",
    Category.OUTAGE: "noc",
    Category.SPEED: "technical_support",
    Category.INSTALLATION: "field_operations",
    Category.OTHER: "general_support",
}


def route(prediction: Prediction) -> str:
    if prediction.category is Category.OUTAGE and prediction.urgency is Urgency.HIGH:
        return OUTAGE_PRIORITY_QUEUE
    queue = _CATEGORY_QUEUES[prediction.category]
    return f"{queue}_priority" if prediction.urgency is Urgency.HIGH else queue
