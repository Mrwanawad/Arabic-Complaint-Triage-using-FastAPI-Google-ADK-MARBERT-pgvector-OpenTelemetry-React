"""FastMCP adapter: a second inbound door onto the same use case as the REST API."""

from __future__ import annotations

from fastmcp import FastMCP

from triage.bootstrap import Container
from triage.domain.models import Complaint


def build_mcp(container: Container) -> FastMCP:
    mcp = FastMCP("arabic-complaint-triage")

    @mcp.tool
    async def classify_complaint(text: str, channel: str = "mcp") -> dict:
        """Classify a telecom complaint in Arabic, Franco-Arabic, or English, and route it."""
        ticket = await container.classify.execute(Complaint(text=text[:5000], channel=channel[:32]))
        p = ticket.prediction
        return {
            "ticket_id": str(ticket.ticket_id),
            "category": p.category.value,
            "urgency": p.urgency.value,
            "sentiment": p.sentiment.value,
            "confidence": p.confidence,
            "target_queue": ticket.target_queue,
            "status": ticket.status.value,
            "decided_by": ticket.decided_by.value,
        }

    return mcp
