"""FastAPI adapter: translates HTTP to use cases and nothing else."""

from __future__ import annotations

import secrets
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from triage.application.ports import TranscriptionFailed
from triage.application.use_cases import UnknownModel
from triage.bootstrap import Container
from triage.domain.models import Category, Complaint, Sentiment, Ticket, Urgency

MAX_TEXT_LENGTH = 5000
MAX_BATCH_SIZE = 64


def key_matches(provided: str | None, accepted: list[str]) -> bool:
    return bool(provided) and any(secrets.compare_digest(provided, key) for key in accepted)


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    channel: str = Field(default="api", max_length=32)
    external_id: str | None = Field(default=None, max_length=128)
    customer_id: str | None = Field(default=None, max_length=128)


class BatchRequest(BaseModel):
    items: list[ClassifyRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class TriageResponse(BaseModel):
    ticket_id: uuid.UUID
    category: Category
    category_confidence: float
    urgency: Urgency
    urgency_confidence: float
    sentiment: Sentiment
    sentiment_confidence: float
    normalized_text: str
    script: str
    decided_by: str
    target_queue: str
    status: str
    review_reason: str | None
    model_version: str
    latency_ms: float

    @classmethod
    def from_ticket(cls, ticket: Ticket) -> TriageResponse:
        p = ticket.prediction
        return cls(
            ticket_id=ticket.ticket_id,
            category=p.category,
            category_confidence=p.category_confidence,
            urgency=p.urgency,
            urgency_confidence=p.urgency_confidence,
            sentiment=p.sentiment,
            sentiment_confidence=p.sentiment_confidence,
            normalized_text=ticket.normalized_text,
            script=ticket.script.value,
            decided_by=ticket.decided_by.value,
            target_queue=ticket.target_queue,
            status=ticket.status.value,
            review_reason=ticket.review_reason,
            model_version=p.model_version,
            latency_ms=ticket.latency_ms,
        )


class ReviewItem(TriageResponse):
    masked_text: str

    @classmethod
    def from_ticket(cls, ticket: Ticket) -> ReviewItem:
        base = TriageResponse.from_ticket(ticket).model_dump()
        return cls(**base, masked_text=ticket.masked_text)


class ReviewRequest(BaseModel):
    category: Category
    urgency: Urgency
    sentiment: Sentiment
    reviewer: str = Field(min_length=1, max_length=64)


def build_router(container: Container) -> APIRouter:
    router = APIRouter()

    def require_admin(x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = container.settings.admin_api_key
        if not expected:
            raise HTTPException(503, "ADMIN_API_KEY is not configured")
        if not key_matches(x_api_key, [expected]):
            raise HTTPException(401, "Invalid API key")

    def require_client(x_api_key: Annotated[str | None, Header()] = None) -> None:
        accepted = container.settings.client_keys
        if accepted and not key_matches(x_api_key, accepted):
            raise HTTPException(401, "Invalid API key")

    admin = [Depends(require_admin)]
    client = [Depends(require_client)]

    def complaint(request: ClassifyRequest) -> Complaint:
        return Complaint(
            text=request.text,
            channel=request.channel,
            external_id=request.external_id,
            customer_id=request.customer_id,
        )

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok", "models": sorted(container.classify.available_models)}

    @router.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @router.post("/classify", response_model=TriageResponse, dependencies=client)
    async def classify(request: ClassifyRequest, model: str | None = None) -> TriageResponse:
        try:
            ticket = await container.classify.execute(complaint(request), model=model)
        except UnknownModel as error:
            raise HTTPException(400, f"Unknown model: {error}") from error
        return TriageResponse.from_ticket(ticket)

    @router.post("/classify/batch", response_model=list[TriageResponse], dependencies=client)
    async def classify_batch(
        request: BatchRequest, model: str | None = None
    ) -> list[TriageResponse]:
        results = []
        try:
            for item in request.items:
                ticket = await container.classify.execute(complaint(item), model=model)
                results.append(TriageResponse.from_ticket(ticket))
        except UnknownModel as error:
            raise HTTPException(400, f"Unknown model: {error}") from error
        return results

    @router.post("/classify/voice", response_model=TriageResponse, dependencies=client)
    async def classify_voice(
        audio: Annotated[UploadFile, File()],
        channel: Annotated[str, Form(max_length=32)] = "voice",
        external_id: Annotated[str | None, Form(max_length=128)] = None,
        customer_id: Annotated[str | None, Form(max_length=128)] = None,
    ) -> TriageResponse:
        if container.voice is None:
            raise HTTPException(503, "Voice transcription is disabled")
        limit = container.settings.max_audio_bytes
        data = await audio.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, "Audio file too large")
        try:
            ticket = await container.voice.execute(data, channel, external_id, customer_id)
        except TranscriptionFailed as error:
            raise HTTPException(422, str(error)) from error
        return TriageResponse.from_ticket(ticket)

    @router.get("/review", response_model=list[ReviewItem], dependencies=admin)
    async def pending(limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[ReviewItem]:
        return [ReviewItem.from_ticket(t) for t in await container.review.pending(limit)]

    @router.post("/review/{ticket_id}", response_model=TriageResponse, dependencies=admin)
    async def review(ticket_id: uuid.UUID, request: ReviewRequest) -> TriageResponse:
        ticket = await container.review.execute(
            ticket_id, request.category, request.urgency, request.sentiment, request.reviewer
        )
        if ticket is None:
            raise HTTPException(404, "Ticket not found")
        return TriageResponse.from_ticket(ticket)

    @router.get("/dashboard", dependencies=admin)
    async def dashboard(hours: Annotated[int, Query(ge=1, le=720)] = 24) -> dict:
        return asdict(await container.dashboard.execute(hours))

    return router


def mount_frontend(app: FastAPI, dist: Path) -> None:
    """Registered last so API routes take priority over the static catch-all."""
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
