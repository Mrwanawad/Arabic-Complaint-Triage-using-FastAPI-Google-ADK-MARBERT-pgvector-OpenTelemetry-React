"""ASGI entry point: uvicorn triage.main:app"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from triage.adapters.inbound.http import build_router, mount_frontend
from triage.adapters.inbound.mcp import build_mcp
from triage.application.ports import NullTelemetry
from triage.artifacts import ensure_artifacts
from triage.bootstrap import Container, build_container
from triage.settings import Settings


def create_app(container: Container | None = None) -> FastAPI:
    settings = container.settings if container else Settings()
    instrument = settings.otel_enabled and container is None
    if instrument:
        from triage.adapters.outbound.telemetry import OtelTelemetry, configure_telemetry

        configure_telemetry()
        telemetry = OtelTelemetry()
    else:
        telemetry = NullTelemetry()
    if container is None:
        ensure_artifacts(settings.artifacts_dir, settings.artifacts_url, settings.artifacts_sha256)
    active = container or build_container(settings, telemetry)
    mcp_app = build_mcp(active).http_app(path="/", stateless_http=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await active.startup()
        async with mcp_app.lifespan(app):
            yield
        await active.shutdown()

    app = FastAPI(title="Arabic Complaint Triage", version="0.1.0", lifespan=lifespan)
    app.include_router(build_router(active))
    app.mount("/mcp", mcp_app)
    if instrument:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
    mount_frontend(app, Path("frontend/dist"))
    return app


app = create_app()
