"""Composition root: the only place that knows which adapter implements which port."""

from __future__ import annotations

import asyncio
import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any

from triage.adapters.outbound.classifiers import (
    BaselineClassifier,
    LightGbmUrgency,
    OnnxClassifier,
    WhisperTranscriber,
)
from triage.adapters.outbound.embedders import HashingEmbedder, OnnxEmbedder
from triage.adapters.outbound.postgres import (
    PostgresExamples,
    PostgresRateLimiter,
    PostgresSemanticCache,
    PostgresTickets,
    apply_schema,
    create_pool,
)
from triage.application.ports import Classifier, Telemetry
from triage.application.use_cases import (
    ClassifyComplaint,
    GetDashboard,
    IngestVoiceNote,
    ReviewTicket,
    TriagePolicy,
)
from triage.settings import Settings

_PURGE_INTERVAL_SECONDS = 3600


def load_classifiers(settings: Settings) -> dict[str, Classifier]:
    classifiers: dict[str, Classifier] = {}
    if (settings.encoder_dir / "model.onnx").exists():
        classifiers["encoder"] = OnnxClassifier(settings.encoder_dir)
    if settings.baseline_path.exists():
        classifiers["baseline"] = BaselineClassifier(settings.baseline_path)
    if not classifiers:
        raise RuntimeError(f"No classifier found in {settings.artifacts_dir}. Train one first.")
    return classifiers


def build_escalator(settings: Settings, limiter: Any, telemetry: Telemetry):
    """Only providers with a configured key join the route table."""
    if not settings.llm_enabled:
        return None
    # ADK imports are slow and heavy, so deployments with no LLM skip them.
    from triage.adapters.outbound.adk_escalator import (
        AdkEscalator,
        CircuitBreaker,
        build_route,
        litellm_model,
    )

    def breaker() -> CircuitBreaker:
        return CircuitBreaker(settings.breaker_failures, settings.breaker_cooldown_seconds)

    routes = []
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        routes.append(
            build_route(
                "gemini",
                settings.gemini_model,
                settings.gemini_model,
                settings.gemini_weight,
                True,
                breaker(),
            )
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        routes.append(
            build_route(
                "anthropic",
                litellm_model(settings.anthropic_model, settings.llm_timeout_seconds),
                settings.anthropic_model,
                settings.anthropic_weight,
                True,
                breaker(),
            )
        )
    if os.getenv("DEEPSEEK_API_KEY"):
        routes.append(
            build_route(
                "deepseek",
                litellm_model(settings.deepseek_model, settings.llm_timeout_seconds),
                settings.deepseek_model,
                1.0,
                False,
                breaker(),
            )
        )
    if not routes:
        return None
    return AdkEscalator(
        routes, limiter, settings.provider_rpm, settings.llm_timeout_seconds, telemetry
    )


def build_embedder(settings: Settings):
    if (settings.embedder_dir / "model.onnx").exists():
        return OnnxEmbedder(settings.embedder_dir)
    return HashingEmbedder()


@dataclass
class Container:
    settings: Settings
    classify: ClassifyComplaint
    review: ReviewTicket
    dashboard: GetDashboard
    voice: IngestVoiceNote | None
    pools: list[Any] = field(default_factory=list)
    limiter: PostgresRateLimiter | None = None
    cache: PostgresSemanticCache | None = None
    _purge_task: asyncio.Task | None = None

    async def startup(self) -> None:
        await apply_schema(self.settings.database_url)
        for pool in self.pools:
            await pool.open(wait=False)
        self._purge_task = asyncio.create_task(self._purge_loop())

    async def shutdown(self) -> None:
        if self._purge_task is not None:
            self._purge_task.cancel()
        for pool in self.pools:
            await pool.close()

    async def _purge_loop(self) -> None:
        while True:
            await asyncio.sleep(_PURGE_INTERVAL_SECONDS)
            try:
                await self.cache.purge_expired()
                await self.limiter.purge_old()
            except Exception:
                # Purging is housekeeping. The next cycle retries it.
                continue


def build_container(settings: Settings, telemetry: Telemetry) -> Container:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required")

    primary = create_pool(settings.database_url, settings.db_pool_size)
    pools = [primary]
    replica = None
    if settings.database_replica_url:
        replica = create_pool(settings.database_replica_url, settings.db_pool_size)
        pools.append(replica)

    tickets = PostgresTickets(primary, replica)
    cache = PostgresSemanticCache(primary, settings.cache_threshold, settings.cache_ttl_days)
    examples = PostgresExamples(primary)
    limiter = PostgresRateLimiter(primary)
    embedder = build_embedder(settings)
    classifiers = load_classifiers(settings)
    default_model = (
        settings.default_model if settings.default_model in classifiers else next(iter(classifiers))
    )
    urgency_model = (
        LightGbmUrgency(settings.urgency_path) if settings.urgency_path.exists() else None
    )

    classify = ClassifyComplaint(
        classifiers=classifiers,
        default_model=default_model,
        tickets=tickets,
        policy=TriagePolicy(
            settings.encoder_threshold, settings.llm_threshold, settings.few_shot_examples
        ),
        urgency_model=urgency_model,
        escalator=build_escalator(settings, limiter, telemetry),
        embedder=embedder,
        cache=cache,
        examples=examples,
        telemetry=telemetry,
    )
    voice = None
    if settings.transcriber_enabled and importlib.util.find_spec("faster_whisper"):
        voice = IngestVoiceNote(WhisperTranscriber(settings.whisper_model), classify)
    return Container(
        settings=settings,
        classify=classify,
        review=ReviewTicket(tickets, embedder, examples, telemetry),
        dashboard=GetDashboard(tickets),
        voice=voice,
        pools=pools,
        limiter=limiter,
        cache=cache,
    )
