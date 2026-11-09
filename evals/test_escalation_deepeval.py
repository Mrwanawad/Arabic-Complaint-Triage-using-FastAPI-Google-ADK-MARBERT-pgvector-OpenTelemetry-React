"""LLM regression suite using DeepEval with custom, deterministic metrics.

With provider keys it calls real providers and costs a small amount:
    pytest evals/test_escalation_deepeval.py -m live
Without keys, or with EVAL_PROVIDER=offline, the same path runs against a local
OpenAI-compatible server answered by the TF-IDF baseline (evals/offline_provider.py).
That checks the harness and the ADK -> LiteLLM -> HTTP path, not real LLM quality.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals import offline_provider
from triage.adapters.outbound.adk_escalator import (
    PROMPT_VERSION,
    AdkEscalator,
    CircuitBreaker,
    build_route,
    litellm_model,
)
from triage.application.ports import NullTelemetry
from triage.bootstrap import build_escalator
from triage.domain.models import Category, Prediction, Sentiment, Urgency
from triage.domain.pii import mask_pii
from triage.settings import Settings

pytestmark = pytest.mark.live

GOLDEN = [
    json.loads(line)
    for line in Path(__file__).with_name("golden.jsonl").read_text(encoding="utf-8").splitlines()
]
UNSURE = Prediction(Category.OTHER, 0.3, Urgency.NORMAL, 0.3, Sentiment.NEUTRAL, 0.3, "eval")


class LabelMatch(BaseMetric):
    """Exact category and urgency match. Tone labels are subjective, so sentiment is excluded."""

    def __init__(self) -> None:
        self.threshold = 1.0
        self.evaluation_model = None
        self.include_reason = True
        self.strict_mode = True
        self.async_mode = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual = json.loads(test_case.actual_output)
        expected = json.loads(test_case.expected_output)
        matched = all(actual[key] == expected[key] for key in ("category", "urgency"))
        self.score = float(matched)
        self.success = matched
        self.reason = f"expected {expected}, got {actual}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Label Match"


@pytest.fixture(scope="module")
def escalator():
    settings = Settings()
    built = None
    if os.getenv("EVAL_PROVIDER") != "offline":
        built = build_escalator(settings, limiter=None, telemetry=NullTelemetry())
    if built is not None:
        yield built
        return
    if not settings.baseline_path.exists():
        pytest.skip("No provider key and no baseline artifact for the offline provider")
    with offline_provider.serve(settings.baseline_path) as url:
        model = litellm_model("openai/offline-baseline", 15, api_base=url, api_key="offline")
        route = build_route("offline", model, "offline-baseline", 1.0, True, CircuitBreaker())
        yield AdkEscalator([route], timeout_seconds=30)


@pytest.mark.parametrize("case", GOLDEN, ids=[g["text"][:30] for g in GOLDEN])
def test_escalation_matches_golden(escalator, case):
    result = asyncio.run(escalator.decide(mask_pii(case["text"]), UNSURE, []))
    assert_test(
        LLMTestCase(
            input=case["text"],
            actual_output=json.dumps(
                {
                    "category": result.category.value,
                    "urgency": result.urgency.value,
                    "sentiment": result.sentiment.value,
                }
            ),
            expected_output=json.dumps(case),
            additional_metadata={
                "prompt_version": PROMPT_VERSION,
                "model": result.model_version,
            },
        ),
        [LabelMatch()],
        run_async=False,
    )
