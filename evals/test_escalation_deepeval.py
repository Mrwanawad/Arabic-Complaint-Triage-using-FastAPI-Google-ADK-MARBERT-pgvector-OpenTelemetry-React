"""LLM regression suite using DeepEval with custom, deterministic metrics.

Calls real providers, so it needs API keys and costs a small amount.
Usage: pytest evals/test_escalation_deepeval.py -m live
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from triage.adapters.outbound.adk_escalator import PROMPT_VERSION
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
    built = build_escalator(Settings(), limiter=None, telemetry=NullTelemetry())
    if built is None:
        pytest.skip("No LLM provider key configured")
    return built


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
