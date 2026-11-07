"""Exercises the FastAPI adapter and MCP tool over the use cases, with in-memory fakes."""

import asyncio
import dataclasses
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import Client

from tests.fakes import FakeClassifier, FakeTickets, FakeTranscriber, make_prediction
from triage.adapters.inbound.http import build_router
from triage.adapters.inbound.mcp import build_mcp
from triage.application.use_cases import (
    ClassifyComplaint,
    GetDashboard,
    IngestVoiceNote,
    ReviewTicket,
)
from triage.settings import Settings

KEY = "test-admin-key"


@dataclass
class Harness:
    client: TestClient
    tickets: FakeTickets
    container: object


def harness(admin_key=KEY, voice=True, classifier=None) -> Harness:
    tickets = FakeTickets()
    classify = ClassifyComplaint(
        {"encoder": classifier or FakeClassifier(), "baseline": FakeClassifier()},
        "encoder",
        tickets,
    )

    @dataclass
    class StubContainer:
        settings: Settings
        classify: ClassifyComplaint
        review: ReviewTicket
        dashboard: GetDashboard
        voice: IngestVoiceNote | None

    container = StubContainer(
        Settings(admin_api_key=admin_key, max_audio_bytes=100, _env_file=None),
        classify,
        ReviewTicket(tickets),
        GetDashboard(tickets),
        IngestVoiceNote(FakeTranscriber(), classify) if voice else None,
    )
    app = FastAPI()
    app.include_router(build_router(container))
    return Harness(TestClient(app), tickets, container)


def test_classify_returns_contract_fields():
    body = harness().client.post("/classify", json={"text": "النت واقع"}).json()
    assert body["category"] == "outage"
    assert body["decided_by"] == "encoder"
    assert body["target_queue"] == "noc_priority"
    for field in ("ticket_id", "normalized_text", "model_version", "latency_ms", "sentiment"):
        assert field in body
    assert "raw_text" not in body


def test_classify_masks_pii_in_response():
    body = harness().client.post("/classify", json={"text": "رقمي 01012345678"}).json()
    assert "01012345678" not in body["normalized_text"]


def test_baseline_query_parameter():
    body = harness().client.post("/classify?model=baseline", json={"text": "x"}).json()
    assert body["decided_by"] == "baseline"


def test_unknown_model_is_400():
    assert harness().client.post("/classify?model=nope", json={"text": "x"}).status_code == 400


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 5001}, {}])
def test_invalid_text_is_422(payload):
    assert harness().client.post("/classify", json=payload).status_code == 422


def test_batch_limit():
    client = harness().client
    assert client.post("/classify/batch", json={"items": []}).status_code == 422
    too_many = {"items": [{"text": "x"}] * 65}
    assert client.post("/classify/batch", json=too_many).status_code == 422
    assert len(client.post("/classify/batch", json={"items": [{"text": "x"}] * 2}).json()) == 2


def test_replay_with_external_id_returns_same_ticket():
    client = harness().client
    payload = {"text": "x", "channel": "whatsapp", "external_id": "m1"}
    assert (
        client.post("/classify", json=payload).json()["ticket_id"]
        == client.post("/classify", json=payload).json()["ticket_id"]
    )


def test_voice_endpoint():
    client = harness().client
    response = client.post("/classify/voice", files={"audio": ("a.ogg", "النت واقع".encode())})
    assert response.status_code == 200
    assert response.json()["normalized_text"] == "النت واقع"


def test_voice_with_no_speech_is_422():
    response = harness().client.post("/classify/voice", files={"audio": ("a.ogg", b"   ")})
    assert response.status_code == 422


def test_voice_too_large_is_413():
    response = harness().client.post("/classify/voice", files={"audio": ("a.ogg", b"x" * 101)})
    assert response.status_code == 413


def test_voice_disabled_is_503():
    response = harness(voice=False).client.post("/classify/voice", files={"audio": ("a", b"x")})
    assert response.status_code == 503


@pytest.mark.parametrize("path", ["/review", "/dashboard"])
def test_admin_routes_reject_missing_and_wrong_key(path):
    client = harness().client
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get(path, headers={"X-API-Key": KEY}).status_code == 200


def test_admin_routes_closed_when_key_unset():
    assert harness(admin_key=None).client.get("/review").status_code == 503


def test_review_flow_exposes_masked_text_only():
    h = harness(classifier=FakeClassifier(make_prediction(0.3)))
    created = h.client.post("/classify", json={"text": "رقمي 01012345678"}).json()
    assert created["status"] == "pending_review"
    headers = {"X-API-Key": KEY}
    queue = h.client.get("/review", headers=headers).json()
    assert queue[0]["masked_text"] == "رقمي [phone]"
    assert "raw_text" not in queue[0]
    decision = {"category": "speed", "urgency": "normal", "sentiment": "neutral", "reviewer": "a"}
    done = h.client.post(f"/review/{created['ticket_id']}", json=decision, headers=headers)
    assert done.json()["decided_by"] == "human"
    assert h.client.get("/review", headers=headers).json() == []


def test_review_unknown_is_404():
    decision = {"category": "speed", "urgency": "normal", "sentiment": "neutral", "reviewer": "a"}
    response = harness().client.post(
        "/review/00000000-0000-0000-0000-000000000000",
        json=decision,
        headers={"X-API-Key": KEY},
    )
    assert response.status_code == 404


def test_dashboard_shape():
    h = harness()
    h.client.post("/classify", json={"text": "x"})
    body = h.client.get("/dashboard", headers={"X-API-Key": KEY}).json()
    assert body["total"] == 1
    assert body["by_category"] == {"outage": 1}


def test_health_and_metrics_are_public():
    client = harness().client
    assert sorted(client.get("/health").json()["models"]) == ["baseline", "encoder"]
    assert client.get("/metrics").status_code == 200


def test_mcp_tool_calls_same_use_case():
    h = harness()

    async def call():
        async with Client(build_mcp(h.container)) as client:
            tools = await client.list_tools()
            result = await client.call_tool("classify_complaint", {"text": "النت واقع"})
            return tools, result

    tools, result = asyncio.run(call())
    assert [t.name for t in tools] == ["classify_complaint"]
    assert result.data["target_queue"] == "noc_priority"
    assert len(h.tickets.items) == 1


def test_fakes_stay_frozen():
    assert dataclasses.is_dataclass(make_prediction())
