"""Drives ADK -> LiteLLM -> real HTTP against a local stub of DeepSeek's OpenAI-compatible API.

This covers the request and response path without keys. It cannot prove a real provider accepts
the request. For that, run evals/test_escalation_deepeval.py with real keys.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.fakes import make_prediction
from tests.test_adk_escalator import VALID
from triage.adapters.outbound.adk_escalator import (
    AdkEscalator,
    CircuitBreaker,
    build_route,
    litellm_model,
)
from triage.application.ports import EscalationUnavailable
from triage.domain.models import Category


@pytest.fixture
def stub_api():
    state = {"requests": [], "status": 200, "content": VALID}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "body": body})
            payload = {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": state["content"]},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            encoded = json.dumps(payload).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def escalator(stub_api):
    model = litellm_model("deepseek/deepseek-chat", 5, api_base=stub_api["url"], api_key="test-key")
    route = build_route("deepseek", model, "deepseek-chat", 1.0, False, CircuitBreaker())
    return AdkEscalator([route], timeout_seconds=15)


def decide(stub_api, text="النت واقع [phone]"):
    return asyncio.run(escalator(stub_api).decide(text, make_prediction(0.4), []))


def test_request_and_response_roundtrip(stub_api):
    result = decide(stub_api)
    assert result.category is Category.BILLING
    assert result.model_version == "deepseek-chat+triage-v1"
    body = stub_api["requests"][0]["body"]
    assert body["model"] == "deepseek-chat"
    assert body["temperature"] == 0
    text = json.dumps(body, ensure_ascii=False)
    assert "النت واقع [phone]" in text
    assert "You classify telecom customer complaints" in text


def test_provider_error_becomes_unavailable(stub_api):
    stub_api["status"] = 500
    with pytest.raises(EscalationUnavailable):
        decide(stub_api)


def test_malformed_content_becomes_unavailable(stub_api):
    stub_api["content"] = "I cannot help with that"
    with pytest.raises(EscalationUnavailable):
        decide(stub_api)
