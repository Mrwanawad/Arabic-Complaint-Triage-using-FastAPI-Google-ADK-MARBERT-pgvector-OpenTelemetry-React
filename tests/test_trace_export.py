"""Checks the Langfuse setup from .env.example against a local OTLP receiver.

Langfuse accepts OTLP/HTTP traces with Basic auth. This proves the exporter honors the
documented variables and sends spans with that header. A real Langfuse account is still the
only proof that Langfuse accepts them.
"""

import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

PATH = "/api/public/otel/v1/traces"


@pytest.fixture
def receiver():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append({"path": self.path, "headers": self.headers, "body": body})
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_spans_reach_langfuse_style_endpoint_with_basic_auth(receiver, monkeypatch):
    url, received = receiver
    token = base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", url + PATH)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", f"Authorization=Basic {token}")

    provider = TracerProvider(resource=Resource.create({"service.name": "arabic-complaint-triage"}))
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    with provider.get_tracer("triage").start_as_current_span(
        "triage.llm_call", attributes={"provider": "gemini"}
    ):
        pass
    provider.shutdown()

    assert len(received) == 1
    request = received[0]
    assert request["path"] == PATH
    assert request["headers"]["Authorization"] == f"Basic {token}"
    assert request["headers"]["Content-Type"] == "application/x-protobuf"
    export = ExportTraceServiceRequest.FromString(request["body"])
    spans = [s for r in export.resource_spans for ss in r.scope_spans for s in ss.spans]
    assert [s.name for s in spans] == ["triage.llm_call"]
