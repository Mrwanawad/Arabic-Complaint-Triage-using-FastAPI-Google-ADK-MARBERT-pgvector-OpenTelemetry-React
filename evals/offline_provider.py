"""Local OpenAI-compatible chat API that answers with the TF-IDF baseline, for keyless eval runs.

It lets the DeepEval suite drive the real path (ADK -> LiteLLM -> HTTP -> schema validation)
without provider keys. Scores then measure the baseline behind that path, not a real LLM.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from triage.adapters.outbound.classifiers import BaselineClassifier
from triage.domain.normalization import normalize

COMPLAINT = re.compile(r"<complaint>(.*?)</complaint>", re.DOTALL)


def answer(classifier: BaselineClassifier, prompt: str) -> str:
    match = COMPLAINT.search(prompt)
    p = classifier.predict(normalize(match.group(1) if match else prompt))
    return json.dumps(
        {
            "category": p.category.value,
            "category_confidence": round(p.category_confidence, 4),
            "urgency": p.urgency.value,
            "urgency_confidence": round(p.urgency_confidence, 4),
            "sentiment": p.sentiment.value,
            "sentiment_confidence": round(p.sentiment_confidence, 4),
        }
    )


def last_user_text(body: dict) -> str:
    for message in reversed(body.get("messages", [])):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content)
            return content or ""
    return ""


@contextmanager
def serve(baseline_path: Path) -> Iterator[str]:
    """Yields the base URL of a running server and stops it on exit."""
    classifier = BaselineClassifier(baseline_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = answer(classifier, last_user_text(body))
            encoded = json.dumps(
                {
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "offline"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
