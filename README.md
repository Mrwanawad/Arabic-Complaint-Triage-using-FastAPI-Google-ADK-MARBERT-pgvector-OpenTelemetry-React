# Arabic Complaint Triage

Classifies telecom complaints in Egyptian Arabic, Franco-Arabic, English, and mixed text by category, urgency, and sentiment, then routes them to a support queue.

A fine-tuned encoder handles confident cases. Uncertain messages go to a Google ADK agent (Gemini and Anthropic, with DeepSeek as fallback), and anything still uncertain goes to a human review queue.

The design is in [architecture.md](architecture.md).

## Results and status

All model numbers come from **synthetic template data** (992 rows, split by template). They show the pipeline works, not how it performs on real complaints.

| Check | Result |
|---|---|
| Category macro-F1, MARBERT vs baseline | 0.66 vs 0.74 |
| High-urgency recall, MARBERT vs baseline | 0.90 vs 0.61 |
| Sentiment macro-F1, MARBERT vs baseline | 0.73 vs 0.65 |
| MARBERT category accuracy after int8 export | 0.87 |
| ONNX int8 size, CPU latency p50 / p95 | 163 MB, 9 / 14 ms |
| int8 agreement with PyTorch | category 93%, urgency 93%, sentiment 99.5% |
| Few-shot label precision@5 | 0.40 (RAGAS context precision 0.72, recall 0.89) |
| LightGBM urgency | Tied the encoder on simulated metadata, so it is not in serving |
| Service memory | About 1.2 GB before Whisper |

Verified by tests or a local run: the ADK Runner, failover to a stubbed DeepSeek API over real HTTP, circuit breaking, semantic cache hits and the negation guard, PII masking, idempotency under concurrency, shared rate limiting, hash partitions, streaming replication with dashboard reads from the replica, the review flow, MCP, voice transcription, and the React console.

Not verified yet: calls to the real Gemini, Anthropic, and DeepSeek APIs, the DeepEval suite (which needs those keys), and the Docker image, Nginx, Collector, Prometheus, and Compose replica scripts (which need Docker running). `docker compose config` passes.

## Layout

| Path | Contents |
|---|---|
| `src/triage/domain` | Labels, routing policy, normalization, PII masking |
| `src/triage/application` | Ports and use cases |
| `src/triage/adapters` | FastAPI, FastMCP, ONNX, LightGBM, ADK, Postgres, Whisper, OpenTelemetry |
| `src/triage/bootstrap.py` | Composition root |
| `training/` | Data generation, training, tuning, export, label quality, drift |
| `evals/` | DeepEval and RAGAS suites |
| `frontend/` | React review console and dashboard |
| `deploy/` | Nginx, Postgres replication, Collector, Prometheus |

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev,train,eval,voice]"
```

## Train

Run these from the repository root.

```bash
python -m training.generate_synthetic
python -m training.train_baseline
python -m training.train_encoder --epochs 4
python -m training.export_onnx
python -m training.train_urgency
```

The bundled data is synthetic and only proves the pipeline runs. Replace `data/synthetic.csv` with labeled tickets before trusting any metric.

The embedder goes in `artifacts/embedder/` as `model.onnx` and `tokenizer.json`, taken from the ONNX folder of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Without it, the service falls back to a lexical hashing embedder meant only for development.

## Run

Without Docker, using an embedded PostgreSQL:

```bash
PYTHONPATH=src:. ADMIN_API_KEY=choose-a-key python scripts/dev_server.py --seed data/synthetic.csv
```

With the full topology, copy `.env.example` to `.env`, fill it in, and start the full topology:

```bash
docker compose up --build --scale api=3
```

| Service | URL |
|---|---|
| Console and API (through Nginx) | http://localhost:8080 |
| Swagger | http://localhost:8080/docs |
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| MLflow | http://localhost:5000 |

Seed the few-shot store after the database is up:

```bash
DATABASE_URL=postgresql://triage:<POSTGRES_PASSWORD>@localhost:5432/triage python -m training.seed_examples
```

## Test

```bash
pytest tests -q
pytest evals/test_escalation_deepeval.py -m live   # calls paid providers
python -m evals.eval_retrieval_ragas
```

## API

```bash
curl -X POST localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "el net wa2e3 men embare7", "channel": "whatsapp", "external_id": "msg-1"}'
```

| Endpoint | Purpose | Auth |
|---|---|---|
| `POST /classify` | One message; add `?model=baseline` to compare | None |
| `POST /classify/batch` | Up to 64 messages | None |
| `POST /classify/voice` | Voice note, multipart field `audio` | None |
| `GET /review` | Pending tickets, masked text only | `X-API-Key` |
| `POST /review/{id}` | Human decision | `X-API-Key` |
| `GET /dashboard` | Aggregates, served from the read replica | `X-API-Key` |
| `GET /health`, `GET /metrics` | Health and Prometheus metrics | None |
| `/mcp` | MCP tool `classify_complaint` | None |

Add authentication before exposing `/classify` or `/mcp` publicly: they write tickets and can trigger paid LLM calls, limited only by the rate limits.
