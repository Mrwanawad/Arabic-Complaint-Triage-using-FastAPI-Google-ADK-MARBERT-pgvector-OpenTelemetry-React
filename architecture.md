# Arabic Customer-Complaint Triage (Telecom Support) — Architecture

## Scenario
A telecom/ISP support center receives complaints in Egyptian Arabic, Franco-Arabic ("el net wa2e3"), and mixed Arabic/English through tickets, WhatsApp, and social media, as text or voice notes. Agents triage manually, so urgent outages wait behind billing questions. The system classifies each message by category and urgency, scores sentiment, and routes it to the right queue in near real time.

A fine-tuned encoder handles most traffic. Only low-confidence messages go to an LLM, and anything still uncertain goes to a human.

## Input
- Raw complaint text (any script: Arabic, Franco-Arabic, English, mixed), or a voice note.
- Metadata: channel, external_id, timestamp, customer_id (optional).
- Labels for training: category (billing, outage, speed, installation, other), urgency (high/normal), sentiment (negative/neutral/positive).

## Output
- Per message JSON: `ticket_id`, `category`, `category_confidence`, `urgency`, `sentiment`, `normalized_text`, `decided_by` (encoder / baseline / llm / cache / human), `model_version`, `latency_ms`.
- Routing decision: target queue; low-confidence messages flagged for human review.
- Aggregates for dashboard: complaint volume by category/hour, negative-sentiment spikes.

## Application Architecture — Hexagonal (Ports & Adapters)
Core logic depends only on interfaces (ports). Frameworks and vendors live in adapters. Dependencies always point inward.

```
src/triage/
  domain/         labels, entities, routing policy, normalization, PII masking. No framework imports.
  application/
    ports.py      Classifier, Escalator, Embedder, TicketRepository, SemanticCache,
                  RateLimiter, Transcriber, UrgencyModel
    use_cases/    classify_complaint, review_ticket, ingest_voice_note, get_dashboard
  adapters/
    inbound/      http (FastAPI), mcp (FastMCP)
    outbound/     onnx_classifier, baseline_classifier, lightgbm_urgency, onnx_embedder,
                  adk_escalator, postgres, whisper, telemetry
  bootstrap.py    composition root: builds adapters from settings, injects into use cases
training/         offline only: data, labeling, baseline, fine-tuning, tuning, ONNX export
evals/            DeepEval and RAGAS suites
frontend/         React + TypeScript review console and dashboard
deploy/           Nginx, Postgres replication, OpenTelemetry Collector, Prometheus
```

Ports exist only at real boundaries. There is no per-provider adapter: the ADK adapter owns every LLM provider.

## Request Flow

```
 Channels (tickets / WhatsApp / social / voice notes)
        |
        v
 [Nginx]  least_conn load balancing, edge rate limit per client
        |
        v
 [FastAPI replicas]  stateless; REST + FastMCP share the same use cases
        |
        |  voice note -> [Whisper] -> text
        v
 [Idempotency]  ticket_id = UUIDv5(channel, external_id); replays return the stored result
        |
        v
 [Text Normalization]
   - script detection: Arabic / Franco-Arabic / English / mixed
   - Arabic: unify alef forms, ya/alef maqsura, ta marbuta; strip diacritics & tatweel
   - collapse repeated letters, handle emojis (map to tokens), URLs, phone numbers
   - Franco-Arabic: digit-letter mapping (2,3,5,7) + transliteration rules
        |
        v
 [Encoder]  MARBERT ONNX int8 (multi-head: category, urgency, sentiment)
        |   TF-IDF + LinearSVC stays available with ?model=baseline
        |
        +-- confidence >= threshold ---------------------------+
        |                                                      |
        v                                                      |
 [PII Masking]  phones, national IDs, emails, customer IDs     |
        |       always before any external provider call       |
        v                                                      |
 [Semantic Cache]  pgvector; hit -> reuse LLM decision --------+
        |                                                      |
        v                                                      |
 [Few-Shot Retrieval]  similar labeled complaints (pgvector)   |
        |                                                      |
        v                                                      |
 [Google ADK Agent]  structured output, schema validation      |
        |   Provider router:                                   |
        |     primaries: Gemini, Anthropic (weighted)          |
        |     fallback:  DeepSeek                              |
        |     circuit breaker + shared Postgres rate limit     |
        |                                                      |
        +-- LLM confident --> write semantic cache ------------+
        |                                                      |
        v                                                      v
 [Human Review Queue]  durable in Postgres          [LightGBM Urgency]
        |  all providers failed or LLM unsure       [Routing Policy] -> target queue
        |                                                      |
        +-- override --> relabeled data --> retraining         v
                                                      [Postgres primary]
                                                               |
                                                               v
                                                      [Read replica] -> dashboard

 Every step emits OpenTelemetry traces and metrics -> Collector
   -> Prometheus + Jaeger, and Langfuse over OTLP when configured
```

Offline loop: `Label Studio -> cleanlab flags noisy labels -> relabel -> baseline and MARBERT training (MLflow, Optuna) -> ONNX export + dynamic quantization -> evals -> release`.

## System Design

### Why no Redis
OpenTelemetry observes the system but stores no application state, so it cannot take over Redis's work directly. Each job Redis would have done has a new home:

| Job | Solution |
|---|---|
| Semantic cache | `semantic_cache` table with a pgvector HNSW index |
| Shared LLM rate limit | Atomic fixed-window counter in an UNLOGGED Postgres table |
| Edge rate limit | Nginx `limit_req` |
| Webhook deduplication | Primary key on the deterministic `ticket_id` |

Cost: every LLM-bound request gains Postgres round trips. That is acceptable because this path already waits on an LLM. High-confidence encoder traffic never touches these tables.

### Semantic caching
- Scope: only LLM decisions, keyed by embedding, `prompt_version`, and label-schema version. Changing either invalidates old entries.
- Stored text is masked first, so the cache holds no raw PII.
- Threshold: cosine similarity >= 0.97. Arabic negation is the main risk: "النت شغال" and "النت مش شغال" embed closely but mean opposite things. A hit also requires identical negation markers. The threshold must be validated on real data.
- TTL: 7 days, so outage-wave answers do not outlive the event.

### Rate limiting
- Edge: Nginx `limit_req` per client IP, returning 429.
- Application: per-provider requests-per-minute in Postgres, shared across all replicas. When a provider is over budget, the router moves to the next one. When all are over budget, the ticket goes to human review, so no complaint is dropped.

### Load balancing
- API replicas: Nginx `least_conn`, because inference times vary. Replicas are stateless. Health checks use `GET /health`.
- LLM providers: weighted selection between Gemini and Anthropic; DeepSeek receives traffic only when both are unavailable. A circuit breaker opens after repeated failures and retries after a cooldown.

### Replicas and sharding
- API: horizontal replicas behind Nginx.
- Postgres: one primary plus one streaming read replica. Dashboards and aggregates read from the replica, so reporting cannot slow triage. Review actions and idempotency checks stay on the primary because they need read-your-writes. Replica lag makes dashboards slightly stale, which is acceptable.
- Partitioning: `tickets` is hash-partitioned into 8 partitions on `ticket_id`. This keeps global uniqueness for idempotency and gives clean shard boundaries.
- Multi-node sharding is deliberately not built. One Postgres node comfortably handles a single support center's write volume, and cross-shard aggregates and HNSW search would add heavy operational cost. If writes ever outgrow vertical scaling, the 8 hash partitions can move to Citus using `ticket_id` as the distribution key.

### Failure behavior
| Failure | Behavior |
|---|---|
| One LLM provider down | Circuit opens, traffic moves to the next provider |
| All LLM providers down | Encoder result kept, ticket flagged for human review |
| Replica down | Dashboard falls back to the primary |
| Primary down | Writes fail with 503; manual promotion (no automatic failover in scope) |
| Telemetry backend down | Telemetry dropped; triage unaffected |

## Tools Used
- Data & EDA: pandas, NumPy, Matplotlib, Seaborn, Plotly Express, wordcloud (arabic-reshaper + python-bidi for rendering)
- Arabic NLP: rule-based normalization in the domain layer (standard-library regex)
- Labeling & label quality: cleanlab, with flagged rows exported as Label Studio task JSON
- Classical ML: scikit-learn (TfidfVectorizer, calibrated LinearSVC, Pipeline), LightGBM (urgency from encoder probabilities plus channel, hour, repeat-complaint count; gated, see Evaluation Metrics)
- Deep learning: PyTorch, Hugging Face Transformers
- Imbalance: class weights, stratified splits, weighted loss, imbalanced-learn (baseline resampling comparison)
- Tuning: Optuna
- Explainability: LIME for sample predictions, run against the baseline
- LLM orchestration: Google ADK, LiteLLM (ADK's connector for non-Gemini providers)
- LLM providers: Gemini and Anthropic (primary), DeepSeek (fallback)
- Retrieval: SentenceTransformers multilingual model (the published ONNX int8 export), PGVector
- LLM evaluation: DeepEval (escalation regressions), RAGAS (few-shot retrieval quality)
- Speech: Whisper (faster-whisper) for voice notes
- Agent interoperability: FastMCP
- Storage: PostgreSQL + pgvector
- Observability: OpenTelemetry, OpenTelemetry Collector, Prometheus, Jaeger, Langfuse (over OTLP), Evidently (text drift)
- Traffic: Nginx
- MLOps: MLflow, Docker, Docker Compose, GitHub Actions
- Serving: ONNX Runtime, FastAPI, Pydantic
- Frontend: React, TypeScript, Vite, Plotly
- Quality: pytest, ruff

## Tools Not Used (and why)
- Redis: removed by design; see System Design.
- LangChain / LangGraph / CrewAI / AutoGen / PydanticAI: Google ADK is the single orchestration framework.
- Supabase: plain PostgreSQL + pgvector, self-managed for replication control.
- LLM for every message: a fine-tuned encoder is cheaper, faster, and private. LLMs handle the uncertain minority.
- Weights & Biases: duplicates MLflow.
- Kubernetes / Terraform / SageMaker: disproportionate for this scope; Compose demonstrates the topology.
- Temporal: no long-running workflow; the review queue is durable in Postgres.
- LoRA / QLoRA: encoder fine-tuning already fits the task.
- ElevenLabs: the system does not speak.
- Generic multilingual models (mBERT, XLM-R) as main model: Arabic-specific encoders handle dialect better; kept only as optional comparison.
- Stemming-heavy pipelines (ISRIStemmer): hurts dialect text; normalization is enough.
- Spark / distributed training: single GPU fine-tuning is enough.
- Kafka: near-real-time via API is sufficient for the scope.

## Evaluation Metrics
- Primary: macro-F1 (rare classes like "outage" matter as much as common ones).
- Per-class precision, recall, F1; confusion matrix.
- Urgency: recall on high-urgency class (missing an outage is costly). LightGBM joins the path only if it strictly improves on the encoder urgency head without losing recall or macro-F1, measured on real metadata.
- Sentiment: macro-F1.
- Robustness slices: Arabic-only vs. Franco-Arabic vs. mixed text.
- Serving: p50 / p95 latency, throughput; ONNX vs. PyTorch accuracy delta.
- LLM path: escalation accuracy (DeepEval); schema-valid rate, fallback rate, and token usage per provider (metrics). Cost is derived from tokens and each provider's pricing.
- Retrieval: context precision and context recall for few-shot examples (RAGAS), read alongside exact label precision@k because the non-LLM RAGAS metrics use fuzzy string matching.
- Cache: hit rate and wrong-hit rate on a held-out set, including negation pairs.
- Business: % auto-routed, % escalated to LLM, human-override rate. `training/evaluate_routing.py` shows auto-routing against accuracy per threshold. Time-to-first-response needs data from the ticketing system, so it is not measured here.

## Production Monitoring
- Category distribution shift (new outage wave, new billing issue).
- Mean confidence trend and low-confidence rate.
- Unknown-token / new slang rate.
- Human-override rate as proxy label; overridden samples feed the retraining set.
- LLM calls, tokens, errors, failovers, and providers skipped by the circuit breaker or budget.
- Semantic cache hit rate; rate-limit rejections; replica lag.
- Latency and error rate alerts.

## Security & Privacy
- PII masking runs in the domain layer before any provider call or cache write. Raw text stays inside our Postgres.
- Masking is regex-based (Egyptian mobiles, national IDs, emails, customer IDs). It will miss names in free text, so a production rollout needs a data-processing agreement with each provider or a stronger NER step.
- API keys come from environment variables only and are never logged. Traces record masked text only.
- Review and admin endpoints require an API key.

## Deployment Plan

### Full topology — Docker Compose (local / any VM)
Nginx, N API replicas, Postgres primary and read replica, OpenTelemetry Collector, Prometheus, Jaeger, MLflow, and the React console. This is the complete system design and runs with `docker compose up --scale api=3`.

### Public demo — Render (Docker web service)
**Why Render:** the core deliverable is an API that other systems call. Render runs the Docker image as a real web service with a public HTTPS URL, health checks, and auto-deploy from GitHub.

**What gets deployed:**
- One Docker service: `POST /classify`, `POST /classify/batch`, `POST /classify/voice`, `GET /review`, `POST /review/{id}`, `GET /dashboard`, `GET /health`, `GET /metrics`, `GET /docs`, and `/mcp`.
- The React console served as static files at `/`: RTL text box, sample messages, result card, review queue, dashboard.
- Render Postgres with pgvector.

**Model packaging:**
- Fine-tuned MARBERT -> ONNX -> dynamic int8 quantization.
- Model bundle downloaded at startup from `ARTIFACTS_URL` and verified against `ARTIFACTS_SHA256`.
- No torch at runtime.

**Setup:**
- `Dockerfile` (python-slim, onnxruntime CPU), `render.yaml` blueprint with health check path `/health`.

**Limits to know:**
- The service measured about 1.2 GB resident memory with encoder, baseline, embedder, ADK, and LiteLLM loaded, before Whisper. Render's 512 MB free instance cannot run it, so the blueprint uses a paid 2 GB instance. Keep `TRANSCRIBER_ENABLED=false` there because Whisper needs more memory.
- Render does not show the Nginx/replica topology on a single instance. Use the Compose setup to demonstrate it.
