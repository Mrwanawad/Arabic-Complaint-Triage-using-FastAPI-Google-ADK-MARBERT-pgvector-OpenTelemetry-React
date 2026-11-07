CREATE EXTENSION IF NOT EXISTS vector;

-- Hash partitions on the deterministic ticket_id keep idempotency globally unique and give clean
-- shard boundaries if the table ever moves to Citus.
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id            uuid PRIMARY KEY,
    channel              text NOT NULL,
    customer_id          text,
    raw_text             text NOT NULL,
    masked_text          text NOT NULL,
    normalized_text      text NOT NULL,
    script               text NOT NULL,
    category             text NOT NULL,
    category_confidence  real NOT NULL,
    urgency              text NOT NULL,
    urgency_confidence   real NOT NULL,
    sentiment            text NOT NULL,
    sentiment_confidence real NOT NULL,
    model_version        text NOT NULL,
    decided_by           text NOT NULL,
    target_queue         text NOT NULL,
    status               text NOT NULL,
    review_reason        text,
    reviewer             text,
    overridden           boolean,
    latency_ms           real NOT NULL,
    received_at          timestamptz NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now()
) PARTITION BY HASH (ticket_id);

CREATE TABLE IF NOT EXISTS tickets_p0 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE IF NOT EXISTS tickets_p1 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE IF NOT EXISTS tickets_p2 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE IF NOT EXISTS tickets_p3 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE IF NOT EXISTS tickets_p4 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE IF NOT EXISTS tickets_p5 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE IF NOT EXISTS tickets_p6 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE IF NOT EXISTS tickets_p7 PARTITION OF tickets FOR VALUES WITH (MODULUS 8, REMAINDER 7);

CREATE INDEX IF NOT EXISTS tickets_pending_idx ON tickets (received_at) WHERE status = 'pending_review';
CREATE INDEX IF NOT EXISTS tickets_customer_idx ON tickets (customer_id, received_at) WHERE customer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS tickets_received_idx ON tickets (received_at);

CREATE TABLE IF NOT EXISTS semantic_cache (
    id             bigserial PRIMARY KEY,
    masked_text    text NOT NULL,
    embedding      vector(384) NOT NULL,
    negation       text NOT NULL,
    prompt_version text NOT NULL,
    prediction     jsonb NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    expires_at     timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS semantic_cache_hnsw ON semantic_cache USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS semantic_cache_expiry ON semantic_cache (expires_at);

CREATE TABLE IF NOT EXISTS few_shot_examples (
    id         bigserial PRIMARY KEY,
    text       text NOT NULL UNIQUE,
    category   text NOT NULL,
    urgency    text NOT NULL,
    sentiment  text NOT NULL,
    embedding  vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS few_shot_hnsw ON few_shot_examples USING hnsw (embedding vector_cosine_ops);

-- UNLOGGED: counters are disposable, so they skip WAL. They are not replicated and reset on crash,
-- which only forgives a little budget for one window.
CREATE UNLOGGED TABLE IF NOT EXISTS rate_limits (
    key          text NOT NULL,
    window_start timestamptz NOT NULL,
    count        integer NOT NULL,
    PRIMARY KEY (key, window_start)
);
