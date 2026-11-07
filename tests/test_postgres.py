"""Integration tests against a real PostgreSQL with pgvector, provided by the pgserver package."""

import asyncio
import dataclasses
import sys
import uuid
from datetime import UTC, datetime

import pytest

pgserver = pytest.importorskip("pgserver")

from tests.fakes import make_prediction
from triage.adapters.outbound.postgres import (
    PostgresExamples,
    PostgresRateLimiter,
    PostgresSemanticCache,
    PostgresTickets,
    apply_schema,
    create_pool,
)
from triage.domain.models import (
    Category,
    DecidedBy,
    LabeledExample,
    Script,
    Sentiment,
    Ticket,
    TicketStatus,
    Urgency,
)

pytestmark = pytest.mark.postgres

if sys.platform == "win32":
    # psycopg's async mode does not support the default Windows ProactorEventLoop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="module")
def database_url(tmp_path_factory):
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    url = server.get_uri()
    asyncio.run(apply_schema(url))
    yield url
    server.cleanup()


def with_pool(url, work):
    async def runner():
        pool = create_pool(url, 4)
        await pool.open(wait=True)
        try:
            return await work(pool)
        finally:
            await pool.close()

    return asyncio.run(runner())


def ticket(**overrides) -> Ticket:
    values = {
        "ticket_id": uuid.uuid4(),
        "channel": "api",
        "customer_id": None,
        "raw_text": "النت واقع 01012345678",
        "masked_text": "النت واقع [phone]",
        "normalized_text": "النت واقع [phone]",
        "script": Script.ARABIC,
        "prediction": make_prediction(),
        "decided_by": DecidedBy.ENCODER,
        "target_queue": "noc_priority",
        "status": TicketStatus.ROUTED,
        "latency_ms": 12.5,
        "received_at": datetime.now(UTC),
    }
    values.update(overrides)
    return Ticket(**values)


def unit(*values: float) -> list[float]:
    vector = list(values) + [0.0] * (384 - len(values))
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


def test_schema_apply_is_idempotent(database_url):
    asyncio.run(apply_schema(database_url))


def test_tickets_are_hash_partitioned(database_url):
    async def work(pool):
        async with pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT count(*) AS n FROM pg_inherits WHERE inhparent = 'tickets'::regclass"
            )
            return (await cursor.fetchone())["n"]

    assert with_pool(database_url, work) == 8


def test_ticket_roundtrip(database_url):
    original = ticket()

    async def work(pool):
        repo = PostgresTickets(pool)
        await repo.save(original)
        return await repo.get(original.ticket_id)

    loaded = with_pool(database_url, work)
    assert loaded.prediction.category is Category.OUTAGE
    assert loaded.masked_text == original.masked_text
    assert loaded.decided_by is DecidedBy.ENCODER


def test_duplicate_save_keeps_first(database_url):
    first = ticket()
    second = dataclasses.replace(first, target_queue="other")

    async def work(pool):
        repo = PostgresTickets(pool)
        await repo.save(first)
        return await repo.save(second)

    assert with_pool(database_url, work).target_queue == "noc_priority"


def test_concurrent_duplicates_store_one_row(database_url):
    item = ticket()

    async def work(pool):
        repo = PostgresTickets(pool)
        results = await asyncio.gather(*(repo.save(item) for _ in range(8)))
        async with pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT count(*) AS n FROM tickets WHERE ticket_id = %s", (item.ticket_id,)
            )
            return results, (await cursor.fetchone())["n"]

    results, count = with_pool(database_url, work)
    assert count == 1
    assert {r.ticket_id for r in results} == {item.ticket_id}


def test_repeat_complaints_counts_customer(database_url):
    customer = uuid.uuid4().hex

    async def work(pool):
        repo = PostgresTickets(pool)
        await repo.save(ticket(customer_id=customer))
        await repo.save(ticket(customer_id=customer))
        return await repo.repeat_complaints(customer, 24)

    assert with_pool(database_url, work) == 2


def test_review_flow_and_override_rate(database_url):
    pending = ticket(
        status=TicketStatus.PENDING_REVIEW, target_queue="human_review", review_reason="x"
    )

    async def work(pool):
        repo = PostgresTickets(pool)
        await repo.save(pending)
        queue = await repo.pending_reviews(500)
        updated = await repo.apply_review(
            pending.ticket_id,
            Category.SPEED,
            Urgency.NORMAL,
            Sentiment.NEUTRAL,
            "technical_support",
            "agent-1",
        )
        return queue, updated, await repo.dashboard(24)

    queue, updated, dashboard = with_pool(database_url, work)
    assert pending.ticket_id in {t.ticket_id for t in queue}
    assert updated.status is TicketStatus.REVIEWED
    assert updated.decided_by is DecidedBy.HUMAN
    assert updated.prediction.category is Category.SPEED
    assert dashboard.override_rate > 0
    assert dashboard.total >= 1
    assert dashboard.by_hour


def test_review_unknown_ticket_returns_none(database_url):
    async def work(pool):
        return await PostgresTickets(pool).apply_review(
            uuid.uuid4(), Category.OTHER, Urgency.NORMAL, Sentiment.NEUTRAL, "q", "a"
        )

    assert with_pool(database_url, work) is None


def test_dashboard_falls_back_when_replica_is_down(database_url):
    async def work(pool):
        dead = create_pool("postgresql://nobody@127.0.0.1:1/none", 1)
        await dead.open(wait=False)
        try:
            return await PostgresTickets(pool, dead).dashboard(24)
        finally:
            await dead.close()

    assert with_pool(database_url, work).total >= 0


def test_semantic_cache_hit_requires_similarity(database_url):
    tag = uuid.uuid4().hex

    async def work(pool):
        cache = PostgresSemanticCache(pool, threshold=0.97)
        prediction = make_prediction(0.9, Category.BILLING)
        await cache.store("x", unit(1, 0), "0", tag, prediction)
        return (
            await cache.lookup(unit(1, 0.01), "0", tag),
            await cache.lookup(unit(0, 1), "0", tag),
        )

    near, far = with_pool(database_url, work)
    assert near.prediction.category is Category.BILLING
    assert near.similarity > 0.97
    assert far is None


def test_semantic_cache_scoped_by_negation_and_prompt(database_url):
    tag = uuid.uuid4().hex

    async def work(pool):
        cache = PostgresSemanticCache(pool)
        await cache.store("x", unit(1, 2), "0", tag, make_prediction(0.9))
        return (
            await cache.lookup(unit(1, 2), "1", tag),
            await cache.lookup(unit(1, 2), "0", "other-version"),
            await cache.lookup(unit(1, 2), "0", tag),
        )

    other_negation, other_prompt, same = with_pool(database_url, work)
    assert other_negation is None
    assert other_prompt is None
    assert same is not None


def test_semantic_cache_expiry(database_url):
    tag = uuid.uuid4().hex

    async def work(pool):
        cache = PostgresSemanticCache(pool, ttl_days=-1)
        await cache.store("x", unit(3, 1), "0", tag, make_prediction(0.9))
        missed = await cache.lookup(unit(3, 1), "0", tag)
        return missed, await cache.purge_expired()

    missed, purged = with_pool(database_url, work)
    assert missed is None
    assert purged >= 1


def test_examples_return_nearest_and_upsert(database_url):
    marker = uuid.uuid4().hex

    async def work(pool):
        store = PostgresExamples(pool)
        near = LabeledExample(f"near {marker}", Category.OUTAGE, Urgency.HIGH, Sentiment.NEGATIVE)
        await store.add(near, unit(5, 1, 1))
        await store.add(dataclasses.replace(near, category=Category.SPEED), unit(5, 1, 1))
        return await store.similar(unit(5, 1, 1), 1)

    found = with_pool(database_url, work)
    assert found[0].text == f"near {marker}"
    assert found[0].category is Category.SPEED


def test_rate_limiter_is_atomic_under_concurrency(database_url):
    key = uuid.uuid4().hex

    async def work(pool):
        limiter = PostgresRateLimiter(pool)
        return await asyncio.gather(*(limiter.allow(key, 5, 3600) for _ in range(20)))

    results = with_pool(database_url, work)
    assert sum(results) == 5
