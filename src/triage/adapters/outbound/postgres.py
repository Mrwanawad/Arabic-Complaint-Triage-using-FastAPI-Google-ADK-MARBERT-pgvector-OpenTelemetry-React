"""PostgreSQL adapters: tickets, semantic cache, few-shot examples, and the shared rate limiter.

Writes and read-your-writes queries use the primary. Dashboard aggregates use the read replica and
fall back to the primary when it is unavailable.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from triage.application.ports import CachedDecision, Dashboard
from triage.domain.models import (
    Category,
    DecidedBy,
    LabeledExample,
    Prediction,
    Script,
    Sentiment,
    Ticket,
    TicketStatus,
    Urgency,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_SCHEMA_LOCK = 7_231_904


async def apply_schema(database_url: str) -> None:
    """The advisory lock stops replicas that start together from racing on DDL."""
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as connection:
        await connection.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK,))
        try:
            await connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        finally:
            await connection.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK,))


async def _configure(connection: psycopg.AsyncConnection) -> None:
    await register_vector_async(connection)


def create_pool(url: str, max_size: int = 10) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        url,
        min_size=1,
        max_size=max_size,
        open=False,
        configure=_configure,
        kwargs={"row_factory": dict_row},
    )


def _prediction_to_json(prediction: Prediction) -> str:
    return json.dumps(
        {
            "category": prediction.category.value,
            "category_confidence": prediction.category_confidence,
            "urgency": prediction.urgency.value,
            "urgency_confidence": prediction.urgency_confidence,
            "sentiment": prediction.sentiment.value,
            "sentiment_confidence": prediction.sentiment_confidence,
            "model_version": prediction.model_version,
        }
    )


def _prediction_from_row(row: dict) -> Prediction:
    return Prediction(
        category=Category(row["category"]),
        category_confidence=float(row["category_confidence"]),
        urgency=Urgency(row["urgency"]),
        urgency_confidence=float(row["urgency_confidence"]),
        sentiment=Sentiment(row["sentiment"]),
        sentiment_confidence=float(row["sentiment_confidence"]),
        model_version=row["model_version"],
    )


def _ticket_from_row(row: dict) -> Ticket:
    return Ticket(
        ticket_id=row["ticket_id"],
        channel=row["channel"],
        customer_id=row["customer_id"],
        raw_text=row["raw_text"],
        masked_text=row["masked_text"],
        normalized_text=row["normalized_text"],
        script=Script(row["script"]),
        prediction=_prediction_from_row(row),
        decided_by=DecidedBy(row["decided_by"]),
        target_queue=row["target_queue"],
        status=TicketStatus(row["status"]),
        latency_ms=float(row["latency_ms"]),
        received_at=row["received_at"],
        review_reason=row["review_reason"],
    )


class PostgresTickets:
    def __init__(self, primary: AsyncConnectionPool, replica: AsyncConnectionPool | None = None):
        self._primary = primary
        self._replica = replica

    async def get(self, ticket_id: uuid.UUID) -> Ticket | None:
        async with self._primary.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = %s", (ticket_id,)
            )
            row = await cursor.fetchone()
        return _ticket_from_row(row) if row else None

    async def save(self, ticket: Ticket) -> Ticket:
        p = ticket.prediction
        async with self._primary.connection() as connection:
            cursor = await connection.execute(
                """
                INSERT INTO tickets (
                    ticket_id, channel, customer_id, raw_text, masked_text, normalized_text, script,
                    category, category_confidence, urgency, urgency_confidence,
                    sentiment, sentiment_confidence, model_version,
                    decided_by, target_queue, status, review_reason, latency_ms, received_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (ticket_id) DO NOTHING
                RETURNING *
                """,
                (
                    ticket.ticket_id,
                    ticket.channel,
                    ticket.customer_id,
                    ticket.raw_text,
                    ticket.masked_text,
                    ticket.normalized_text,
                    ticket.script.value,
                    p.category.value,
                    p.category_confidence,
                    p.urgency.value,
                    p.urgency_confidence,
                    p.sentiment.value,
                    p.sentiment_confidence,
                    p.model_version,
                    ticket.decided_by.value,
                    ticket.target_queue,
                    ticket.status.value,
                    ticket.review_reason,
                    ticket.latency_ms,
                    ticket.received_at,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                # A concurrent request with the same external id won the insert.
                cursor = await connection.execute(
                    "SELECT * FROM tickets WHERE ticket_id = %s", (ticket.ticket_id,)
                )
                row = await cursor.fetchone()
        return _ticket_from_row(row)

    async def repeat_complaints(self, customer_id: str, hours: int) -> int:
        async with self._primary.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT count(*) AS total FROM tickets
                WHERE customer_id = %s AND received_at > now() - make_interval(hours => %s)
                """,
                (customer_id, hours),
            )
            row = await cursor.fetchone()
        return int(row["total"])

    async def pending_reviews(self, limit: int) -> list[Ticket]:
        async with self._primary.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT * FROM tickets WHERE status = 'pending_review'
                ORDER BY urgency = 'high' DESC, received_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
        return [_ticket_from_row(row) for row in rows]

    async def apply_review(
        self,
        ticket_id: uuid.UUID,
        category: Category,
        urgency: Urgency,
        sentiment: Sentiment,
        target_queue: str,
        reviewer: str,
    ) -> Ticket | None:
        async with self._primary.connection() as connection:
            cursor = await connection.execute(
                """
                UPDATE tickets SET
                    overridden = (category <> %(category)s OR urgency <> %(urgency)s),
                    category = %(category)s, category_confidence = 1,
                    urgency = %(urgency)s, urgency_confidence = 1,
                    sentiment = %(sentiment)s, sentiment_confidence = 1,
                    decided_by = 'human', status = 'reviewed',
                    target_queue = %(queue)s, reviewer = %(reviewer)s
                WHERE ticket_id = %(ticket_id)s
                RETURNING *
                """,
                {
                    "category": category.value,
                    "urgency": urgency.value,
                    "sentiment": sentiment.value,
                    "queue": target_queue,
                    "reviewer": reviewer,
                    "ticket_id": ticket_id,
                },
            )
            row = await cursor.fetchone()
        return _ticket_from_row(row) if row else None

    async def dashboard(self, hours: int) -> Dashboard:
        if self._replica is not None:
            try:
                return await self._read_dashboard(self._replica, hours)
            except (psycopg.OperationalError, TimeoutError):
                pass
        return await self._read_dashboard(self._primary, hours)

    @staticmethod
    async def _read_dashboard(pool: AsyncConnectionPool, hours: int) -> Dashboard:
        window = {"hours": hours}
        where = "received_at > now() - make_interval(hours => %(hours)s)"
        async with pool.connection(timeout=5) as connection:

            async def grouped(column: str) -> dict[str, int]:
                cursor = await connection.execute(
                    f"SELECT {column} AS k, count(*) AS n FROM tickets WHERE {where} GROUP BY 1",
                    window,
                )
                return {row["k"]: row["n"] for row in await cursor.fetchall()}

            by_category = await grouped("category")
            by_urgency = await grouped("urgency")
            by_sentiment = await grouped("sentiment")
            by_decider = await grouped("decided_by")
            cursor = await connection.execute(
                f"""
                SELECT date_trunc('hour', received_at) AS hour, category, count(*) AS n,
                       count(*) FILTER (WHERE sentiment = 'negative') AS negative
                FROM tickets WHERE {where} GROUP BY 1, 2 ORDER BY 1
                """,
                window,
            )
            by_hour = [
                {
                    "hour": row["hour"].isoformat(),
                    "category": row["category"],
                    "count": row["n"],
                    "negative": row["negative"],
                }
                for row in await cursor.fetchall()
            ]
            cursor = await connection.execute(
                f"""
                SELECT count(*) FILTER (WHERE status = 'pending_review') AS pending,
                       count(*) FILTER (WHERE overridden) AS overridden,
                       count(*) FILTER (WHERE status = 'reviewed') AS reviewed
                FROM tickets WHERE {where}
                """,
                window,
            )
            totals = await cursor.fetchone()
        reviewed = totals["reviewed"]
        return Dashboard(
            total=sum(by_category.values()),
            by_category=by_category,
            by_urgency=by_urgency,
            by_sentiment=by_sentiment,
            by_decider=by_decider,
            by_hour=by_hour,
            pending_review=totals["pending"],
            override_rate=totals["overridden"] / reviewed if reviewed else 0.0,
        )


class PostgresSemanticCache:
    def __init__(
        self, pool: AsyncConnectionPool, threshold: float = 0.97, ttl_days: int = 7
    ) -> None:
        self._pool = pool
        self._threshold = threshold
        self._ttl = timedelta(days=ttl_days)

    async def lookup(
        self, embedding: list[float], negation: str, prompt_version: str
    ) -> CachedDecision | None:
        vector = np.asarray(embedding, dtype=np.float32)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT prediction, 1 - (embedding <=> %(v)s) AS similarity
                FROM semantic_cache
                WHERE negation = %(n)s AND prompt_version = %(p)s AND expires_at > now()
                ORDER BY embedding <=> %(v)s
                LIMIT 1
                """,
                {"v": vector, "n": negation, "p": prompt_version},
            )
            row = await cursor.fetchone()
        if row is None or row["similarity"] < self._threshold:
            return None
        return CachedDecision(_prediction_from_row(row["prediction"]), float(row["similarity"]))

    async def store(
        self,
        masked_text: str,
        embedding: list[float],
        negation: str,
        prompt_version: str,
        prediction: Prediction,
    ) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO semantic_cache
                    (masked_text, embedding, negation, prompt_version, prediction, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    masked_text,
                    np.asarray(embedding, dtype=np.float32),
                    negation,
                    prompt_version,
                    _prediction_to_json(prediction),
                    datetime.now(UTC) + self._ttl,
                ),
            )

    async def purge_expired(self) -> int:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM semantic_cache WHERE expires_at <= now()"
            )
            return cursor.rowcount


class PostgresExamples:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def similar(self, embedding: list[float], limit: int) -> list[LabeledExample]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT text, category, urgency, sentiment FROM few_shot_examples
                ORDER BY embedding <=> %s LIMIT %s
                """,
                (np.asarray(embedding, dtype=np.float32), limit),
            )
            rows = await cursor.fetchall()
        return [
            LabeledExample(
                row["text"],
                Category(row["category"]),
                Urgency(row["urgency"]),
                Sentiment(row["sentiment"]),
            )
            for row in rows
        ]

    async def add(self, example: LabeledExample, embedding: list[float]) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO few_shot_examples (text, category, urgency, sentiment, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (text) DO UPDATE SET
                    category = EXCLUDED.category,
                    urgency = EXCLUDED.urgency,
                    sentiment = EXCLUDED.sentiment
                """,
                (
                    example.text,
                    example.category.value,
                    example.urgency.value,
                    example.sentiment.value,
                    np.asarray(embedding, dtype=np.float32),
                ),
            )


class PostgresRateLimiter:
    """Fixed-window counter shared across every API replica."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def allow(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                INSERT INTO rate_limits AS r (key, window_start, count)
                VALUES (
                    %(key)s,
                    to_timestamp(floor(extract(epoch FROM now()) / %(window)s) * %(window)s),
                    1
                )
                ON CONFLICT (key, window_start) DO UPDATE SET count = r.count + 1
                RETURNING r.count
                """,
                {"key": key, "window": window_seconds},
            )
            row = await cursor.fetchone()
        return row["count"] <= limit

    async def purge_old(self) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                "DELETE FROM rate_limits WHERE window_start < now() - interval '1 hour'"
            )
