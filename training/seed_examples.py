"""Load labeled training rows into the few-shot store. Only the training split is used, so evals
measure retrieval on complaints the store has not seen.

Usage: python -m training.seed_examples --data data/synthetic.csv
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from training.common import load_dataset, split
from triage.adapters.outbound.postgres import PostgresExamples, apply_schema, create_pool
from triage.bootstrap import build_embedder
from triage.domain.models import Category, LabeledExample, Sentiment, Urgency
from triage.domain.pii import mask_pii
from triage.settings import Settings


async def seed(data: Path, settings: Settings) -> int:
    train, _ = split(load_dataset(data))
    embedder = build_embedder(settings)
    await apply_schema(settings.database_url)
    pool = create_pool(settings.database_url, 2)
    await pool.open(wait=True)
    try:
        store = PostgresExamples(pool)
        for row in train.itertuples():
            example = LabeledExample(
                mask_pii(row.text),
                Category(row.category),
                Urgency(row.urgency),
                Sentiment(row.sentiment),
            )
            await store.add(example, embedder.embed(row.normalized))
    finally:
        await pool.close()
    return len(train)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"Seeded {asyncio.run(seed(args.data, Settings()))} examples")


if __name__ == "__main__":
    main()
