"""Run the API locally with an embedded PostgreSQL (pgserver), so Docker is not needed.

Usage, from the repository root with PYTHONPATH=src;. (use : on Linux):
    ADMIN_API_KEY=... python scripts/dev_server.py --port 8000 --seed data/synthetic.csv

Stop it with Ctrl+C. A forced kill leaves PostgreSQL running, which blocks the next start until it
is stopped with pg_ctl.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pgserver
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=Path(".devdata/postgres"))
    parser.add_argument("--seed", type=Path, help="CSV to load into the few-shot store first")
    args = parser.parse_args()

    if sys.platform == "win32":
        # psycopg's async mode does not support the default Windows ProactorEventLoop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args.data_dir.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(args.data_dir, cleanup_mode="stop")
    os.environ["DATABASE_URL"] = server.get_uri()
    try:
        if args.seed:
            from training.seed_examples import seed
            from triage.settings import Settings

            print(f"Seeded {asyncio.run(seed(args.seed, Settings()))} few-shot examples")
        uvicorn.run("triage.main:app", host="127.0.0.1", port=args.port, loop="none")
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()
