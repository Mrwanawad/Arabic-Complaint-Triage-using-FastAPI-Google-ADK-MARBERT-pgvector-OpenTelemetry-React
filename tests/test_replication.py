"""Real streaming replication between two local PostgreSQL servers, using pgserver's binaries."""

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import psycopg
import pytest

pgserver = pytest.importorskip("pgserver")

from tests.test_postgres import ticket
from triage.adapters.outbound.postgres import PostgresTickets, apply_schema, create_pool

pytestmark = pytest.mark.postgres

BIN = Path(pgserver.__file__).parent / "pginstall" / "bin"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def tool(name: str) -> str:
    return str(BIN / (f"{name}.exe" if sys.platform == "win32" else name))


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start(data: Path, port: int) -> None:
    options = f"-p {port} -c listen_addresses=127.0.0.1"
    subprocess.run(
        [
            tool("pg_ctl"),
            "-D",
            str(data),
            "-o",
            options,
            "-l",
            str(data / "log.txt"),
            "-w",
            "start",
        ],
        check=True,
        # The server inherits these handles, so capturing them would block until it exits.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop(data: Path) -> None:
    subprocess.run(
        [tool("pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    root = tmp_path_factory.mktemp("cluster")
    primary_dir, replica_dir = root / "primary", root / "replica"
    primary_port, replica_port = free_port(), free_port()

    subprocess.run(
        [tool("initdb"), "-D", str(primary_dir), "-U", "postgres", "-A", "trust", "-E", "UTF8"],
        check=True,
        capture_output=True,
    )
    with (primary_dir / "postgresql.conf").open("a") as config:
        config.write(
            "\nwal_level = replica\nmax_wal_senders = 5\nmax_replication_slots = 5\n"
            "unix_socket_directories = ''\n"
        )
    with (primary_dir / "pg_hba.conf").open("a") as hba:
        hba.write("\nhost replication postgres 127.0.0.1/32 trust\n")
    start(primary_dir, primary_port)
    subprocess.run(
        [
            tool("pg_basebackup"),
            "-h", "127.0.0.1",
            "-p", str(primary_port),
            "-U", "postgres",
            "-D", str(replica_dir),
            "--wal-method=stream",
            "--write-recovery-conf",
            "--create-slot",
            "--slot=replica_1",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    start(replica_dir, replica_port)

    primary_url = f"postgresql://postgres@127.0.0.1:{primary_port}/postgres"
    replica_url = f"postgresql://postgres@127.0.0.1:{replica_port}/postgres"
    asyncio.run(apply_schema(primary_url))
    yield {"primary": primary_url, "replica": replica_url, "primary_dir": primary_dir}
    stop(replica_dir)
    stop(primary_dir)


def wait_for_row(replica_url: str, ticket_id) -> bool:
    for _ in range(100):
        try:
            with psycopg.connect(replica_url) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM tickets WHERE ticket_id = %s", (ticket_id,)
                ).fetchone()
            if exists:
                return True
        except psycopg.errors.UndefinedTable:
            # The schema itself is still replicating.
            pass
        time.sleep(0.1)
    return False


def test_replica_is_hot_standby_and_rejects_writes(cluster):
    with psycopg.connect(cluster["replica"]) as connection:
        assert connection.execute("SELECT pg_is_in_recovery()").fetchone()[0] is True
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            connection.execute("CREATE TABLE should_fail (id int)")


def test_writes_replicate_including_partitions_and_vectors(cluster):
    item = ticket()

    async def write():
        pool = create_pool(cluster["primary"], 2)
        await pool.open(wait=True)
        try:
            await PostgresTickets(pool).save(item)
        finally:
            await pool.close()

    asyncio.run(write())
    assert wait_for_row(cluster["replica"], item.ticket_id)


def test_dashboard_reads_from_replica_after_primary_stops(cluster):
    """Must run last in this module: it stops the primary. Only the replica can answer now."""
    item = ticket()

    async def scenario():
        primary = create_pool(cluster["primary"], 2)
        replica = create_pool(cluster["replica"], 2)
        await primary.open(wait=True)
        await replica.open(wait=True)
        try:
            repo = PostgresTickets(primary, replica)
            await repo.save(item)
            assert await asyncio.to_thread(wait_for_row, cluster["replica"], item.ticket_id)
            await asyncio.to_thread(stop, cluster["primary_dir"])
            return await repo.dashboard(24)
        finally:
            await primary.close()
            await replica.close()

    assert asyncio.run(scenario()).total >= 1
