#!/bin/bash
# Clones the primary on first start, then runs as a hot standby.
set -euo pipefail

export PGDATA=/var/lib/postgresql/data

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  until pg_isready -h postgres-primary -U replicator; do sleep 1; done
  rm -rf "${PGDATA:?}"/*
  pg_basebackup -h postgres-primary -U replicator -D "$PGDATA" \
    --wal-method=stream --write-recovery-conf --create-slot --slot=replica_1 --progress
  chmod 700 "$PGDATA"
fi

exec postgres -c hot_standby=on
