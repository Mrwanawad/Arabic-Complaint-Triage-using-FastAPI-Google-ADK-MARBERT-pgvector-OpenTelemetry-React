#!/bin/bash
# Runs once when the primary data directory is first created.
set -euo pipefail

psql -v ON_ERROR_STOP=1 -v repl_password="$REPLICATION_PASSWORD" \
  --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD :'repl_password';
SQL

echo "host replication replicator all scram-sha-256" >> "$PGDATA/pg_hba.conf"
