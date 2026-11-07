#!/usr/bin/env bash
# Checks the running Compose topology end to end. Start it first:
#   docker compose up -d --build --wait --scale api=3 nginx api postgres-primary postgres-replica otel-collector jaeger prometheus
# Requires ADMIN_API_KEY to match the stack.
set -euo pipefail

BASE=http://localhost:8080
failures=0

check() {
  local name=$1 actual=$2 expected=$3
  if [[ "$actual" == *"$expected"* ]]; then
    echo "PASS  $name"
  else
    echo "FAIL  $name: expected '$expected', got '$actual'"
    failures=$((failures + 1))
  fi
}

at_least() {
  local name=$1 actual=$2 minimum=$3
  if [[ "$actual" =~ ^[0-9]+$ ]] && ((actual >= minimum)); then
    echo "PASS  $name ($actual)"
  else
    echo "FAIL  $name: expected at least $minimum, got '$actual'"
    failures=$((failures + 1))
  fi
}

classify() {
  curl -s -X POST "$BASE/classify" -H 'Content-Type: application/json' \
    -d "{\"text\": \"النت واقع من امبارح ورقمي 01012345678\", \"channel\": \"tickets\", \"external_id\": \"$1\"}"
}

replica_sql() {
  docker compose exec -T postgres-replica psql -U triage -d triage -tAc "$1"
}

check "health through Nginx" "$(curl -s "$BASE/health")" '"status":"ok"'
check "console served" "$(curl -s "$BASE/")" "<title>Complaint Triage</title>"
check "metrics hidden at the edge" "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/metrics")" "404"

first=$(classify smoke-1)
check "classify returns a ticket" "$first" '"ticket_id"'
check "PII masked in response" "$first" "[phone]"
if [[ "$first" == *"01012345678"* ]]; then
  echo "FAIL  raw phone number returned"
  failures=$((failures + 1))
fi
first_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["ticket_id"])' "$first")
check "replay is idempotent" "$(classify smoke-1)" "$first_id"

check "admin route rejects missing key" "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/dashboard")" "401"
check "dashboard works with key" \
  "$(curl -s -H "X-API-Key: $ADMIN_API_KEY" "$BASE/dashboard")" '"total"'

# Spaced out so the edge limiter stays out of the way of this check.
for i in $(seq 1 30); do
  classify "balance-$i" >/dev/null
  sleep 0.1
done
serving=$(docker compose logs --no-color api 2>/dev/null |
  grep 'POST /classify' | cut -d' ' -f1 | sort -u | wc -l || true)
at_least "replicas that served traffic" "$serving" 2

check "replica is a hot standby" "$(replica_sql 'SELECT pg_is_in_recovery()')" "t"
for _ in $(seq 1 30); do
  replicated=$(replica_sql "SELECT count(*) FROM tickets" 2>/dev/null || echo 0)
  ((replicated >= 31)) && break
  sleep 1
done
at_least "tickets replicated to standby" "$replicated" 31
check "standby rejects writes" \
  "$(replica_sql 'CREATE TABLE should_fail (id int)' 2>&1 || true)" "read-only transaction"
check "tickets are hash partitioned" \
  "$(replica_sql "SELECT count(*) FROM pg_inherits WHERE inhparent = 'tickets'::regclass")" "8"

codes=$(seq 1 300 | xargs -P 60 -I{} curl -s -o /dev/null -w '%{http_code}\n' \
  -X POST "$BASE/classify" -H 'Content-Type: application/json' \
  -d '{"text": "النت بطيء", "channel": "tickets", "external_id": "burst-{}"}')
at_least "edge rate limit returned 429" "$(grep -c '^429$' <<<"$codes" || true)" 1
at_least "requests still served during burst" "$(grep -c '^200$' <<<"$codes" || true)" 1
server_errors=$(grep -c '^5' <<<"$codes" || true)
if ((server_errors > 0)); then
  echo "FAIL  no server errors during burst: got $server_errors"
  failures=$((failures + 1))
else
  echo "PASS  no server errors during burst"
fi

# Telemetry export is batched and scraping runs every 15 seconds.
sleep 25
at_least "Prometheus sees API replicas" "$(curl -s http://localhost:9090/api/v1/targets |
  python3 -c 'import json,sys; t=json.load(sys.stdin)["data"]["activeTargets"]; print(sum(x["labels"]["job"]=="triage-api" and x["health"]=="up" for x in t))')" 3
check "ticket metric reached Prometheus" \
  "$(curl -s 'http://localhost:9090/api/v1/query?query=triage_tickets_total')" '"triage_tickets_total"'
check "traces reached Jaeger" "$(curl -s http://localhost:16686/api/services)" "arabic-complaint-triage"

if ((failures > 0)); then
  echo "$failures check(s) failed"
  exit 1
fi
echo "All checks passed"
