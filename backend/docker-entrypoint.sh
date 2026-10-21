#!/bin/sh
# Container start-up: seed the database (idempotent upserts), then run the API.
# A failed seed step is logged but never stops the server from starting.
set -u

run_step() {
  echo "[entrypoint] $1"
  shift
  if ! "$@"; then
    echo "[entrypoint] WARNING: step failed, continuing (the API will still start)"
  fi
}

if [ -f "${MP_HISTORY_CSV:-/app/data/raw/Matches.csv}" ]; then
  run_step "loading match history (first start ~1 min, then seconds)" \
    uv run --no-sync python -m matchpredict.db.seed
else
  echo "[entrypoint] WARNING: ${MP_HISTORY_CSV:-/app/data/raw/Matches.csv} not found - mount data/raw to load history"
fi

run_step "loading UEFA competition history" \
  uv run --no-sync python -m matchpredict.ingest.europe seed

echo "[entrypoint] starting API"
exec uv run --no-sync matchpredict
