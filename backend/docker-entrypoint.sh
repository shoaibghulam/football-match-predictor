#!/bin/sh
# Container start-up. The API listens straight away; with MP_BOOTSTRAP_ENABLED=true (the image
# default) it then, in the background:
#   1. installs the pre-trained models shipped in /app/backend/models into the artifacts
#      volume (only files that are missing - your own / retrained models are never replaced);
#   2. seeds the match history - data/raw/Matches.csv when mounted, otherwise a one-time
#      download of every season from football-data.co.uk (~5 min, cached in the DB volume);
#   3. loads the UEFA history (openfootball) and runs the first ingest (today's fixtures).
# Progress: GET /api/v1/health -> "setup" (the web app shows a banner until it is "ready").
set -u

echo "[entrypoint] starting API (first-start setup runs in the background; see /api/v1/health)"
exec uv run --no-sync matchpredict
