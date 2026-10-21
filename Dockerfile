# Single-container image (web app + API in one service) for hosts that build one Dockerfile from
# the repo root, e.g. Render, Railway, Hugging Face Spaces, Fly.io, Koyeb.
# Local development with two containers still uses docker-compose.yml (backend/ + frontend/).
#
# Memory: the API needs roughly 2-3 GB RAM (all three models + ~250k matches). Hosts with
# 512 MB free tiers will run out of memory.

# ---- 1. build the web app -------------------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. API + built web app -------------------------------------------------------------------
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/src ./src
RUN uv sync --frozen --no-dev

COPY backend/models ./models
COPY data/raw /app/data/raw
COPY --from=web /web/dist /app/web

ENV MP_HOST=0.0.0.0 \
    MP_PORT=8000 \
    MP_STATIC_DIR=/app/web \
    MP_DATABASE_URL=sqlite:////app/backend/data/app.db \
    MP_ARTIFACT_PATH=/app/backend/artifacts/model.joblib \
    MP_EUROPE_ARTIFACT_PATH=/app/backend/artifacts/europe.joblib \
    MP_V2_ARTIFACT_PATH=/app/backend/artifacts/model_v2.joblib \
    MP_OPENFOOTBALL_DIR=/app/backend/data/openfootball \
    MP_HISTORY_CSV=/app/data/raw/Matches.csv \
    MP_BOOTSTRAP_ENABLED=true \
    MP_SHIPPED_MODELS_DIR=/app/backend/models \
    MP_HISTORY_CACHE_DIR=/app/backend/data/history \
    MP_SETUP_STATUS_PATH=/app/backend/data/setup_status.json

RUN useradd --create-home app && mkdir -p data artifacts && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fs "http://localhost:${PORT:-8000}/api/v1/health" || exit 1
# Hosts such as Render pass the port to listen on in $PORT.
CMD ["sh", "-c", "export MP_PORT=\"${PORT:-8000}\"; exec uv run --no-sync matchpredict"]
