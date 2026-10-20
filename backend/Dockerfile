FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

ENV MP_HOST=0.0.0.0 \
    MP_PORT=8000 \
    MP_DATABASE_URL=sqlite:////app/backend/data/app.db \
    MP_ARTIFACT_PATH=/app/backend/artifacts/model.joblib \
    MP_HISTORY_CSV=/app/data/raw/Matches.csv

RUN useradd --create-home app && mkdir -p data artifacts && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/api/v1/health || exit 1
CMD ["uv", "run", "--no-sync", "matchpredict"]
