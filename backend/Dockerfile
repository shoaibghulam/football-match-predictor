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
    MP_EUROPE_ARTIFACT_PATH=/app/backend/artifacts/europe.joblib \
    MP_V2_ARTIFACT_PATH=/app/backend/artifacts/model_v2.joblib \
    MP_OPENFOOTBALL_DIR=/app/backend/data/openfootball \
    MP_HISTORY_CSV=/app/data/raw/Matches.csv

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# strip CRLF in case the file was checked out on Windows
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

RUN useradd --create-home app && mkdir -p data artifacts && chown -R app /app
USER app

EXPOSE 8000
# first start seeds ~250k matches before the API listens, so allow a long start period
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fs http://localhost:8000/api/v1/health || exit 1
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
