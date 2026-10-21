"""FastAPI application factory.

Run: ``uv run matchpredict`` or ``uv run uvicorn matchpredict.api.app:app``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from matchpredict.api.errors import install_error_handlers
from matchpredict.api.ratelimit import RateLimiter
from matchpredict.api.results import router as results_router
from matchpredict.api.routes import router
from matchpredict.bootstrap import SetupTracker
from matchpredict.config import Settings, get_settings
from matchpredict.contracts import PredictionService
from matchpredict.db.session import SessionFactory, create_session_factory
from matchpredict.loader import (
    ServiceHolder,
    domestic_holder,
    europe_holder,
    load_europe_service,
)

log = logging.getLogger("matchpredict.api")
API_PREFIX = "/api/v1"
UK_TZ = ZoneInfo("Europe/London")


class JsonFormatter(logging.Formatter):
    """One JSON object per log line (structured logging)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def uk_today() -> dt.date:
    return dt.datetime.now(UK_TZ).date()


def create_app(
    service: PredictionService | None = None,
    session_factory: SessionFactory | None = None,
    settings: Settings | None = None,
    today: Callable[[], dt.date] = uk_today,
    europe_service: PredictionService | None = None,
    now: Callable[[], dt.datetime] | None = None,
) -> FastAPI:
    """Build the app. With ``service`` given, no artifact is loaded and no scheduler runs."""
    settings = settings or get_settings()
    injected = service is not None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.session_factory is None:
            app.state.session_factory = create_session_factory(settings.database_url)
        if not injected:
            if settings.bootstrap_enabled:  # fresh Docker volume: install the shipped models
                from matchpredict.shipped_models import install_shipped_models

                install_shipped_models(settings.shipped_models_dir, settings)
            app.state.holder.ensure_loaded()
            app.state.europe.ensure_loaded()
        scheduler = None
        if settings.scheduler_enabled and not injected:
            from matchpredict import bootstrap
            from matchpredict.ingest.scheduler import start_scheduler

            tracker = app.state.setup
            # seeding (when bootstrapping) + the catch-up ingest run in the background
            bootstrap.start_setup_thread(
                app.state.session_factory, settings, tracker, app.state.holder, app.state.europe
            )
            scheduler = start_scheduler(
                app.state.session_factory, app.state.holder, settings, europe=app.state.europe,
                catchup=False, ready=lambda: tracker.ready,
            )  # fmt: skip
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)

    app = FastAPI(
        title="MatchPredict API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.session_factory = session_factory
    if injected:
        app.state.holder = ServiceHolder(service, artifact_path=settings.artifact_path)
        app.state.europe = ServiceHolder(europe_service, artifact_path=None, loader=load_europe_service)
    else:  # v2 per scope when its artifact passed the promotion gate, else the v1 models
        app.state.holder = domestic_holder(settings)
        app.state.europe = europe_holder(settings)
    app.state.today = today
    app.state.now = now or (lambda: dt.datetime.now(dt.UTC))
    app.state.setup = SetupTracker(None if injected else settings.setup_status_path)

    install_error_handlers(app)
    app.middleware("http")(RateLimiter(settings.rate_limit, frozenset({f"{API_PREFIX}/health"})))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next: Callable) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        log.info(
            "request",
            extra={
                "fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            },
        )
        return response

    app.include_router(router, prefix=API_PREFIX)
    app.include_router(results_router, prefix=API_PREFIX)
    return app


def _default_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    return create_app(settings=settings)


app = _default_app()
