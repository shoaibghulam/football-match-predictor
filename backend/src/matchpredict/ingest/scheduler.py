"""APScheduler wiring: daily ingest plus a catch-up run at startup."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from matchpredict.config import Settings
from matchpredict.db import repository
from matchpredict.db.session import SessionFactory
from matchpredict.ingest.run import run_ingest
from matchpredict.loader import ServiceHolder

log = logging.getLogger(__name__)

CATCHUP_AFTER = timedelta(hours=12)
JOB_ID = "daily_ingest"
LIVE_JOB_ID = "uefa_live_refresh"
RETRAIN_JOB_ID = "weekly_v2_retrain"


def needs_catchup(session_factory: SessionFactory, now: datetime | None = None) -> bool:
    """True when the last successful ingest is missing or older than 12 hours."""
    now = now or datetime.now(UTC).replace(tzinfo=None)
    with session_factory() as session:
        last = repository.last_successful_run(session)
    return last is None or last.finished_at is None or now - last.finished_at > CATCHUP_AFTER


def start_scheduler(
    session_factory: SessionFactory,
    holder: ServiceHolder,
    settings: Settings,
    europe: ServiceHolder | None = None,
) -> BackgroundScheduler:
    def job() -> None:
        run_ingest(session_factory, holder, settings, europe=europe)

    def live_job() -> None:
        from matchpredict.ingest.europe import refresh_live

        summary = refresh_live(session_factory, settings, europe)
        if not summary.skipped:
            log.info("uefa live refresh: rows=%d live=%d errors=%d", summary.rows, summary.live, len(summary.errors))

    def retrain_job() -> None:
        from matchpredict.predictor.v2.retrain import weekly_retrain

        holders = [h for h in (holder, europe) if h is not None]
        weekly_retrain(settings, holders)

    scheduler = BackgroundScheduler(timezone=UTC)
    scheduler.add_job(
        job,
        CronTrigger(hour=settings.ingest_hour_utc, minute=0, timezone=UTC),
        id=JOB_ID,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    if needs_catchup(session_factory):
        log.info("last ingest is stale; scheduling a catch-up run now")
        scheduler.add_job(job, id="catchup_ingest", max_instances=1)  # runs immediately
    if europe is not None and settings.europe_ingest_enabled:
        # Polls only while a UEFA match is live / within ±3 h of kick-off (see refresh_live).
        scheduler.add_job(
            live_job,
            IntervalTrigger(minutes=settings.live_refresh_minutes, timezone=UTC),
            id=LIVE_JOB_ID,
            max_instances=1,
            coalesce=True,
        )
    if settings.retrain_enabled:
        scheduler.add_job(
            retrain_job,
            CronTrigger(day_of_week=settings.retrain_day_of_week, hour=settings.retrain_hour_utc, minute=0, timezone=UTC),
            id=RETRAIN_JOB_ID,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=6 * 3600,
        )
    scheduler.start()
    log.info("scheduler started (daily at %02d:00 UTC)", settings.ingest_hour_utc)
    return scheduler
