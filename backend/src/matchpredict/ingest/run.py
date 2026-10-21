"""Daily ingest: results -> DB -> ratings update -> fixtures -> DB.

Usage: ``uv run python -m matchpredict.ingest.run``
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta

from matchpredict.config import Settings, get_settings
from matchpredict.db import repository
from matchpredict.db.session import SessionFactory, create_session_factory
from matchpredict.ingest import football_data_org, football_data_uk
from matchpredict.ingest.http import Fetcher, fetch_bytes
from matchpredict.loader import ServiceHolder, artifact_path_of

log = logging.getLogger(__name__)

STALE_FIXTURE_DAYS = 3
ORG_HORIZON_DAYS = 14
RESUME_OVERLAP_DAYS = 7
_RUN_LOCK = threading.Lock()  # one ingest at a time per process


@dataclass
class IngestSummary:
    status: str = "skipped"
    results_rows: int = 0
    fixtures_rows: int = 0
    org_rows: int = 0
    inserted: int = 0
    updated: int = 0
    pruned: int = 0
    applied_results: int = 0
    model_saved: bool = False
    europe_rows: int = 0
    snapshots: int = 0
    errors: list[str] = field(default_factory=list)


def _apply_to_model(
    session_factory: SessionFactory, holder: ServiceHolder | None, today: date, settings: Settings
) -> tuple[int, bool]:
    """Feed finished matches newer than the model's last result; save the artifact if changed."""
    service = holder.ensure_loaded() if holder else None
    if service is None:
        log.info("no model loaded; skipping ratings update")
        return 0, False
    with holder.lock:
        last = service.last_result_date
        # include the last applied day(s); the service skips what it already has
        start = last.date() if last is not None else today - timedelta(days=365)
        after = start - timedelta(days=int(getattr(service, "update_lookback_days", 1)))
        with session_factory() as session:
            frame = repository.finished_frame(session, after=after)
        applied = int(service.update_results(frame)) if not frame.empty else 0
        saved = False
        if applied:
            holder.invalidate()
            save = getattr(service, "save", None)
            if callable(save):
                save(artifact_path_of(service, settings.artifact_path))
                saved = True
    log.info("model updated with %d results (saved=%s)", applied, saved)
    return applied, saved


def freeze_snapshots(
    session_factory: SessionFactory,
    holder: ServiceHolder | None,
    europe: ServiceHolder | None,
    settings: Settings,
    now: datetime | None = None,
) -> int:
    """Store / refresh pre-kickoff prediction snapshots for the coming days (never raises)."""
    try:
        from matchpredict.api.predictions import Models
        from matchpredict.api.snapshots import freeze_upcoming

        models = Models(domestic=holder or ServiceHolder(), europe=europe or ServiceHolder())
        return freeze_upcoming(
            session_factory, models, now or datetime.now(UTC), settings.snapshot_days_ahead,
            settings.snapshot_refresh_minutes,
        )  # fmt: skip
    except Exception:
        log.exception("snapshot freezing failed")
        return 0


def _org_fixtures(
    session_factory: SessionFactory, settings: Settings, today: date, sleep: Callable[[float], None]
) -> football_data_uk.FetchResult | None:
    if not settings.football_data_org_key:
        return None
    with session_factory() as session:
        known = repository.known_teams_by_league(session)
    return football_data_org.fetch_org_fixtures(
        settings.football_data_org_key,
        today,
        today + timedelta(days=ORG_HORIZON_DAYS),
        known,
        sleep=sleep,
    )


def run_ingest(
    session_factory: SessionFactory,
    holder: ServiceHolder | None = None,
    settings: Settings | None = None,
    *,
    fetch: Fetcher = fetch_bytes,
    today: date | None = None,
    sleep: Callable[[float], None] = time.sleep,
    europe: ServiceHolder | None = None,
) -> IngestSummary:
    """Run one full ingest; never raises (failures are recorded on the IngestRun).

    With ``europe`` (the European model holder) UEFA fixtures / results are refreshed too.
    """
    settings = settings or get_settings()
    today = today or datetime.now(UTC).date()
    if not _RUN_LOCK.acquire(blocking=False):
        log.info("ingest already running; skipping")
        return IngestSummary(status="skipped", errors=["already running"])
    summary = IngestSummary(status="running")
    with session_factory() as session:
        run = repository.start_run(session)
    try:
        _run_steps(session_factory, holder, settings, fetch, today, sleep, summary)
        if europe is not None and settings.europe_ingest_enabled:
            _europe_step(session_factory, europe, settings, fetch, today, sleep, summary)
        summary.snapshots = freeze_snapshots(session_factory, holder, europe, settings)
        summary.status = "partial" if summary.errors else "success"
    except Exception as exc:
        log.exception("ingest failed")
        summary.status = "failed"
        summary.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        _RUN_LOCK.release()
    with session_factory() as session:
        repository.finish_run(
            session,
            session.merge(run),
            summary.status,
            results_rows=summary.results_rows,
            fixtures_rows=summary.fixtures_rows + summary.org_rows,
            inserted=summary.inserted,
            updated=summary.updated,
            applied_results=summary.applied_results,
            error="\n".join(summary.errors)[:4000] or None,
        )
    log.info("ingest %s: %s", summary.status, json.dumps(asdict(summary), default=str))
    return summary


def _run_steps(
    session_factory: SessionFactory,
    holder: ServiceHolder | None,
    settings: Settings,
    fetch: Fetcher,
    today: date,
    sleep: Callable[[float], None],
    summary: IngestSummary,
) -> None:
    # Fixtures first (2 small files): a fresh install shows matches within seconds.
    # Upserts are order-independent (status only moves forward).
    fixtures = football_data_uk.fetch_fixtures(fetch=fetch, sleep=sleep)
    summary.errors.extend(fixtures.errors)
    summary.fixtures_rows = len(fixtures.frame)
    with session_factory() as session:
        stats = repository.upsert_matches(session, fixtures.frame, source="football-data.co.uk")

    with session_factory() as session:  # resume extra leagues from their latest stored result
        since = {
            league: day - timedelta(days=RESUME_OVERLAP_DAYS)
            for league, day in repository.latest_finished_by_league(session).items()
        }
    results = football_data_uk.fetch_results(
        today, lookback_days=settings.results_lookback_days, since=since, fetch=fetch, sleep=sleep
    )
    summary.errors.extend(results.errors)
    summary.results_rows = len(results.frame)
    with session_factory() as session:
        stats = stats + repository.upsert_matches(session, results.frame, source="football-data.co.uk")

    try:
        summary.applied_results, summary.model_saved = _apply_to_model(
            session_factory, holder, today, settings
        )
    except Exception as exc:
        log.exception("model update failed")
        summary.errors.append(f"model update: {type(exc).__name__}: {exc}")

    org = _org_fixtures(session_factory, settings, today, sleep)
    if org is not None:
        summary.errors.extend(org.errors)
        summary.org_rows = len(org.frame)
        with session_factory() as session:
            stats = stats + repository.upsert_matches(session, org.frame, source="football-data.org")

    with session_factory() as session:
        summary.pruned = repository.prune_stale_fixtures(
            session, before=today - timedelta(days=STALE_FIXTURE_DAYS)
        )
    if holder is not None:
        holder.invalidate()
    summary.inserted, summary.updated = stats.inserted, stats.updated


def _europe_step(
    session_factory: SessionFactory,
    europe: ServiceHolder,
    settings: Settings,
    fetch: Fetcher,
    today: date,
    sleep: Callable[[float], None],
    summary: IngestSummary,
) -> None:
    from matchpredict.ingest.europe import refresh_european

    result = refresh_european(session_factory, settings, europe, today=today, fetch=fetch, sleep=sleep)
    summary.europe_rows = result.rows
    summary.inserted += result.inserted
    summary.updated += result.updated
    summary.errors.extend(f"uefa: {e}" for e in result.errors)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run one MatchPredict data ingest")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--no-model", action="store_true", help="skip the ratings update")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    factory = create_session_factory(args.database_url)
    from matchpredict.loader import domestic_holder, europe_holder

    holder = None if args.no_model else domestic_holder(settings)
    europe = europe_holder(settings)
    summary = run_ingest(factory, holder, settings, europe=europe)
    print(json.dumps(asdict(summary), indent=2, default=str))  # CLI output
    if summary.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
