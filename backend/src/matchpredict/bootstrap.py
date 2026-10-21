"""First-start setup: history seed, UEFA history, first ingest — in the background.

The API listens straight away; :class:`SetupTracker` exposes the progress on
``/api/v1/health`` (``setup``) so the web app can show a "setting up" banner, and mirrors
it to a small JSON file (``settings.setup_status_path``) for ``docker compose exec`` users.

Stages: ``starting`` -> ``seeding_history`` -> ``seeding_uefa`` -> ``ingesting`` -> ``ready``.
A start-up on a database that has already been ingested once is ``ready`` immediately;
the seeding and catch-up steps then run quietly in the background.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from matchpredict.config import Settings
from matchpredict.db import repository
from matchpredict.db.session import SessionFactory
from matchpredict.loader import ServiceHolder

log = logging.getLogger(__name__)

STAGES = ("starting", "seeding_history", "seeding_uefa", "ingesting", "ready")
READY = "ready"


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SetupState:
    stage: str = READY
    detail: str = ""
    progress: float | None = None  # 0..1 within the stage, when known
    started_at: str | None = None
    updated_at: str | None = None
    first_start: bool = False


class SetupTracker:
    """Thread-safe current setup state (immutable snapshots), optionally mirrored to a file."""

    def __init__(self, path: Path | None = None, state: SetupState | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._state = state or SetupState()

    @property
    def state(self) -> SetupState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state.stage == READY

    def set(self, stage: str, detail: str = "", progress: float | None = None) -> SetupState:
        if stage not in STAGES:
            raise ValueError(f"unknown setup stage {stage!r}")
        with self._lock:
            started = self._state.started_at or _now()
            self._state = replace(self._state, stage=stage, detail=detail, progress=progress,
                                  started_at=started, updated_at=_now())  # fmt: skip
            state = self._state
        self._persist(state)
        return state

    def begin(self, first_start: bool) -> SetupState:
        with self._lock:
            self._state = SetupState(stage="starting" if first_start else READY,
                                     detail="Preparing first start" if first_start else "",
                                     started_at=_now(), updated_at=_now(), first_start=first_start)  # fmt: skip
            state = self._state
        self._persist(state)
        return state

    def report(self, stage: str, detail: str = "", progress: float | None = None) -> None:
        """Show progress only during a first start (otherwise the app stays ``ready``)."""
        if self._state.first_start and not self.ready:
            self.set(stage, detail, progress)

    def as_dict(self) -> dict:
        return asdict(self._state)

    def _persist(self, state: SetupState) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(asdict(state)), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            log.warning("could not write setup status to %s", self._path, exc_info=True)


@dataclass(frozen=True)
class SetupSteps:
    """Injectable step functions (tests replace them)."""

    seed_csv: Callable[..., object]
    seed_web: Callable[..., object]
    seed_uefa: Callable[..., object]
    ingest: Callable[..., object]
    needs_catchup: Callable[[SessionFactory], bool]
    import_backtest: Callable[..., object] = lambda *a, **k: None  # shipped Results-page backtest


def default_steps() -> SetupSteps:
    from matchpredict.db.backtest_bundle import import_bundle
    from matchpredict.db.seed import seed as seed_csv
    from matchpredict.ingest.europe import seed_history as seed_uefa
    from matchpredict.ingest.history import seed_from_web
    from matchpredict.ingest.run import run_ingest
    from matchpredict.ingest.scheduler import needs_catchup

    return SetupSteps(seed_csv=seed_csv, seed_web=seed_from_web, seed_uefa=seed_uefa,
                      ingest=run_ingest, needs_catchup=needs_catchup, import_backtest=import_bundle)  # fmt: skip


def is_first_start(session_factory: SessionFactory) -> bool:
    """No ingest has ever succeeded on this database."""
    with session_factory() as session:
        return repository.last_successful_run(session) is None


def _step(name: str, errors: list[str], fn: Callable[[], object]) -> object | None:
    try:
        return fn()
    except Exception as exc:  # one failed step never blocks the next ones
        log.exception("setup step %s failed", name)
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return None


def _seed_history(session_factory: SessionFactory, settings: Settings, tracker: SetupTracker,
                  steps: SetupSteps, fresh: dict[str, bytes], errors: list[str]) -> None:  # fmt: skip
    from matchpredict.ingest import history

    if history.is_complete(settings.history_cache_dir):
        return
    csv_path = Path(settings.history_csv)
    if csv_path.exists():
        tracker.report("seeding_history", f"Loading match history from {csv_path.name}")
        if _step("history csv", errors, lambda: steps.seed_csv(session_factory, csv_path)) is not None:
            history.mark_complete(settings.history_cache_dir, {"source": str(csv_path)})
        return

    def progress(done: int, total: int) -> None:
        tracker.report("seeding_history", f"Downloading match history from football-data.co.uk ({done}/{total} files)",
                       progress=round(done / total, 3) if total else None)  # fmt: skip

    tracker.report("seeding_history", "Downloading match history from football-data.co.uk", progress=0.0)
    result = _step("history download", errors,
                   lambda: steps.seed_web(session_factory, settings.history_cache_dir, progress=progress,
                                          fresh=fresh, workers=settings.history_workers))  # fmt: skip
    errors.extend(f"history: {e}" for e in getattr(result, "errors", [])[:5])


def run_setup(
    session_factory: SessionFactory,
    settings: Settings,
    tracker: SetupTracker,
    holder: ServiceHolder | None = None,
    europe: ServiceHolder | None = None,
    steps: SetupSteps | None = None,
) -> list[str]:
    """Seed (when bootstrapping) and run the catch-up ingest; always ends ``ready``. Returns errors."""
    steps = steps or default_steps()
    errors: list[str] = []
    fresh: dict[str, bytes] = {}
    try:
        if settings.bootstrap_enabled:
            _seed_history(session_factory, settings, tracker, steps, fresh, errors)
            tracker.report("seeding_uefa", "Loading UEFA competition history")
            _step("uefa history", errors, lambda: steps.seed_uefa(session_factory, settings, europe))
            # after both seeds: rows are joined to matches by (league, date, home, away)
            _step("backtest import", errors,
                  lambda: steps.import_backtest(session_factory, settings.backtest_bundle_path))  # fmt: skip
        for h in (holder, europe):  # models may have appeared meanwhile (lazy load)
            if h is not None:
                h.ensure_loaded()
        if steps.needs_catchup(session_factory):
            tracker.report("ingesting", "Downloading today's fixtures and the latest results")
            _step("ingest", errors, lambda: _ingest(session_factory, settings, steps, holder, europe, fresh))
    finally:
        detail = "Setup finished" if not errors else "Setup finished with errors: " + "; ".join(errors)[:500]
        if not tracker.ready:
            tracker.set(READY, detail)
        log.info("setup done (%d errors)", len(errors))
    return errors


def _ingest(session_factory: SessionFactory, settings: Settings, steps: SetupSteps,
            holder: ServiceHolder | None, europe: ServiceHolder | None, fresh: dict[str, bytes]) -> object:  # fmt: skip
    from matchpredict.ingest.history import memo_fetcher
    from matchpredict.ingest.http import fetch_bytes

    kwargs = {"europe": europe}
    if fresh:  # reuse the current-season files the history step just downloaded
        kwargs["fetch"] = memo_fetcher(fresh, fetch_bytes)
    return steps.ingest(session_factory, holder, settings, **kwargs)


def start_setup_thread(
    session_factory: SessionFactory,
    settings: Settings,
    tracker: SetupTracker,
    holder: ServiceHolder | None = None,
    europe: ServiceHolder | None = None,
    steps: SetupSteps | None = None,
) -> threading.Thread:
    tracker.begin(first_start=is_first_start(session_factory))
    thread = threading.Thread(
        target=run_setup, args=(session_factory, settings, tracker, holder, europe, steps),
        name="matchpredict-setup", daemon=True,
    )  # fmt: skip
    thread.start()
    return thread
