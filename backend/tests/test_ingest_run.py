import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
from mp_testkit import FIXTURES, FakeService, frame, memory_factory

from matchpredict import loader
from matchpredict.config import Settings
from matchpredict.db import repository as repo
from matchpredict.db.models import IngestRun, Match
from matchpredict.ingest import run as run_mod
from matchpredict.ingest import scheduler as sched_mod
from matchpredict.ingest.http import FetchError
from matchpredict.loader import ServiceHolder

TODAY = dt.date(2026, 10, 20)
NO_SLEEP = lambda _s: None


def _fetch(url: str) -> bytes:
    files = {
        "/mmz4281/2627/E0.csv": "E0.csv",
        "/new/ARG.csv": "ARG.csv",
        "/fixtures.csv": "fixtures.csv",
        "/new_league_fixtures.csv": "new_league_fixtures.csv",
    }
    for suffix, name in files.items():
        if url.endswith(suffix):
            return (FIXTURES / name).read_bytes()
    raise FetchError(f"404 for {url}")


def _settings(tmp_path: Path, **kw) -> Settings:
    base = {
        "database_url": "sqlite:///:memory:",
        "artifact_path": tmp_path / "model.joblib",
        "v2_artifact_path": tmp_path / "no-v2.joblib",
        "scheduler_enabled": False,
        "football_data_org_key": None,
    }
    return Settings(**{**base, **kw})


def test_full_ingest_without_model_stores_results_and_fixtures(tmp_path):
    factory = memory_factory()
    summary = run_mod.run_ingest(
        factory, None, _settings(tmp_path), fetch=_fetch, today=TODAY, sleep=NO_SLEEP
    )
    assert summary.status == "partial"  # most result files 404 in this fake
    assert summary.results_rows == 6  # 3 E0 + 3 recent ARG
    assert summary.fixtures_rows == 10  # 5 main + 5 extra-league samples
    assert summary.applied_results == 0 and not summary.model_saved
    with factory() as s:
        assert repo.count_matches(s, "finished") == 6
        assert repo.count_matches(s, "scheduled") == summary.fixtures_rows
        run = s.query(IngestRun).one()
        assert run.status == "partial" and run.finished_at and "404" in run.error


def test_ingest_updates_model_saves_and_finishes_fixture(tmp_path):
    factory = memory_factory()
    with factory() as s:  # a fixture that is now finished in E0.csv
        repo.upsert_matches(
            s, frame([{"date": "2026-10-19", "league": "E0", "home": "Tottenham", "away": "Coventry"}])
        )
    service = FakeService()
    holder = ServiceHolder(service)
    settings = _settings(tmp_path)
    summary = run_mod.run_ingest(factory, holder, settings, fetch=_fetch, today=TODAY, sleep=NO_SLEEP)
    assert summary.applied_results == 4  # E0 19/10 + 3 ARG after 2026-10-01
    assert summary.model_saved and service.saved == [settings.artifact_path]
    assert service.last_result_date == pd.Timestamp("2026-10-20")
    with factory() as s:
        m = s.query(Match).filter_by(home="Tottenham", away="Coventry").one()
        assert m.status == "finished" and (m.home_goals, m.away_goals) == (3, 1)


def test_model_failure_does_not_block_fixtures(tmp_path):
    service = FakeService()

    def boom(_df):
        raise RuntimeError("ratings exploded")

    service.update_results = boom
    factory = memory_factory()
    summary = run_mod.run_ingest(
        factory, ServiceHolder(service), _settings(tmp_path), fetch=_fetch, today=TODAY, sleep=NO_SLEEP
    )
    assert any("ratings exploded" in e for e in summary.errors)
    assert summary.fixtures_rows > 0


def test_unexpected_failure_is_recorded(tmp_path, monkeypatch):
    def broken(*_a, **_k):
        raise ValueError("bad")

    monkeypatch.setattr(run_mod.football_data_uk, "fetch_results", broken)
    factory = memory_factory()
    summary = run_mod.run_ingest(factory, None, _settings(tmp_path), fetch=_fetch, today=TODAY)
    assert summary.status == "failed"
    with factory() as s:
        assert s.query(IngestRun).one().status == "failed"


def test_concurrent_run_is_skipped(tmp_path):
    assert run_mod._RUN_LOCK.acquire(blocking=False)
    try:
        summary = run_mod.run_ingest(memory_factory(), None, _settings(tmp_path), fetch=_fetch)
    finally:
        run_mod._RUN_LOCK.release()
    assert summary.status == "skipped"


def test_org_fixtures_used_when_key_set(tmp_path, monkeypatch):
    captured = {}

    def fake_org(key, start, end, known, sleep):
        captured.update(key=key, start=start, end=end, known=known)
        return run_mod.football_data_uk.FetchResult(
            frame=frame([{"date": "2026-10-31", "league": "E0", "home": "Arsenal", "away": "Coventry",
                          "time": "15:00"}])
        )

    monkeypatch.setattr(run_mod.football_data_org, "fetch_org_fixtures", fake_org)
    factory = memory_factory()
    settings = _settings(tmp_path, football_data_org_key="k")
    summary = run_mod.run_ingest(factory, None, settings, fetch=_fetch, today=TODAY, sleep=NO_SLEEP)
    assert summary.org_rows == 1
    assert captured["end"] == TODAY + dt.timedelta(days=14)
    assert "Arsenal" in captured["known"]["E0"]


def test_cli_main_prints_summary(tmp_path, monkeypatch, capsys):
    db = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
    monkeypatch.setattr(run_mod, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(run_mod, "run_ingest", lambda f, h, s, **kw: run_mod.IngestSummary(status="success"))
    run_mod.main(["--database-url", db, "--no-model"])
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    monkeypatch.setattr(run_mod, "run_ingest", lambda f, h, s, **kw: run_mod.IngestSummary(status="failed"))
    with pytest.raises(SystemExit):
        run_mod.main(["--database-url", db])


def test_needs_catchup_and_scheduler(tmp_path, monkeypatch):
    factory = memory_factory()
    assert sched_mod.needs_catchup(factory)
    with factory() as s:
        repo.finish_run(s, repo.start_run(s), "success")
    assert not sched_mod.needs_catchup(factory)
    later = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=13)
    assert sched_mod.needs_catchup(factory, now=later)

    ran = []
    monkeypatch.setattr(sched_mod, "run_ingest", lambda f, h, s, **kw: ran.append(1))
    monkeypatch.setattr(sched_mod, "needs_catchup", lambda f: True)
    scheduler = sched_mod.start_scheduler(factory, ServiceHolder(), _settings(tmp_path))
    try:
        job = scheduler.get_job(sched_mod.JOB_ID)
        assert "hour='7'" in str(job.trigger)
        assert scheduler.get_job("catchup_ingest") is not None or ran
    finally:
        scheduler.shutdown(wait=True)
    assert ran == [1]


def test_load_service_missing_and_broken(tmp_path, monkeypatch):
    assert loader.load_service(tmp_path / "missing.joblib") is None
    art = tmp_path / "model.joblib"
    art.write_bytes(b"not a model")
    # either the predictor module is missing or loading garbage fails: both degrade to None
    assert loader.load_service(art) is None


def test_holder_caches_predictions_and_lazy_loads(tmp_path, monkeypatch):
    service = FakeService()
    holder = ServiceHolder(service)
    day = pd.Timestamp("2026-10-24")
    holder.predict("Arsenal", "Chelsea", "E0", day)
    holder.predict("Arsenal", "Chelsea", "E0", day)
    assert len(service.calls) == 1
    holder.invalidate()
    holder.predict("Arsenal", "Chelsea", "E0", day)
    assert len(service.calls) == 2

    empty = ServiceHolder(artifact_path=tmp_path / "m.joblib")
    with pytest.raises(RuntimeError):
        empty.predict("A", "B", "E0", day)
    monkeypatch.setattr(loader, "load_service", lambda p: service)
    assert empty.ensure_loaded() is service and empty.loaded


def test_holder_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(loader, "PREDICTION_CACHE_SIZE", 2)
    holder = ServiceHolder(FakeService())
    for i in range(3):
        holder.predict("A", f"B{i}", "E0", pd.Timestamp("2026-10-24"))
    assert len(holder._cache) == 2
