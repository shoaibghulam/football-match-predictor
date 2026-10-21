import json
from types import SimpleNamespace

import pytest
from mp_testkit import FakeService, memory_factory

from matchpredict import bootstrap
from matchpredict.config import Settings
from matchpredict.db import repository as repo
from matchpredict.ingest import history
from matchpredict.loader import ServiceHolder
from matchpredict.shipped_models import install_shipped_models


def _settings(tmp_path, **kw) -> Settings:
    base = {
        "database_url": "sqlite:///:memory:",
        "artifact_path": tmp_path / "art" / "model.joblib",
        "europe_artifact_path": tmp_path / "art" / "europe.joblib",
        "v2_artifact_path": tmp_path / "art" / "model_v2.joblib",
        "history_csv": tmp_path / "raw" / "Matches.csv",
        "history_cache_dir": tmp_path / "history",
        "setup_status_path": tmp_path / "setup_status.json",
        "bootstrap_enabled": True,
        "scheduler_enabled": False,
        "football_data_org_key": None,
    }
    return Settings(**{**base, **kw})


class Steps:
    """Records the order of the setup steps; each can be told to fail."""

    def __init__(self, fail: set[str] | None = None, catchup: bool = True, tracker=None) -> None:
        self.calls: list[str] = []
        self.kwargs: dict[str, dict] = {}
        self.fail = fail or set()
        self.catchup = catchup
        self.stages: list[str] = []
        self.tracker = tracker

    def _run(self, name, kwargs):
        self.calls.append(name)
        self.kwargs[name] = kwargs
        if self.tracker is not None:
            self.stages.append(self.tracker.state.stage)
        if name in self.fail:
            raise RuntimeError(f"{name} broke")

    def build(self) -> bootstrap.SetupSteps:
        def seed_web(factory, cache_dir, **kw):
            self._run("web", kw)
            kw["progress"](1, 2)
            kw["fresh"]["https://football-data.co.uk/mmz4281/2627/E0.csv"] = b"E0"
            return history.HistoryResult(status="partial", errors=["one file failed"])

        return bootstrap.SetupSteps(
            seed_csv=lambda factory, path, **kw: self._run("csv", kw) or "ok",
            seed_web=seed_web,
            seed_uefa=lambda factory, settings, europe, **kw: self._run("uefa", kw),
            ingest=lambda factory, holder, settings, **kw: self._run("ingest", kw),
            needs_catchup=lambda factory: self.catchup,
            import_backtest=lambda factory, path, **kw: self._run("backtest", kw),
        )


# --- tracker ------------------------------------------------------------------------


def test_tracker_persists_snapshots(tmp_path):
    path = tmp_path / "s" / "status.json"
    tracker = bootstrap.SetupTracker(path)
    assert tracker.ready and tracker.as_dict()["stage"] == "ready"
    tracker.begin(first_start=True)
    assert tracker.state.stage == "starting" and not tracker.ready
    before = tracker.state
    tracker.set("seeding_history", "downloading", progress=0.5)
    assert before.stage == "starting"  # snapshots are immutable
    saved = json.loads(path.read_text())
    assert saved["stage"] == "seeding_history" and saved["progress"] == 0.5 and saved["first_start"]
    with pytest.raises(ValueError):
        tracker.set("bogus")


def test_tracker_report_only_during_first_start(tmp_path):
    tracker = bootstrap.SetupTracker()
    tracker.begin(first_start=False)
    tracker.report("ingesting", "catch-up")
    assert tracker.ready  # an existing install never shows the setup banner
    tracker.begin(first_start=True)
    tracker.report("ingesting", "first ingest")
    assert tracker.state.stage == "ingesting"


def test_tracker_survives_unwritable_status_file(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    tracker = bootstrap.SetupTracker(blocker / "status.json")  # parent is a file
    tracker.set("ingesting")
    assert tracker.state.stage == "ingesting"


# --- run_setup ----------------------------------------------------------------------


def test_first_start_downloads_history_then_uefa_then_ingest(tmp_path):
    factory = memory_factory()
    tracker = bootstrap.SetupTracker(tmp_path / "status.json")
    tracker.begin(first_start=bootstrap.is_first_start(factory))
    steps = Steps(tracker=tracker)
    holder = ServiceHolder(FakeService())
    errors = bootstrap.run_setup(factory, _settings(tmp_path), tracker, holder, None, steps.build())
    assert steps.calls == ["web", "uefa", "ingest", "backtest"]
    assert steps.stages == ["seeding_history", "seeding_uefa", "ingesting", "ingesting"]
    assert tracker.ready and "errors" in tracker.state.detail and errors == ["history: one file failed"]
    # the first ingest reuses the files the history step just downloaded
    fetch = steps.kwargs["ingest"]["fetch"]
    assert fetch("https://www.football-data.co.uk/mmz4281/2627/E0.csv") == b"E0"


def test_csv_present_uses_fast_path_and_marks_complete(tmp_path):
    settings = _settings(tmp_path)
    settings.history_csv.parent.mkdir(parents=True)
    settings.history_csv.write_text("x")
    steps = Steps()
    tracker = bootstrap.SetupTracker()
    tracker.begin(first_start=True)
    assert bootstrap.run_setup(memory_factory(), settings, tracker, steps=steps.build()) == []
    assert steps.calls == ["csv", "uefa", "ingest", "backtest"] and "fetch" not in steps.kwargs["ingest"]
    assert history.is_complete(settings.history_cache_dir)
    assert tracker.state.detail == "Setup finished"

    again = Steps(catchup=False)
    bootstrap.run_setup(memory_factory(), settings, tracker, steps=again.build())
    assert again.calls == ["uefa", "backtest"]  # history already complete, ingest fresh


def test_failed_steps_do_not_block_later_ones(tmp_path):
    steps = Steps(fail={"csv", "uefa"})
    settings = _settings(tmp_path)
    settings.history_csv.parent.mkdir(parents=True)
    settings.history_csv.write_text("x")
    tracker = bootstrap.SetupTracker()
    tracker.begin(first_start=True)
    errors = bootstrap.run_setup(memory_factory(), settings, tracker, steps=steps.build())
    assert steps.calls == ["csv", "uefa", "ingest", "backtest"]
    assert len(errors) == 2 and tracker.ready
    assert not history.is_complete(settings.history_cache_dir)  # csv seed retried next start


def test_without_bootstrap_only_the_catchup_ingest_runs(tmp_path):
    steps = Steps()
    tracker = bootstrap.SetupTracker()
    bootstrap.run_setup(memory_factory(), _settings(tmp_path, bootstrap_enabled=False), tracker, steps=steps.build())
    assert steps.calls == ["ingest"] and tracker.ready


def test_setup_thread_and_first_start_detection(tmp_path):
    factory = memory_factory()
    assert bootstrap.is_first_start(factory)
    steps = Steps(catchup=False)
    tracker = bootstrap.SetupTracker()
    thread = bootstrap.start_setup_thread(factory, _settings(tmp_path), tracker, steps=steps.build())
    thread.join(timeout=10)
    assert tracker.ready and steps.calls == ["web", "uefa", "backtest"]
    with factory() as s:
        repo.finish_run(s, repo.start_run(s), "partial")
    assert not bootstrap.is_first_start(factory)


def test_default_steps_are_the_real_functions():
    steps = bootstrap.default_steps()
    assert steps.seed_web is history.seed_from_web
    from matchpredict.db.backtest_bundle import import_bundle

    assert steps.import_backtest is import_bundle
    assert callable(steps.ingest) and callable(steps.needs_catchup)


# --- shipped models -----------------------------------------------------------------


def _ship(tmp_path, names):
    src = tmp_path / "models"
    src.mkdir()
    for name in names:
        (src / name).write_text(f"shipped {name}")
    return src


def test_install_copies_only_missing_files(tmp_path):
    settings = _settings(tmp_path)
    src = _ship(tmp_path, ["model.joblib", "europe.joblib", "model_v2.joblib", "metrics.json", ".hidden",
                           "backtest_predictions.csv.gz"])
    settings.artifact_path.parent.mkdir(parents=True)
    settings.europe_artifact_path.write_text("my own europe model")
    installed = install_shipped_models(src, settings)
    assert sorted(p.name for p in installed) == ["metrics.json", "model.joblib", "model_v2.joblib"]
    assert settings.europe_artifact_path.read_text() == "my own europe model"  # never overwritten
    assert settings.artifact_path.read_text() == "shipped model.joblib"
    assert not (settings.artifact_path.parent / ".hidden").exists()
    assert install_shipped_models(src, settings) == []  # idempotent


def test_sidecar_only_installed_with_its_model(tmp_path):
    settings = _settings(tmp_path)
    src = _ship(tmp_path, ["model_v2.joblib", "model_v2.joblib.state"])
    settings.artifact_path.parent.mkdir(parents=True)
    settings.v2_artifact_path.write_text("user-trained v2")
    assert install_shipped_models(src, settings) == []  # shipped state must not pair with it
    settings.v2_artifact_path.unlink()
    names = [p.name for p in install_shipped_models(src, settings)]
    assert names == ["model_v2.joblib", "model_v2.joblib.state"]


def test_install_handles_missing_dir_and_copy_errors(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    assert install_shipped_models(tmp_path / "nope", settings) == []
    src = _ship(tmp_path, ["model.joblib"])
    import matchpredict.shipped_models as sm

    monkeypatch.setattr(sm, "_copy", lambda a, b: (_ for _ in ()).throw(OSError("read-only")))
    assert install_shipped_models(src, settings) == []


def test_health_reports_setup_stage(tmp_path):
    from fastapi.testclient import TestClient

    from matchpredict.api.app import create_app

    app = create_app(FakeService(), memory_factory(), _settings(tmp_path, rate_limit="1000/minute"))
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["setup"]["stage"] == "ready"
        app.state.setup.begin(first_start=True)
        app.state.setup.set("seeding_history", "Downloading (3/610 files)", progress=0.005)
        setup = client.get("/api/v1/health").json()["setup"]
    assert setup["stage"] == "seeding_history" and setup["progress"] == 0.005 and setup["first_start"]
    assert setup["started_at"] and setup["detail"].startswith("Downloading")


def test_lifespan_bootstrap_installs_models_and_starts_setup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from matchpredict.api.app import create_app
    from matchpredict.ingest import scheduler as sched_mod

    src = _ship(tmp_path, ["model.joblib"])
    settings = _settings(tmp_path, shipped_models_dir=src, scheduler_enabled=True, rate_limit="1000/minute",
                         database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}")  # fmt: skip
    started = []
    monkeypatch.setattr(bootstrap, "start_setup_thread", lambda *a, **k: started.append(a[2]))
    monkeypatch.setattr(sched_mod, "start_scheduler",
                        lambda *a, **kw: started.append(kw) or SimpleNamespace(shutdown=lambda wait: None))  # fmt: skip
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/api/v1/health").json()["model_loaded"] is False  # fake file: load fails
    assert settings.artifact_path.read_text() == "shipped model.joblib"
    tracker, kw = started
    assert isinstance(tracker, bootstrap.SetupTracker)
    assert kw["catchup"] is False and kw["ready"]() is True


def test_scheduler_jobs_wait_for_setup(tmp_path, monkeypatch):
    from matchpredict.ingest import scheduler as sched_mod

    ran = []
    monkeypatch.setattr(sched_mod, "run_ingest", lambda *a, **kw: ran.append("ingest"))
    monkeypatch.setattr(sched_mod, "needs_catchup", lambda f: pytest.fail("catch-up is the setup thread's job"))
    ready = {"value": False}
    europe = ServiceHolder()
    scheduler = sched_mod.start_scheduler(memory_factory(), ServiceHolder(), _settings(tmp_path), europe=europe,
                                          catchup=False, ready=lambda: ready["value"])  # fmt: skip
    try:
        daily, live = scheduler.get_job(sched_mod.JOB_ID).func, scheduler.get_job(sched_mod.LIVE_JOB_ID).func
        daily()
        live()  # skipped while setting up (would otherwise touch the DB)
        assert ran == [] and scheduler.get_job("catchup_ingest") is None
        ready["value"] = True
        daily()
        assert ran == ["ingest"]
    finally:
        scheduler.shutdown(wait=False)
