"""Gate-aware loading of the v2 service and the weekly retrain promotion rule."""

from __future__ import annotations

import subprocess
from pathlib import Path

import joblib
import pytest
from mp_testkit import FakeService

from matchpredict import loader
from matchpredict.config import Settings
from matchpredict.ingest import scheduler as sched_mod
from matchpredict.predictor.v2 import retrain


class FakeV2(FakeService):
    def __init__(self, gate: dict, path: Path | None = None) -> None:
        super().__init__()
        self.model_version = "v2-fake"
        self.gate = gate
        self.artifact_path = path


@pytest.fixture(autouse=True)
def _clean_cache():
    loader.clear_v2_cache()
    yield
    loader.clear_v2_cache()


def _settings(tmp_path: Path, **kw) -> Settings:
    base = {"database_url": "sqlite:///:memory:", "artifact_path": tmp_path / "m.joblib",
            "europe_artifact_path": tmp_path / "e.joblib", "v2_artifact_path": tmp_path / "model_v2.joblib",
            "scheduler_enabled": False, "football_data_org_key": None}  # fmt: skip
    return Settings(**{**base, **kw})


def test_gated_loader_serves_v2_only_for_passed_scopes(tmp_path, monkeypatch):
    v2 = FakeV2({"domestic": True, "europe": False})
    monkeypatch.setattr(loader, "load_v2_service", lambda path: v2)
    v1_dom, v1_eur = FakeService(), FakeService()
    dom = loader.gated_loader("domestic", tmp_path / "v2", lambda p: v1_dom)
    eur = loader.gated_loader("europe", tmp_path / "v2", lambda p: v1_eur)
    assert dom(tmp_path / "x") is v2
    assert eur(tmp_path / "x") is v1_eur
    monkeypatch.setattr(loader, "load_v2_service", lambda path: None)
    assert dom(tmp_path / "x") is v1_dom
    with pytest.raises(ValueError):
        loader.gated_loader("moon", tmp_path, lambda p: None)


def test_load_v2_service_missing_broken_and_cached(tmp_path):
    assert loader.load_v2_service(tmp_path / "missing.joblib") is None
    bad = tmp_path / "bad.joblib"
    joblib.dump({"artifact_version": 1}, bad)
    assert loader.load_v2_service(bad) is None


def test_holders_and_artifact_path_of(tmp_path):
    settings = _settings(tmp_path)
    assert loader.domestic_holder(settings).artifact_path == settings.artifact_path
    assert loader.europe_holder(settings).artifact_path == settings.europe_artifact_path
    off = _settings(tmp_path, v2_enabled=False)
    assert loader.domestic_holder(off).ensure_loaded() is None  # v1 artifact missing too
    assert loader.artifact_path_of(FakeV2({}, tmp_path / "own.joblib"), tmp_path / "d") == tmp_path / "own.joblib"
    assert loader.artifact_path_of(FakeService(), tmp_path / "d") == tmp_path / "d"


def test_holder_reload_and_stage_passthrough(tmp_path):
    calls = []

    class StageService(FakeService):
        def predict(self, home, away, league, date, odds=None, stage=None):
            calls.append(stage)
            return super().predict(home, away, league, date, odds)

    services = iter([StageService(), StageService()])
    holder = loader.ServiceHolder(artifact_path=tmp_path / "x", loader=lambda p: next(services))
    first = holder.ensure_loaded()
    holder.predict("A", "B", "UCL", "2026-10-21", None, stage="ko")
    holder.predict("A", "B", "E0", "2026-10-21")
    assert calls == ["ko", None]
    assert holder.reload() is not first


def test_compare_rule():
    cur = {"domestic_log_loss": 1.00, "europe_log_loss": 0.97}
    assert retrain.compare(cur, {"domestic_log_loss": 1.002, "europe_log_loss": 0.96}, 0.003)[0]
    ok, why = retrain.compare(cur, {"domestic_log_loss": 1.01, "europe_log_loss": 0.96}, 0.003)
    assert not ok and "domestic" in why
    assert not retrain.compare(cur, {"domestic_log_loss": 1.0}, 0.003)[0]  # europe missing
    assert retrain.compare({}, {"domestic_log_loss": 1.0, "europe_log_loss": float("nan")}, 0.0)[0]


def _fake_run(validation: dict | None, code: int = 0):
    def run(cmd, **kwargs):
        if validation is not None:
            joblib.dump({"validation": validation, "model_version": "v2-new"}, Path(cmd[cmd.index("--out") + 1]))
        return subprocess.CompletedProcess(cmd, code, stdout="", stderr="boom" if code else "")

    return run


def test_weekly_retrain_promotes_rejects_and_skips(tmp_path):
    settings = _settings(tmp_path)
    current = settings.v2_artifact_path
    assert retrain.weekly_retrain(settings).status == "skipped"  # no artifact yet
    joblib.dump({"validation": {"domestic_log_loss": 1.0, "europe_log_loss": 0.97}, "model_version": "v2-old"}, current)
    current.with_name(current.name + ".state").write_text("old state")
    rejected = retrain.weekly_retrain(settings, run=_fake_run({"domestic_log_loss": 1.1, "europe_log_loss": 0.97}))
    assert rejected.status == "rejected" and joblib.load(current)["model_version"] == "v2-old"
    reloads = []
    holder = loader.ServiceHolder(artifact_path=tmp_path / "x", loader=lambda p: reloads.append(p) or FakeService())
    promoted = retrain.weekly_retrain(settings, [holder], run=_fake_run({"domestic_log_loss": 0.99, "europe_log_loss": 0.97}))
    assert promoted.status == "promoted" and joblib.load(current)["model_version"] == "v2-new"
    assert not current.with_name(current.name + ".state").exists() and reloads
    failed = retrain.weekly_retrain(settings, run=_fake_run(None, code=1))
    assert failed.status == "failed" and "boom" in failed.reason
    crashed = retrain.weekly_retrain(settings, run=lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert crashed.status == "failed" and "disk" in crashed.reason
    lines = current.with_name("retrain_log.jsonl").read_text().splitlines()
    assert len(lines) == 5  # every enabled run is logged, including the first skip
    disabled = retrain.weekly_retrain(_settings(tmp_path, retrain_enabled=False))
    assert disabled.status == "skipped" and "disabled" in disabled.reason


def test_scheduler_registers_the_weekly_retrain_job(tmp_path, monkeypatch):
    from mp_testkit import memory_factory

    monkeypatch.setattr(sched_mod, "needs_catchup", lambda *a, **k: False)  # no real ingest

    scheduler = sched_mod.start_scheduler(memory_factory(), loader.ServiceHolder(), _settings(tmp_path))
    try:
        job = scheduler.get_job(sched_mod.RETRAIN_JOB_ID)
        assert job is not None and "sun" in str(job.trigger)
    finally:
        scheduler.shutdown(wait=False)
    off = sched_mod.start_scheduler(memory_factory(), loader.ServiceHolder(), _settings(tmp_path, retrain_enabled=False))
    try:
        assert off.get_job(sched_mod.RETRAIN_JOB_ID) is None
    finally:
        off.shutdown(wait=False)
