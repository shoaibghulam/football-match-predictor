"""v2 training pipeline (tiny synthetic world) and V2Service predict / update / persistence."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from v2_kit import TINY, synthetic_frame

from matchpredict.predictor.v2 import train as v2train
from matchpredict.predictor.v2.data import build_canonicaliser, prepare_frame, standardise
from matchpredict.predictor.v2.service import V2Service


@pytest.fixture(scope="module")
def dataset():
    raw = synthetic_frame(seasons=range(2016, 2022))
    names = build_canonicaliser(standardise(raw))
    return v2train.prepare(prepare_frame(raw, names), names, TINY, None)


@pytest.fixture(scope="module")
def artifact_path(dataset, tmp_path_factory):
    artifact = v2train.fit_artifact(dataset, TINY, gate={"domestic": True, "europe": True}, model_version="v2-test")
    path = tmp_path_factory.mktemp("v2") / "model_v2.joblib"
    v2train.save_artifact(artifact, path)
    return path


@pytest.fixture
def service(artifact_path) -> V2Service:
    sidecar = artifact_path.with_name(artifact_path.name + ".state")
    sidecar.unlink(missing_ok=True)
    return V2Service.load(artifact_path)


def test_walk_forward_report_and_gate(dataset):
    report = v2train.walk_forward(dataset, TINY, seasons=(2020, 2021))
    pooled = report["pooled"]
    assert set(pooled) >= {"domestic_odds", "domestic_all", "europe", "all", "domestic_odds_v1rows", "europe_v1rows"}
    assert pooled["domestic_odds"]["v2_market"]["n"] == pooled["domestic_odds"]["bookmaker"]["n"] > 0
    assert pooled["europe"]["v2_base"]["n"] == 24
    for m in pooled["all"].values():
        assert 0 < m["rps"] < 0.5 and len(m["ece"]) == 3
    assert set(report["per_fold"]) == {"2020/21", "2021/22"}
    assert set(report["per_competition"]) >= {"UCL", "main draw"}
    assert report["goal_markets"]["over_2_5"]["n"] == len(dataset.matches[dataset.season >= 2020])
    assert report["reliability"]["domestic_v2_served"]["home"]
    worse = {"n": 1, "rps": 0.5, "log_loss": 2.0}
    better = {"n": 1, "rps": 0.01, "log_loss": 0.1}
    gate = v2train.gate_decision(report, {"domestic_market": worse, "domestic_base": worse, "europe": worse})
    assert gate["domestic"] and gate["europe"]
    gate = v2train.gate_decision(report, {"domestic_market": better, "domestic_base": worse, "europe": None})
    assert not gate["domestic"] and not gate["europe"]
    text = v2train.format_report(report, {"x": worse}, gate)
    assert "### European (UCL/UEL/UECL)" in text and "Gate: domestic=False" in text


def test_selection_experiments_run_on_the_selection_split(dataset):
    out = v2train.selection_experiments(dataset, TINY)
    assert out["n_domestic"] > 0 and out["n_europe"] == 12
    assert {"v1 features only", "+ ClubElo (full)", "member goals", "pool base only", "bookmaker (dom)"} <= set(out)


def test_v1_reference_reads_metrics_files(tmp_path, monkeypatch):
    dom, eur = tmp_path / "m.json", tmp_path / "e.json"
    dom.write_text(json.dumps({"pooled": {"base": {"n": 1, "rps": 0.2, "log_loss": 1.0}}}))
    eur.write_text(json.dumps({"walk_forward": {"overall": {"europe_lr_dom": {"n": 2, "rps": 0.21, "log_loss": 0.97}}}}))
    monkeypatch.setattr(v2train, "V1_METRICS", dom)
    monkeypatch.setattr(v2train, "V1_EUROPE_METRICS", eur)
    ref = v2train.v1_reference()
    assert ref["domestic_base"]["rps"] == 0.2 and ref["europe"]["n"] == 2 and ref["domestic_market"] is None
    monkeypatch.setattr(v2train, "V1_METRICS", tmp_path / "missing.json")
    assert v2train.v1_reference()["domestic_base"] is None


def test_artifact_contents(artifact_path):
    artifact = joblib.load(artifact_path)
    assert artifact["artifact_version"] == 2 and artifact["model_version"] == "v2-test"
    val = artifact["validation"]
    assert val["n_domestic"] > 0 and 0.5 < val["domestic_log_loss"] < 1.5
    assert artifact["league_last_date"]["UCL"] <= artifact["last_result_date"]


def _probs_ok(pred) -> None:
    p = pred["probs"]
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0 < v < 1 for v in p.values())
    assert len(pred["top_scores"]) == 5 and 0 < pred["over_2_5"] < 1 and 0 < pred["btts"] < 1


def test_predict_domestic_market_and_europe(service):
    day = pd.Timestamp("2022-02-01")
    base = service.predict("A1", "A10", "E0", day)
    market = service.predict("A1", "A10", "E0", day, odds=(1.5, 4.0, 7.0))
    _probs_ok(base)
    _probs_ok(market)
    assert base["model"] == "base" and market["model"] == "market"
    assert base["probs"]["home"] > base["probs"]["away"]
    assert base["confidence"] == "high" and base["model_version"] == "v2-test"
    euro = service.predict("A1", "B1", "UCL", day)
    assert euro["model"] == "europe" and euro["probs"]["home"] > euro["probs"]["away"]
    unknown = service.predict("Brand New FC", "B1", "UEL", day, stage="qual")
    assert unknown["confidence"] == "low"
    final = service.predict("A1", "B1", "UCL", day, stage="final")
    assert final["expected_goals"]["home"] > 0
    promoted = service.predict("Newcomer", "A1", "E0", day)  # unseen club in a known league
    assert promoted["probs"]["away"] > promoted["probs"]["home"]
    with pytest.raises(KeyError):
        service.predict("Nobody", "A1", "XX9", day)


def test_names_queries_and_ratings(service):
    assert service.resolve("A1") == "A1"
    assert service.resolve("a1") == "A1"
    assert service.resolve("Completely Unknown Club Name") is None
    known = service.known_teams()
    assert known["A1"] == "E0" and known["B1"] == "SP1"
    assert service.strength("A1") > service.strength("A10")
    assert service.strength("nobody at all") is None
    ratings = service.team_ratings("A1")
    assert {"elo", "pi_home", "form5", "attack", "defence", "played", "country_offset"} <= set(ratings)
    with pytest.raises(KeyError):
        service.team_ratings("nobody at all")


def _result(day: str, league: str, home: str, away: str, hg: float, ag: float) -> pd.DataFrame:
    row = {"date": pd.Timestamp(day), "time": None, "league": league, "home": home, "away": away,
           "home_goals": hg, "away_goals": ag}  # fmt: skip
    return pd.DataFrame([row])


def test_update_results_dedupes_and_handles_partial_days(service, artifact_path):
    last = service.last_result_date
    day = (last + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    before = service.state.rating.elo("A1")
    first = _result(day, "UCL", "A1", "B1", 3, 0)
    assert service.update_results(first) == 1
    assert service.state.rating.elo("A1") > before
    assert service.update_results(first) == 0  # applied once
    late = pd.concat([first, _result(day, "UCL", "A2", "B2", 0, 1)])  # a later result of the same day
    assert service.update_results(late) == 1
    old = _result("2016-09-01", "E0", "A1", "A2", 9, 0)
    assert service.update_results(old) == 0  # long before the league's cutoff
    upcoming = _result(day, "E0", "A3", "A4", np.nan, np.nan)
    assert service.update_results(upcoming) == 0
    assert service.last_result_date == pd.Timestamp(day)
    sidecar = service.save()
    assert sidecar.name == "model_v2.joblib.state"
    reloaded = V2Service.load(artifact_path)
    assert reloaded.state.rating.elo("A1") == pytest.approx(service.state.rating.elo("A1"))
    assert reloaded.update_results(first) == 0
    assert reloaded.state.clubelo is service.state.clubelo or reloaded.state.clubelo is None


def test_sidecar_for_another_version_is_ignored(service, artifact_path):
    sidecar = artifact_path.with_name(artifact_path.name + ".state")
    joblib.dump({"model_version": "other", "state": None}, sidecar)
    fresh = V2Service.load(artifact_path)
    assert fresh.last_result_date == service.last_result_date
    sidecar.write_text("garbage")
    assert V2Service.load(artifact_path).model_version == "v2-test"
    sidecar.unlink()


def test_rejects_non_v2_artifacts(tmp_path):
    with pytest.raises(ValueError):
        V2Service({"artifact_version": 1})
    svc = V2Service.__new__(V2Service)
    svc.artifact_path = None
    with pytest.raises(ValueError):
        V2Service.save(svc)
