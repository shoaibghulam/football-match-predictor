import time

import numpy as np
import pandas as pd
import pytest

from matchpredict.contracts import MATCH_COLUMNS, PredictionService
from matchpredict.predictor.data import load_history
from matchpredict.predictor.service import ModelService
from matchpredict.predictor.train import (
    TrainConfig,
    fit_artifact,
    format_tables,
    main,
    prepare,
    walk_forward,
)

TINY = TrainConfig(iterations=40, learning_rate=0.2, early_stopping_rounds=10, goals_iterations=20)
TEAMS = [f"Team{i}" for i in range(10)]


def _raw_history(start="2021-08-01", weeks=250, seed=7) -> pd.DataFrame:
    """Synthetic football-data style rows: one league, 10 teams, weekly rounds."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(-0.6, 0.6, len(TEAMS))))
    rows = []
    for week in range(weeks):
        day = pd.Timestamp(start) + pd.Timedelta(days=7 * week)
        order = rng.permutation(TEAMS)
        for h, a in zip(order[::2], order[1::2]):
            lh, la = np.exp(0.3 + strength[h] - strength[a]), np.exp(0.05 + strength[a] - strength[h])
            p = np.array([0.45 + 0.3 * (strength[h] - strength[a]), 0.27, 0.0])
            p[2] = 1.0 - p[0] - p[1]
            odds = np.round(0.95 / np.clip(p, 0.05, None), 2)
            rows.append({
                "Division": "E0", "MatchDate": day.strftime("%Y-%m-%d"), "MatchTime": "15:00:00",
                "HomeTeam": h, "AwayTeam": a, "FTHome": rng.poisson(lh), "FTAway": rng.poisson(la),
                "HomeShots": rng.poisson(12), "AwayShots": rng.poisson(10),
                "OddHome": odds[0], "OddDraw": odds[1], "OddAway": odds[2],
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def history_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "Matches.csv"
    _raw_history().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def history(history_csv):
    return load_history(history_csv)


@pytest.fixture(scope="module")
def service(history):
    cutoff = history["date"].max() - pd.Timedelta(days=14)
    artifact = fit_artifact(history[history["date"] <= cutoff], TINY, model_version="test-v1")
    return ModelService(artifact)


def test_service_satisfies_protocol(service):
    typed: PredictionService = service  # static check; protocol is not runtime_checkable
    for name in ("predict", "update_results", "known_teams", "team_ratings", "last_result_date"):
        assert hasattr(typed, name)
    assert service.model_version == "test-v1"


def test_predict_base_shape_and_rounding(service):
    pred = service.predict("Team9", "Team0", "E0", pd.Timestamp("2026-07-01"))
    probs = pred["probs"]
    assert pred["model"] == "base"
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(round(v, 4) == v for v in probs.values())
    assert probs["home"] > probs["away"]  # strongest team at home vs weakest
    assert pred["expected_goals"]["home"] > pred["expected_goals"]["away"]
    assert 0 < pred["over_2_5"] < 1 and 0 < pred["btts"] < 1
    assert len(pred["top_scores"]) == 5
    ps = [s["p"] for s in pred["top_scores"]]
    assert ps == sorted(ps, reverse=True)


def test_predict_uses_market_model_only_with_valid_odds(service):
    date = pd.Timestamp("2026-07-01")
    assert service.predict("Team1", "Team2", "E0", date, odds=(2.1, 3.3, 3.6))["model"] == "market"
    assert service.predict("Team1", "Team2", "E0", date, odds=(0.0, 3.3, 3.6))["model"] == "base"
    assert service.predict("Team1", "Team2", "E0", date, odds=(float("nan"), 3.3, 3.6))["model"] == "base"


def test_predict_unknown_team_in_unknown_league_raises(service):
    with pytest.raises(KeyError):
        service.predict("Nobody FC", "Team1", "ZZZ", pd.Timestamp("2026-07-01"))


def test_predict_seeds_new_team_in_known_league(service):
    pred = service.predict("Promoted FC", "Team9", "E0", pd.Timestamp("2026-07-01"))
    assert pred["probs"]["away"] > pred["probs"]["home"]  # seeded below a strong side
    assert "Promoted FC" not in service.known_teams()  # predicting never mutates state


def test_update_results_dedupes_per_league(history):
    cutoff = history["date"].max() - pd.Timedelta(days=14)
    svc = ModelService(fit_artifact(history[history["date"] <= cutoff], TINY))
    assert svc.league_last_date["E0"] == cutoff.normalize() or svc.league_last_date["E0"] <= cutoff
    lagging = history.head(20).assign(league="USA", date=lambda d: d["date"] + pd.Timedelta(days=30))
    applied = svc.update_results(pd.concat([history, lagging]))
    assert applied == int((history["date"] > cutoff).sum()) + len(lagging)
    assert "USA" in svc.league_last_date
    assert svc.last_result_date == history["date"].max()  # global max kept for the contract
    older_usa = lagging.assign(date=lagging["date"].min() - pd.Timedelta(days=1))
    assert svc.update_results(older_usa) == 0


def test_predict_is_fast(service):
    service.predict("Team3", "Team4", "E0", pd.Timestamp("2026-07-01"))  # warm-up
    started = time.perf_counter()
    for _ in range(10):
        service.predict("Team3", "Team4", "E0", pd.Timestamp("2026-07-01"), odds=(2.0, 3.4, 3.9))
    assert (time.perf_counter() - started) / 10 < 0.05


def test_known_teams_and_ratings(service):
    teams = service.known_teams()
    assert set(teams) == set(TEAMS)
    assert teams["Team0"] == "E0"
    ratings = service.team_ratings("Team9")
    assert {"elo", "pi_home", "pi_away", "form5"} <= set(ratings)
    assert ratings["elo"] > service.team_ratings("Team0")["elo"]
    with pytest.raises(KeyError):
        service.team_ratings("Nobody FC")


def test_update_results_applies_only_newer_matches(history, tmp_path):
    cutoff = history["date"].max() - pd.Timedelta(days=14)
    svc = ModelService(fit_artifact(history[history["date"] <= cutoff], TINY))
    before = svc.team_ratings("Team5")["played"]
    assert svc.update_results(history) == int((history["date"] > cutoff).sum())
    assert svc.last_result_date == history["date"].max()
    assert svc.team_ratings("Team5")["played"] > before
    assert svc.update_results(history) == 0  # idempotent
    upcoming = history.tail(1).assign(date=history["date"].max() + pd.Timedelta(days=7),
                                      home_goals=np.nan, away_goals=np.nan)
    assert svc.update_results(upcoming[list(MATCH_COLUMNS)]) == 0  # no result yet


def test_save_and_load_roundtrip(service, tmp_path):
    path = tmp_path / "model.joblib"
    service.save(path)
    loaded = ModelService.load(path)
    args = ("Team2", "Team7", "E0", pd.Timestamp("2026-07-01"))
    assert loaded.predict(*args) == service.predict(*args)
    assert loaded.last_result_date == service.last_result_date


def test_walk_forward_report_and_tables(history):
    ds = prepare(history, TINY)
    report = walk_forward(ds, (2024,), TINY)
    pooled = report["pooled"]
    assert set(pooled) == {"base_rate", "elo_logit", "bookmaker", "goals", "base", "market"}
    assert all(0 < m["rps"] < 0.5 for m in pooled.values())
    assert "E0" in report["per_league_last_fold"]
    assert "| market |" in format_tables(report)


def test_cli_quick_writes_artifact(history_csv, tmp_path):
    out = tmp_path / "artifacts" / "model.joblib"
    main(["--quick", "--data", str(history_csv), "--out", str(out)])
    assert out.exists()
    assert (out.parent / "metrics.json").exists()
    svc = ModelService.load(out)
    assert svc.predict("Team1", "Team2", "E0", pd.Timestamp("2026-08-01"))["probs"]
