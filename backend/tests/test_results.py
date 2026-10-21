"""Results page: repository filters, live vs backtest separation, aggregates and the API."""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from mp_testkit import FakeService, frame, memory_factory

from matchpredict.api import results_stats as stats
from matchpredict.api.app import create_app
from matchpredict.config import Settings
from matchpredict.db import repository as repo
from matchpredict.db import results_repository as rr
from matchpredict.db.models import Match, PredictionSnapshot
from matchpredict.db.results_models import BacktestPrediction
from matchpredict.predictor.metrics import rps

TODAY = dt.date(2026, 10, 21)
MATCHES = [
    # key, row, source, probs (home, draw, away)
    ("m1", {"date": "2026-10-19", "time": "20:00", "league": "E0", "home": "Chelsea", "away": "Coventry",
            "home_goals": 3, "away_goals": 1, "odds_home": 1.5, "odds_draw": 4.5, "odds_away": 6.0},
     "live", (0.6, 0.25, 0.15)),
    ("m2", {"date": "2026-10-12", "league": "E0", "home": "Coventry", "away": "Chelsea",
            "home_goals": 0, "away_goals": 0}, "live", (0.3, 0.3, 0.4)),
    ("m3", {"date": "2026-10-15", "time": "20:00", "league": "UCL", "home": "Arsenal", "away": "Lyon",
            "home_goals": 1, "away_goals": 2}, "live", (0.5, 0.3, 0.2)),
    ("m4", {"date": "2026-10-24", "time": "15:00", "league": "E0", "home": "Arsenal", "away": "Chelsea"},
     "live", (0.45, 0.3, 0.25)),
    ("m5", {"date": "2024-03-10", "league": "F1", "home": "Lyon", "away": "Nice",
            "home_goals": 2, "away_goals": 0}, "backtest", (0.55, 0.25, 0.2)),
    ("m6", {"date": "2024-03-17", "league": "F1", "home": "Nice", "away": "Lyon",
            "home_goals": 1, "away_goals": 1}, "backtest", (0.4, 0.32, 0.28)),
]  # fmt: skip


def _seed(factory) -> dict[str, int]:
    ids: dict[str, int] = {}
    with factory() as s:
        repo.upsert_matches(s, frame([row for _, row, _, _ in MATCHES]))
        by_key = {(m.league, m.home, m.away): m.id for m in s.query(Match)}
        for key, row, source, (ph, pd_, pa) in MATCHES:
            mid = by_key[(row["league"], row["home"], row["away"])]
            ids[key] = mid
            if source == "live":
                payload = json.dumps({"model": "market" if "odds_home" in row else "base"})
                s.add(PredictionSnapshot(match_id=mid, model_version="v2-test", prob_home=ph, prob_draw=pd_,
                                         prob_away=pa, payload=payload,
                                         updated_at=dt.datetime(2026, 10, 1, 12, 0)))  # fmt: skip
            else:
                s.add(BacktestPrediction(match_id=mid, season=2023, model="base", model_version="v2-backtest-2023/24",
                                         prob_home=ph, prob_draw=pd_, prob_away=pa))  # fmt: skip
        s.commit()
    return ids


@pytest.fixture
def seeded():
    factory = memory_factory()
    return factory, _seed(factory)


@pytest.fixture
def client(tmp_path, seeded):
    factory, _ = seeded
    settings = Settings(database_url="sqlite:///:memory:", artifact_path=tmp_path / "m.joblib",
                        europe_artifact_path=tmp_path / "e.joblib", v2_artifact_path=tmp_path / "v2.joblib",
                        scheduler_enabled=False, football_data_org_key=None, rate_limit="1000/minute")  # fmt: skip
    app = create_app(FakeService(), factory, settings, today=lambda: TODAY)
    with TestClient(app) as c:
        yield c


def _ids(rows) -> list[int]:
    return [r.match_id for r in rows]


# --- repository ------------------------------------------------------------------


def test_live_and_backtest_are_never_mixed(seeded):
    factory, ids = seeded
    with factory() as s:
        live, n_live = rr.list_results(s, rr.ResultFilters(source="live"), 1, 50)
        bt, n_bt = rr.list_results(s, rr.ResultFilters(source="backtest"), 1, 50)
        both, n_all = rr.list_results(s, rr.ResultFilters(source="all"), 1, 50)
    assert _ids(live) == [ids["m1"], ids["m3"], ids["m2"]]  # newest first, pending excluded
    assert {r.source for r in live} == {"live"} and n_live == 3
    assert _ids(bt) == [ids["m6"], ids["m5"]] and {r.source for r in bt} == {"backtest"} and n_bt == 2
    assert n_all == 5 and [r.source for r in both] == ["live"] * 3 + ["backtest"] * 2


def test_pagination(seeded):
    factory, ids = seeded
    with factory() as s:
        page3, total = rr.list_results(s, rr.ResultFilters(source="all"), 3, 2)
        beyond, _ = rr.list_results(s, rr.ResultFilters(source="all"), 9, 2)
    assert total == 5 and _ids(page3) == [ids["m5"]] and beyond == []


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"outcome": "correct"}, ["m1"]),
        ({"outcome": "wrong"}, ["m3", "m2"]),
        ({"confidence": "high"}, ["m1"]),  # 0.6 counts as high
        ({"confidence": "low"}, ["m3", "m2"]),
        ({"league": "UEFA"}, ["m3"]),
        ({"league": "E0"}, ["m1", "m2"]),
        ({"team": "coven"}, ["m1", "m2"]),
        ({"team": "100%_"}, []),  # LIKE wildcards are escaped
        ({"date_from": dt.date(2026, 10, 13), "date_to": dt.date(2026, 10, 19)}, ["m1", "m3"]),
        ({"status": "pending"}, ["m4"]),
        ({"status": "pending", "source": "backtest"}, []),
        ({"source": "backtest", "outcome": "correct"}, ["m5"]),
    ],
)
def test_filters(seeded, filters, expected):
    factory, ids = seeded
    with factory() as s:
        rows, total = rr.list_results(s, rr.ResultFilters(**filters), 1, 50)
    assert _ids(rows) == [ids[k] for k in expected] and total == len(expected)


def test_counts(seeded):
    factory, _ = seeded
    with factory() as s:
        assert rr.available(s) == {"live": 3, "backtest": 2}
        assert rr.count_pending(s, rr.ResultFilters(source="all")) == 1
        assert rr.count_pending(s, rr.ResultFilters(source="backtest")) == 0


# --- aggregates ------------------------------------------------------------------


def test_summary_math_matches_metrics(seeded):
    factory, _ = seeded
    with factory() as s:
        out = stats.summarise(rr.metric_rows(s, rr.ResultFilters(source="live", status="pending")))
    probs = np.array([[0.6, 0.25, 0.15], [0.5, 0.3, 0.2], [0.3, 0.3, 0.4]])
    y = np.array([0, 2, 1])
    assert out["n"] == 3 and out["correct"] == 1 and out["accuracy"] == pytest.approx(1 / 3)
    assert out["rps"] == pytest.approx(rps(probs, y))
    assert out["by_outcome"] == {"H": {"n": 1, "correct": 1, "accuracy": 1.0},
                                 "D": {"n": 1, "correct": 0, "accuracy": 0.0},
                                 "A": {"n": 1, "correct": 0, "accuracy": 0.0}}  # fmt: skip
    assert out["by_predicted"]["H"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    buckets = {b["bucket"]: b for b in out["by_confidence_bucket"]}
    assert buckets["60-70%"]["n"] == 1 and buckets["60-70%"]["accuracy"] == 1.0
    assert buckets["50-60%"]["n"] == 1 and buckets["40-50%"]["avg_prob"] == pytest.approx(0.4)
    assert buckets["80-100%"]["n"] == 0 and buckets["80-100%"]["accuracy"] is None
    book = out["bookmaker"]  # only m1 has odds; the bookmaker favourite (home) won
    assert book["n"] == 1 and book["accuracy"] == 1.0 and book["model_accuracy"] == 1.0
    assert [lg["league"] for lg in out["by_league"]] == ["E0", "UCL"]
    assert [d["date"] for d in out["daily"]] == ["2026-10-12", "2026-10-15", "2026-10-19"]
    assert out["monthly"] == [{"month": "2026-10", "n": 3, "correct": 1, "accuracy": pytest.approx(1 / 3)}]


def test_summary_of_nothing():
    out = stats.summarise([])
    assert out["n"] == 0 and out["accuracy"] is None and out["bookmaker"] is None
    assert len(out["by_confidence_bucket"]) == len(stats.BUCKETS)


def test_ties_follow_numpy_argmax_and_no_odds_means_no_bookmaker():
    row = SimpleNamespace(prob_home=0.35, prob_draw=0.35, prob_away=0.3, home_goals=1, away_goals=1,
                          odds_home=None, odds_draw=None, odds_away=None, league="E0", date=dt.date(2025, 1, 1))  # fmt: skip
    out = stats.summarise([row])
    assert out["by_predicted"]["H"]["n"] == 1 and out["correct"] == 0 and out["bookmaker"] is None


# --- API -------------------------------------------------------------------------


def test_results_endpoint_items(client, seeded):
    _, ids = seeded
    body = client.get("/api/v1/results").json()
    assert body["total"] == 3 and body["page"] == 1 and body["page_size"] == 25
    first = body["items"][0]
    assert first["id"] == f"live-{ids['m1']}" and first["source"] == "live" and first["status"] == "finished"
    assert first["score"] == {"home": 3, "away": 1} and first["actual"] == "H" and first["predicted"] == "H"
    assert first["correct"] is True and first["prob_of_actual"] == pytest.approx(0.6)
    assert first["rps"] == pytest.approx(rps(np.array([[0.6, 0.25, 0.15]]), np.array([0])))
    assert first["model"] == "market" and first["model_version"] == "v2-test"
    assert first["kickoff"].startswith("2026-10-19T20:00:00+01:00") and first["frozen_at"].startswith("2026-10-01")
    draw = body["items"][2]
    assert draw["actual"] == "D" and draw["predicted"] == "A" and draw["correct"] is False
    assert draw["kickoff"] is None and draw["model"] == "base"


def test_results_endpoint_backtest_and_pending(client):
    bt = client.get("/api/v1/results", params={"source": "backtest", "page_size": 1}).json()
    assert bt["total"] == 2 and len(bt["items"]) == 1
    item = bt["items"][0]
    assert item["source"] == "backtest" and item["model_version"] == "v2-backtest-2023/24"
    assert item["frozen_at"] is None and item["model"] == "base"
    pending = client.get("/api/v1/results", params={"status": "pending"}).json()["items"]
    assert len(pending) == 1 and pending[0]["status"] == "pending"
    assert pending[0]["score"] is None and pending[0]["correct"] is None and pending[0]["actual"] is None
    assert pending[0]["predicted"] == "H" and pending[0]["prob_predicted"] == pytest.approx(0.45)


def test_summary_endpoint(client):
    live = client.get("/api/v1/results/summary").json()
    assert live["source"] == "live" and live["n"] == 3 and live["correct"] == 1
    assert live["pending"] == 1 and live["available"] == {"live": 3, "backtest": 2}
    assert live["bookmaker"]["n"] == 1
    every = client.get("/api/v1/results/summary", params={"source": "all", "league": "F1"}).json()
    assert every["n"] == 2 and every["correct"] == 1 and every["pending"] == 0
    assert every["first_date"] == "2024-03-10" and every["last_date"] == "2024-03-17"


@pytest.mark.parametrize(
    ("params", "status", "code"),
    [
        ({"page_size": 101}, 422, "VALIDATION_ERROR"),
        ({"page": 0}, 422, "VALIDATION_ERROR"),
        ({"source": "bogus"}, 422, "VALIDATION_ERROR"),
        ({"outcome": "maybe"}, 422, "VALIDATION_ERROR"),
        ({"date": "2026-10-10", "date_to": "2026-10-01"}, 422, "INVALID_DATE_RANGE"),
        ({"date": "2024-01-01", "date_to": "2026-01-01"}, 422, "DATE_RANGE_TOO_WIDE"),
        ({"league": "XX"}, 404, "LEAGUE_NOT_FOUND"),
        ({"team": "<script>"}, 422, "VALIDATION_ERROR"),
    ],
)
def test_validation_errors_use_the_envelope(client, params, status, code):
    paths = ["/api/v1/results"] if "page" in params or "page_size" in params else ["/api/v1/results", "/api/v1/results/summary"]
    for path in paths:
        res = client.get(path, params=params)
        assert res.status_code == status and res.json()["error"]["code"] == code


def test_backtest_allows_wide_ranges(client):
    res = client.get("/api/v1/results", params={"source": "backtest", "date": "2022-07-01", "date_to": "2026-06-30"})
    assert res.status_code == 200 and res.json()["total"] == 2
