"""Frozen pre-kickoff prediction snapshots (DB, API, ingest freezing)."""

from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient
from mp_testkit import FakeService, frame, memory_factory

from matchpredict.api.app import create_app
from matchpredict.api.predictions import Models
from matchpredict.api.schemas import PredictionOut
from matchpredict.api.snapshots import (
    freeze_upcoming,
    frozen,
    is_pre_kickoff,
    kickoff_utc,
    needs_refresh,
)
from matchpredict.config import Settings
from matchpredict.db import repository as repo
from matchpredict.db.models import Match, PredictionSnapshot
from matchpredict.ingest.run import freeze_snapshots
from matchpredict.loader import ServiceHolder

DAY = dt.date(2026, 10, 24)
BEFORE = dt.datetime(2026, 10, 24, 10, 0, tzinfo=dt.UTC)  # 15:00 UK kickoff is 14:00 UTC
AFTER = dt.datetime(2026, 10, 24, 18, 0, tzinfo=dt.UTC)


def _settings(tmp_path) -> Settings:
    return Settings(database_url="sqlite:///:memory:", artifact_path=tmp_path / "m.joblib",
                    europe_artifact_path=tmp_path / "e.joblib", v2_artifact_path=tmp_path / "v2.joblib",
                    scheduler_enabled=False, football_data_org_key=None, rate_limit="1000/minute")  # fmt: skip


def _seed(factory) -> None:
    rows = [
        {"date": "2026-10-24", "time": "15:00", "league": "E0", "home": "Arsenal", "away": "Chelsea",
         "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 3.8},
        {"date": "2026-10-24", "time": "20:00", "league": "UCL", "home": "Arsenal", "away": "Lyon"},
        {"date": "2026-10-20", "time": "20:00", "league": "E0", "home": "Chelsea", "away": "Coventry",
         "home_goals": 3, "away_goals": 1},
    ]  # fmt: skip
    with factory() as s:
        repo.upsert_matches(s, frame(rows))


class SharpService(FakeService):
    """Same as FakeService but a very different prediction (stands for post-match ratings)."""

    def predict(self, home, away, league, date, odds=None):
        pred = super().predict(home, away, league, date, odds)
        return {**pred, "probs": {"home": 0.9, "draw": 0.05, "away": 0.05}}


def _client(tmp_path, factory, service, now):
    app = create_app(service, factory, _settings(tmp_path), today=lambda: DAY, now=lambda: now)
    return TestClient(app)


def test_kickoff_and_pre_kickoff_rules():
    m = Match(id=1, league="E0", date=DAY, time="15:00", home="A", away="B", status="scheduled")
    assert kickoff_utc(m) == dt.datetime(2026, 10, 24, 14, 0)
    assert is_pre_kickoff(m, BEFORE) and not is_pre_kickoff(m, AFTER)
    m.status = "live"
    assert not is_pre_kickoff(m, BEFORE)
    no_time = Match(id=2, league="E0", date=DAY, time=None, home="A", away="B", status="scheduled")
    assert is_pre_kickoff(no_time, dt.datetime(2026, 10, 23, 22, 0, tzinfo=dt.UTC))
    assert not is_pre_kickoff(no_time, BEFORE)


def test_needs_refresh_rules():
    pred = PredictionOut.model_validate(FakeService().predict("A", "B", "E0", DAY))
    snap = PredictionSnapshot(match_id=1, model_version=pred.model_version, payload=pred.model_dump_json(),
                              updated_at=dt.datetime(2026, 10, 24, 9, 50))  # fmt: skip
    assert needs_refresh(None, pred, BEFORE, 30)
    assert not needs_refresh(snap, pred, BEFORE, 30)
    assert needs_refresh(snap, pred, BEFORE + dt.timedelta(minutes=30), 30)
    assert needs_refresh(snap, pred.model_copy(update={"model_version": "new"}), BEFORE, 30)
    assert frozen(snap).probs.home == pred.probs.home
    snap.payload = "{broken"
    assert frozen(snap) is None


def test_api_freezes_before_kickoff_and_serves_the_snapshot_afterwards(tmp_path):
    factory = memory_factory()
    _seed(factory)
    with _client(tmp_path, factory, FakeService(), BEFORE) as c:
        items = c.get("/api/v1/fixtures", params={"date": "2026-10-24"}).json()
    e0 = next(i for i in items if i["league"] == "E0")
    assert e0["prediction"]["probs"]["home"] == 0.5 and e0["prediction_frozen_at"] is None
    with factory() as s:
        snaps = repo.snapshots_for(s, [e0["id"]])
        assert snaps[e0["id"]].prob_home == 0.5 and snaps[e0["id"]].kickoff == dt.datetime(2026, 10, 24, 14, 0)
        s.query(Match).filter(Match.id == e0["id"]).update({"status": "finished", "home_goals": 1, "away_goals": 0})
        s.commit()
    with _client(tmp_path, factory, SharpService(), AFTER) as c:
        items = c.get("/api/v1/fixtures", params={"date": "2026-10-24"}).json()
        old = c.get("/api/v1/fixtures", params={"date": "2026-10-20"}).json()
    e0 = next(i for i in items if i["league"] == "E0")
    assert e0["status"] == "finished"
    assert e0["prediction"]["probs"]["home"] == 0.5  # frozen pre-match view, not the re-price
    assert e0["prediction_frozen_at"] is not None
    # a finished match that was never priced before kickoff is re-priced and says so
    assert old[0]["prediction"]["probs"]["home"] == 0.9 and old[0]["prediction_frozen_at"] is None
    with factory() as s:  # nothing new frozen after kickoff
        assert s.query(PredictionSnapshot).count() == 1


def test_freeze_upcoming_prices_fixtures_and_respects_loaded_scopes(tmp_path):
    factory = memory_factory()
    _seed(factory)
    europe_only = Models(domestic=ServiceHolder(), europe=ServiceHolder(FakeService()))
    assert freeze_upcoming(factory, europe_only, BEFORE) == 1  # only the UCL fixture
    both = Models(domestic=ServiceHolder(FakeService()), europe=ServiceHolder(FakeService()))
    assert freeze_upcoming(factory, both, BEFORE) == 1  # E0 new, UCL fresh enough
    assert freeze_upcoming(factory, both, BEFORE + dt.timedelta(minutes=31)) == 2  # both refreshed
    assert freeze_upcoming(factory, Models(ServiceHolder(), ServiceHolder()), BEFORE) == 0
    settings = _settings(tmp_path)
    # domestic only (no UEFA model): just the E0 fixture is refreshed
    assert freeze_snapshots(factory, ServiceHolder(FakeService()), None, settings, now=BEFORE + dt.timedelta(hours=2)) == 1
    assert freeze_snapshots(None, None, None, settings) == 0  # failures are swallowed and logged


def test_upsert_snapshots_round_trip():
    factory = memory_factory()
    pred = FakeService().predict("A", "B", "E0", DAY)
    with factory() as s:
        assert repo.upsert_snapshots(s, []) == 0
        assert repo.upsert_snapshots(s, [{"match_id": 7, "model_version": "v", "prediction": pred, "kickoff": None}]) == 1
        pred2 = {**pred, "probs": {"home": 0.2, "draw": 0.3, "away": 0.5}}
        repo.upsert_snapshots(s, [{"match_id": 7, "model_version": "v2", "prediction": pred2, "kickoff": None}])
        snap = repo.snapshots_for(s, [7])[7]
        assert snap.prob_away == 0.5 and snap.model_version == "v2"
        assert s.query(PredictionSnapshot).count() == 1
