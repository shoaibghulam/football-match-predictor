import datetime as dt

import pytest
from europe_kit import make_service
from fastapi.testclient import TestClient
from mp_testkit import FakeService, frame, memory_factory

from matchpredict.api.app import create_app
from matchpredict.api.predictions import estimate_minute
from matchpredict.config import Settings
from matchpredict.db import repository as repo

TODAY = dt.date(2026, 10, 20)
NOW = dt.datetime(2026, 10, 20, 19, 40, tzinfo=dt.UTC)  # 40' after a 20:00 UK kick-off


def _settings(tmp_path, **kw) -> Settings:
    base = {"database_url": "sqlite:///:memory:", "artifact_path": tmp_path / "missing.joblib",
            "europe_artifact_path": tmp_path / "missing-europe.joblib", "v2_artifact_path": tmp_path / "no-v2.joblib", "scheduler_enabled": False,
            "rate_limit": "1000/minute", "football_data_org_key": None}  # fmt: skip
    return Settings(**{**base, **kw})


def _seed(factory):
    rows = [
        {"date": "2026-10-20", "time": "20:00", "league": "UCL", "home": "Paris SG", "away": "Barcelona",
         "home_goals": 3, "away_goals": 2},
        {"date": "2026-10-20", "time": "17:45", "league": "UCL", "home": "Celtic", "away": "Club Brugge",
         "home_goals": 1, "away_goals": 0},
        {"date": "2026-10-20", "time": "20:00", "league": "UECL", "home": "Mystery Town", "away": "Benfica"},
        {"date": "2026-10-20", "time": "15:00", "league": "E0", "home": "Arsenal", "away": "Chelsea"},
        {"date": "2026-10-20", "time": "12:30", "league": "F1", "home": "Paris SG", "away": "Lens"},
        {"date": "2026-10-20", "time": "18:00", "league": "SC0", "home": "Celtic", "away": "Lens"},
    ]  # fmt: skip
    df = frame(rows)
    df["status"] = ["live", "finished", "scheduled", "scheduled", "scheduled", "scheduled"]
    df["stage"] = ["league", "league", "league", None, None, None]
    with factory() as s:
        repo.upsert_matches(s, df)


@pytest.fixture(scope="module")
def europe_service():
    return make_service()


@pytest.fixture
def client(tmp_path, europe_service):
    factory = memory_factory()
    _seed(factory)
    service = FakeService({"Arsenal": "E0", "Chelsea": "E0", "Celtic": "SC0", "Paris SG": "F1", "Lens": "F1"})
    app = create_app(service, factory, _settings(tmp_path, football_data_org_key="k"), today=lambda: TODAY,
                     europe_service=europe_service, now=lambda: NOW)  # fmt: skip
    with TestClient(app) as c:
        c.fake = service
        yield c


def test_health_reports_europe_and_key(client, tmp_path):
    body = client.get("/api/v1/health").json()
    assert body["europe_model_loaded"] is True and body["europe_model_version"] == "europe-test"
    assert body["football_data_org_key_configured"] is True and body["live_matches"] == 1
    assert "k" not in {str(v) for v in body.values()}


def test_fixtures_include_uefa_with_europe_predictions_live_and_flags(client):
    body = client.get("/api/v1/fixtures").json()
    by = {(f["home"], f["away"]): f for f in body}
    psg = by[("Paris SG", "Barcelona")]
    assert psg["competition_type"] == "europe" and psg["featured"] and psg["stage"] == "league"
    assert psg["status"] == "live" and psg["score"] == {"home": 3, "away": 2}
    assert psg["live"] == {"minute": 40, "minute_estimated": True, "score": {"home": 3, "away": 2}}
    assert psg["prediction"]["model"] == "europe" and psg["prediction"]["confidence"] == "high"
    # every UEFA fixture is priced, even with an unknown club (low confidence prior)
    uefa = [f for f in body if f["competition_type"] == "europe"]
    assert len(uefa) == 3 and all(f["prediction"] is not None for f in uefa)
    assert by[("Mystery Town", "Benfica")]["prediction"]["confidence"] == "low"
    ars = by[("Arsenal", "Chelsea")]
    assert ars["competition_type"] == "domestic" and ars["prediction"]["model"] == "base"
    assert ars["live"] is None and ars["featured"] is False
    assert by[("Paris SG", "Lens")]["prediction"]["model"] == "base"
    # Celtic (SC0) v Lens (F1): different countries -> European model fallback
    assert by[("Celtic", "Lens")]["prediction"]["model"] == "europe"


def test_fixtures_fall_back_to_europe_when_domestic_fails(client):
    client.fake.fail_for.add("Paris SG")
    body = client.get("/api/v1/fixtures", params={"league": "F1"}).json()
    assert body[0]["prediction"]["model"] == "europe"


def test_featured_orders_live_then_uefa_then_big_domestic(client):
    body = client.get("/api/v1/featured").json()
    names = [(f["home"], f["away"]) for f in body]
    assert names[0] == ("Paris SG", "Barcelona")
    assert set(names[1:3]) == {("Celtic", "Club Brugge"), ("Mystery Town", "Benfica")}
    assert all(f["featured"] for f in body) and len(body) <= 12
    assert client.get("/api/v1/featured", params={"date": "2026-11-20"}).json() == []


def test_predict_uefa_and_cross_country(client):
    body = client.get("/api/v1/predict", params={"home": "FC Barcelona", "away": "Paris SG", "league": "UCL"}).json()
    assert body["home"] == "Barcelona" and body["league"] == "UCL" and body["model"] == "europe"
    cross = client.get("/api/v1/predict", params={"home": "Arsenal", "away": "Lens"}).json()
    assert cross["model"] == "europe" and cross["league"] == "UCL"
    assert client.get("/api/v1/predict", params={"home": "Arsenal", "away": "Chelsea"}).json()["model"] == "base"
    unknown_domestic = client.get("/api/v1/predict", params={"home": "Bayern Munich", "away": "Arsenal"}).json()
    assert unknown_domestic["model"] == "europe" and unknown_domestic["home"] == "Bayern Munich"
    assert client.get("/api/v1/predict", params={"home": "Nobody FC", "away": "Arsenal", "league": "UCL"}).status_code == 404
    same = client.get("/api/v1/predict", params={"home": "FC Barcelona", "away": "Barcelona", "league": "UCL"})
    assert same.status_code == 422 and same.json()["error"]["code"] == "SAME_TEAM"


def test_predict_uefa_without_europe_model(tmp_path):
    factory = memory_factory()
    app = create_app(FakeService(), factory, _settings(tmp_path), today=lambda: TODAY)
    with TestClient(app) as c:
        resp = c.get("/api/v1/predict", params={"home": "Arsenal", "away": "Lyon", "league": "UCL"})
        assert resp.status_code == 503 and resp.json()["error"]["code"] == "MODEL_NOT_LOADED"
        body = c.get("/api/v1/health").json()
        assert body["europe_model_loaded"] is False and body["football_data_org_key_configured"] is False
        assert c.get("/api/v1/predict", params={"home": "Nobody", "away": "Arsenal"}).status_code == 404


def test_live_minute_from_feed(tmp_path, europe_service):
    factory = memory_factory()
    df = frame([{"date": "2026-10-20", "time": "20:00", "league": "UCL", "home": "Paris SG",
                 "away": "Barcelona", "home_goals": 0, "away_goals": 0}])  # fmt: skip
    df["status"], df["minute"] = ["live"], [67]
    with factory() as s:
        repo.upsert_matches(s, df)
    app = create_app(FakeService(), factory, _settings(tmp_path), today=lambda: TODAY,
                     europe_service=europe_service, now=lambda: NOW)  # fmt: skip
    with TestClient(app) as c:
        live = c.get("/api/v1/featured").json()[0]["live"]
    assert live == {"minute": 67, "minute_estimated": False, "score": {"home": 0, "away": 0}}


@pytest.mark.parametrize(
    ("minutes", "expected"), [(-5, None), (0, 1), (30, 30), (50, 45), (75, 60), (200, 90)]
)
def test_estimate_minute(minutes, expected):
    ko = dt.datetime(2026, 10, 20, 19, tzinfo=dt.UTC)
    assert estimate_minute(ko, ko + dt.timedelta(minutes=minutes)) == expected
    assert estimate_minute(None, ko) is None
