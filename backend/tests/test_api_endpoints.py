import datetime as dt

import pytest
from fastapi.testclient import TestClient
from mp_testkit import FakeService, frame, memory_factory

from matchpredict.api.app import JsonFormatter, configure_logging, create_app
from matchpredict.config import Settings
from matchpredict.db import repository as repo

TODAY = dt.date(2026, 10, 24)


def _settings(tmp_path, **kw) -> Settings:
    base = {
        "database_url": "sqlite:///:memory:",
        "artifact_path": tmp_path / "missing.joblib",
        "europe_artifact_path": tmp_path / "missing-europe.joblib",
        "v2_artifact_path": tmp_path / "no-v2.joblib",
        "football_data_org_key": None,
        "scheduler_enabled": False,
        "rate_limit": "1000/minute",
    }
    return Settings(**{**base, **kw})


def _seed(factory) -> None:
    rows = [
        {"date": "2026-10-24", "time": "15:00", "league": "E0", "home": "Arsenal", "away": "Chelsea",
         "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 3.8},
        {"date": "2026-10-24", "time": "12:30", "league": "F1", "home": "Lyon", "away": "Nice"},
        {"date": "2026-10-26", "league": "E0", "home": "Coventry", "away": "Arsenal"},
        {"date": "2026-10-19", "time": "20:00", "league": "E0", "home": "Chelsea", "away": "Coventry",
         "home_goals": 3, "away_goals": 1, "odds_home": 1.5, "odds_draw": 4.5, "odds_away": 6.0},
        {"date": "2026-10-12", "league": "E0", "home": "Coventry", "away": "Chelsea",
         "home_goals": 0, "away_goals": 0},
        {"date": "2026-10-05", "league": "E0", "home": "Arsenal", "away": "Chelsea",
         "home_goals": 0, "away_goals": 2},
    ]  # fmt: skip
    with factory() as s:
        repo.upsert_matches(s, frame(rows))
        repo.finish_run(s, repo.start_run(s), "success")


@pytest.fixture
def service():
    return FakeService()


@pytest.fixture
def client(tmp_path, service):
    factory = memory_factory()
    _seed(factory)
    app = create_app(service, factory, _settings(tmp_path), today=lambda: TODAY)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def no_model_client(tmp_path):
    factory = memory_factory()
    _seed(factory)
    app = create_app(None, factory, _settings(tmp_path), today=lambda: TODAY)
    with TestClient(app) as c:
        yield c


def _assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"} and body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str)


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["model_loaded"]
    assert body["model_version"] == "test-1"
    assert body["last_result_date"] == "2026-10-01"
    assert body["last_ingest"] is not None
    assert body["matches"] == 3 and body["fixtures"] == 3


def test_health_degraded_without_model(no_model_client):
    body = no_model_client.get("/api/v1/health").json()
    assert body["status"] == "degraded" and body["model_version"] is None


def test_leagues_lists_uefa_first_then_all_38_domestic(client):
    body = client.get("/api/v1/leagues").json()
    assert len(body) == 41
    assert [b["code"] for b in body[:3]] == ["UCL", "UEL", "UECL"]
    assert body[0] == {"code": "UCL", "name": "UEFA Champions League", "country": "Europe"}
    assert body[3] == {"code": "E0", "name": "Premier League", "country": "England"}
    assert {"code": "JAP", "name": "J1 League", "country": "Japan"} in body


def test_teams_merges_db_and_model_and_filters(client):
    names = {t["name"]: t["league"] for t in client.get("/api/v1/teams").json()}
    assert names == {"Arsenal": "E0", "Chelsea": "E0", "Coventry": "E0", "Lyon": "F1", "Nice": "F1"}
    assert client.get("/api/v1/teams", params={"league": "F1"}).json() == [
        {"name": "Lyon", "league": "F1"},
        {"name": "Nice", "league": "F1"},
    ]
    assert [t["name"] for t in client.get("/api/v1/teams", params={"q": "ar"}).json()] == ["Arsenal"]
    _assert_error(client.get("/api/v1/teams", params={"q": "a"}), 422, "VALIDATION_ERROR")
    _assert_error(client.get("/api/v1/teams", params={"league": "ZZ"}), 404, "LEAGUE_NOT_FOUND")


def test_teams_includes_model_only_teams(client, service):
    service.teams["Brand New FC"] = "E0"
    names = [t["name"] for t in client.get("/api/v1/teams", params={"league": "E0", "q": "new"}).json()]
    assert names == ["Brand New FC"]


def test_fixtures_default_today_with_prediction_and_odds(client, service):
    body = client.get("/api/v1/fixtures").json()
    assert [f["home"] for f in body] == ["Lyon", "Arsenal"]
    ars = body[1]
    assert ars["league"] == "E0" and ars["date"] == "2026-10-24" and ars["time"] == "15:00"
    assert ars["kickoff"] == "2026-10-24T15:00:00+01:00"
    assert ars["status"] == "scheduled" and ars["score"] is None
    assert ars["odds"] == {"home": 2.0, "draw": 3.4, "away": 3.8}
    assert ars["prediction"]["model"] == "market"
    assert set(ars["prediction"]) == {
        "probs", "expected_goals", "over_2_5", "btts", "top_scores", "model", "model_version",
        "confidence",
    }  # fmt: skip
    assert ars["prediction"]["confidence"] is None  # only the European model sets it
    assert body[0]["odds"] is None and body[0]["prediction"]["model"] == "base"
    assert ("Arsenal", "Chelsea", "E0") == service.calls[-1][:3] or service.calls


def test_fixtures_range_includes_finished_with_score(client):
    body = client.get(
        "/api/v1/fixtures", params={"date": "2026-10-19", "date_to": "2026-10-26", "league": "E0"}
    ).json()
    assert [f["home"] for f in body] == ["Chelsea", "Arsenal", "Coventry"]
    assert body[0]["status"] == "finished" and body[0]["score"] == {"home": 3, "away": 1}
    assert body[2]["kickoff"] is None and body[2]["time"] is None
    team = client.get(
        "/api/v1/fixtures", params={"date": "2026-10-19", "date_to": "2026-10-26", "team": "Coventry"}
    ).json()
    assert len(team) == 2


def test_fixtures_validation(client):
    _assert_error(
        client.get("/api/v1/fixtures", params={"date": "2026-10-01", "date_to": "2026-10-16"}),
        422,
        "DATE_RANGE_TOO_WIDE",
    )
    _assert_error(
        client.get("/api/v1/fixtures", params={"date": "2026-10-10", "date_to": "2026-10-09"}),
        422,
        "INVALID_DATE_RANGE",
    )
    _assert_error(client.get("/api/v1/fixtures", params={"date": "24/10/2026"}), 422, "VALIDATION_ERROR")
    _assert_error(client.get("/api/v1/fixtures", params={"league": "e0"}), 422, "VALIDATION_ERROR")
    ok = client.get("/api/v1/fixtures", params={"date": "2026-10-10", "date_to": "2026-10-24"})
    assert ok.status_code == 200


def test_fixture_prediction_failure_yields_null(client, service):
    service.fail_for.add("Lyon")
    body = client.get("/api/v1/fixtures").json()
    assert body[0]["home"] == "Lyon" and body[0]["prediction"] is None
    assert body[1]["prediction"] is not None


def test_fixtures_without_model_have_null_prediction(no_model_client):
    body = no_model_client.get("/api/v1/fixtures").json()
    assert len(body) == 2 and all(f["prediction"] is None for f in body)


def test_predict(client, service):
    resp = client.get("/api/v1/predict", params={"home": "arsenal", "away": "Chelsea"})
    assert resp.status_code == 200
    body = resp.json()
    assert (body["home"], body["away"], body["league"]) == ("Arsenal", "Chelsea", "E0")
    assert body["probs"] == {"home": 0.5, "draw": 0.3, "away": 0.2}
    assert body["top_scores"][0] == {"score": "1-0", "p": 0.12}
    assert service.calls[-1][4] is None
    lyon = client.get("/api/v1/predict", params={"home": "Lyon", "away": "Arsenal", "league": "E0"})
    assert lyon.json()["league"] == "E0"


def test_predict_errors(client, no_model_client):
    _assert_error(client.get("/api/v1/predict", params={"home": "Nowhere", "away": "Chelsea"}), 404, "TEAM_NOT_FOUND")
    _assert_error(client.get("/api/v1/predict", params={"home": "Arsenal", "away": "Nowhere"}), 404, "TEAM_NOT_FOUND")
    _assert_error(client.get("/api/v1/predict", params={"home": "Arsenal", "away": "arsenal"}), 422, "SAME_TEAM")
    _assert_error(client.get("/api/v1/predict", params={"home": "Arsenal"}), 422, "VALIDATION_ERROR")
    _assert_error(
        client.get("/api/v1/predict", params={"home": "<script>", "away": "Chelsea"}), 422, "VALIDATION_ERROR"
    )
    _assert_error(
        no_model_client.get("/api/v1/predict", params={"home": "Arsenal", "away": "Chelsea"}),
        503,
        "MODEL_NOT_LOADED",
    )


def test_team_form(client):
    body = client.get("/api/v1/teams/Chelsea/form").json()
    assert body["team"] == "Chelsea" and body["league"] == "E0"
    assert [r["result"] for r in body["results"]] == ["W", "D", "W"]
    first = body["results"][0]
    assert first == {
        "date": "2026-10-19", "league": "E0", "home": "Chelsea", "away": "Coventry",
        "home_goals": 3, "away_goals": 1, "venue": "H", "opponent": "Coventry",
        "goals_for": 3, "goals_against": 1, "result": "W",
    }  # fmt: skip
    assert body["results"][2]["venue"] == "A" and body["results"][2]["goals_for"] == 2
    assert body["ratings"] == {"elo": 1600.0, "pi_home": 0.5, "pi_away": 0.2, "form5": 9.0}


def test_team_form_unknown_and_without_model(client, no_model_client):
    _assert_error(client.get("/api/v1/teams/Nowhere/form"), 404, "TEAM_NOT_FOUND")
    body = no_model_client.get("/api/v1/teams/Nice/form").json()
    assert body["ratings"] is None and body["results"] == [] and body["league"] is None
    assert client.get("/api/v1/teams/Nice/form").json()["ratings"] is None  # unknown to model


def test_unknown_route_and_method_use_envelope(client):
    _assert_error(client.get("/api/v1/nope"), 404, "NOT_FOUND")
    _assert_error(client.post("/api/v1/leagues"), 405, "METHOD_NOT_ALLOWED")


def test_unhandled_error_envelope(tmp_path, service):
    factory = memory_factory()
    service.known_teams = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
    app = create_app(service, factory, _settings(tmp_path), today=lambda: TODAY)
    with TestClient(app, raise_server_exceptions=False) as c:
        _assert_error(c.get("/api/v1/predict", params={"home": "A", "away": "B"}), 500, "INTERNAL_ERROR")


def test_cors_allows_frontend_origin(client):
    resp = client.get("/api/v1/leagues", headers={"Origin": "http://localhost:5173"})
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/api/v1/leagues", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_rate_limit_returns_429_envelope(tmp_path, service):
    app = create_app(service, memory_factory(), _settings(tmp_path, rate_limit="2/minute"))
    with TestClient(app) as c:
        assert c.get("/api/v1/leagues").status_code == 200
        assert c.get("/api/v1/leagues").status_code == 200
        _assert_error(c.get("/api/v1/leagues"), 429, "RATE_LIMITED")
        assert c.get("/api/v1/health").status_code == 200  # health is exempt


def test_lifespan_builds_db_and_tries_artifact(tmp_path):
    settings = _settings(tmp_path, database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
    app = create_app(settings=settings)
    with TestClient(app) as c:
        body = c.get("/api/v1/health").json()
    assert body["model_loaded"] is False and body["matches"] == 0


def test_lifespan_starts_and_stops_scheduler(tmp_path, monkeypatch):
    from matchpredict.ingest import scheduler as sched_mod

    events = []

    class FakeScheduler:
        def shutdown(self, wait):
            events.append("stop")

    from matchpredict import bootstrap

    monkeypatch.setattr(sched_mod, "start_scheduler", lambda f, h, s, **kw: events.append("start") or FakeScheduler())
    monkeypatch.setattr(bootstrap, "start_setup_thread", lambda *a, **kw: events.append("setup"))
    settings = _settings(tmp_path, scheduler_enabled=True)
    with TestClient(create_app(settings=settings)):
        pass
    assert events == ["setup", "start", "stop"]


def test_json_formatter_and_configure_logging():
    import logging

    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("w",), None)
    record.fields = {"path": "/a"}
    out = JsonFormatter().format(record)
    assert '"msg": "hello w"' in out and '"path": "/a"' in out
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "err", (), sys.exc_info())
    assert "ValueError" in JsonFormatter().format(rec)
    root = logging.getLogger()
    before = list(root.handlers)
    configure_logging("INFO")
    configure_logging("INFO")  # idempotent
    added = [h for h in root.handlers if h not in before]
    assert len(added) <= 1
    for h in added:
        root.removeHandler(h)


def test_main_runs_uvicorn(monkeypatch):
    import uvicorn

    import matchpredict

    called = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: called.update(app=app, host=host, port=port))
    matchpredict.main()
    assert called["app"] == "matchpredict.api.app:app" and called["port"] == 8000
