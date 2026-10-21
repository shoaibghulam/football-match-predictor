"""Single-container mode: the API also serves the built web app (MP_STATIC_DIR)."""

import pytest
from fastapi.testclient import TestClient
from mp_testkit import FakeService, memory_factory

from matchpredict.api.app import create_app
from matchpredict.config import Settings

INDEX = "<!doctype html><title>MatchPredict</title>"


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


@pytest.fixture
def web(tmp_path):
    root = tmp_path / "web"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(INDEX, encoding="utf-8")
    (root / "assets" / "app.js").write_text("console.info(1)", encoding="utf-8")
    (root / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("do not serve", encoding="utf-8")
    return root


def _client(tmp_path, **kw) -> TestClient:
    app = create_app(FakeService(), memory_factory(), _settings(tmp_path, **kw))
    return TestClient(app)


def test_serves_index_assets_and_root_files(tmp_path, web):
    with _client(tmp_path, static_dir=web) as c:
        assert c.get("/").text == INDEX
        assert c.get("/assets/app.js").text == "console.info(1)"
        assert c.get("/favicon.svg").text == "<svg/>"


def test_client_side_routes_fall_back_to_index(tmp_path, web):
    with _client(tmp_path, static_dir=web) as c:
        for path in ("/results", "/team/Arsenal", "/predict?home=A"):
            response = c.get(path)
            assert response.status_code == 200
            assert response.text == INDEX


def test_api_routes_still_win_and_unknown_api_paths_are_404(tmp_path, web):
    with _client(tmp_path, static_dir=web) as c:
        assert c.get("/api/v1/health").json()["status"] in {"ok", "degraded"}
        missing = c.get("/api/v1/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_never_serves_files_outside_the_static_dir(tmp_path, web):
    with _client(tmp_path, static_dir=web) as c:
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt"):
            assert "do not serve" not in c.get(path).text


def test_without_static_dir_root_is_not_served(tmp_path):
    with _client(tmp_path) as c:
        assert c.get("/").status_code == 404
