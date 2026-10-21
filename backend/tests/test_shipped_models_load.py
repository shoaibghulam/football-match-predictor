"""The models shipped in backend/models load and predict without data/raw (no EloRatings.csv)."""

import pandas as pd
import pytest

from matchpredict import loader
from matchpredict.config import BACKEND_DIR, Settings

MODELS = BACKEND_DIR / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "model.joblib").exists(), reason="no shipped models")


@pytest.fixture
def no_raw_data(monkeypatch):
    """Fail loudly if anything tries to read data/raw (EloRatings.csv / Matches.csv)."""
    real = pd.read_csv

    def guarded(path, *args, **kwargs):
        if "EloRatings" in str(path) or "Matches.csv" in str(path):
            raise AssertionError(f"serving must not read {path}")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    loader.clear_v2_cache()
    yield
    loader.clear_v2_cache()


def _settings(tmp_path) -> Settings:
    return Settings(
        artifact_path=MODELS / "model.joblib",
        europe_artifact_path=MODELS / "europe.joblib",
        v2_artifact_path=MODELS / "model_v2.joblib",
        history_csv=tmp_path / "missing" / "Matches.csv",
        database_url="sqlite:///:memory:",
    )


def test_shipped_models_predict_without_raw_data(tmp_path, no_raw_data):
    settings = _settings(tmp_path)
    domestic = loader.domestic_holder(settings).ensure_loaded()
    europe = loader.europe_holder(settings).ensure_loaded()
    assert domestic is not None and europe is not None
    teams = domestic.known_teams()
    assert teams.get("Arsenal") == "E0" and len(teams) > 500
    day = pd.Timestamp.now().normalize()
    pred = domestic.predict("Arsenal", "Chelsea", "E0", day)
    assert abs(sum(pred["probs"].values()) - 1) < 1e-6
    uefa = europe.predict("Arsenal", "Barcelona", "UCL", day)
    assert abs(sum(uefa["probs"].values()) - 1) < 1e-6


def test_v1_services_load_without_raw_data(tmp_path, no_raw_data):
    settings = _settings(tmp_path)
    assert loader.load_service(settings.artifact_path) is not None
    assert loader.load_europe_service(settings.europe_artifact_path) is not None
