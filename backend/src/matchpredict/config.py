"""Runtime settings (environment variables prefixed with ``MP_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'app.db').as_posix()}"
    artifact_path: Path = BACKEND_DIR / "artifacts" / "model.joblib"
    europe_artifact_path: Path = BACKEND_DIR / "artifacts" / "europe.joblib"
    # Model v2 (predictor/v2): used per scope (domestic / europe) only when its artifact passed the gate.
    v2_artifact_path: Path = BACKEND_DIR / "artifacts" / "model_v2.joblib"
    v2_enabled: bool = True
    retrain_enabled: bool = True  # weekly v2 retrain (Sunday 03:00 UTC by default)
    retrain_day_of_week: str = "sun"
    retrain_hour_utc: int = Field(default=3, ge=0, le=23)
    retrain_tolerance: float = Field(default=0.003, ge=0.0, le=0.1)  # allowed validation log-loss increase
    snapshot_refresh_minutes: int = Field(default=30, ge=1, le=1440)  # re-price pre-kickoff snapshots
    snapshot_days_ahead: int = Field(default=7, ge=0, le=14)
    openfootball_dir: Path = BACKEND_DIR / "data" / "openfootball"
    thesportsdb_key: str = "123"  # TheSportsDB free key (no-key UEFA fallback)
    europe_ingest_enabled: bool = True
    live_refresh_minutes: int = Field(default=2, ge=1, le=60)
    live_window_hours: float = Field(default=3.0, gt=0, le=24)
    history_csv: Path = REPO_DIR / "data" / "raw" / "Matches.csv"
    # Zero-setup start (Docker): install the shipped models, seed history (Matches.csv when
    # present, else football-data.co.uk downloads) and UEFA history in the background.
    bootstrap_enabled: bool = False
    shipped_models_dir: Path = BACKEND_DIR / "models"
    history_cache_dir: Path = BACKEND_DIR / "data" / "history"
    history_workers: int = Field(default=3, ge=1, le=4)  # parallel downloads (1 = strictly sequential)
    setup_status_path: Path = BACKEND_DIR / "data" / "setup_status.json"
    backtest_bundle_path: Path = BACKEND_DIR / "models" / "backtest_predictions.csv.gz"
    cors_origins: str = "http://localhost:5173"  # comma-separated
    scheduler_enabled: bool = True
    ingest_hour_utc: int = Field(default=7, ge=0, le=23)
    football_data_org_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MP_FOOTBALL_DATA_ORG_KEY", "FOOTBALL_DATA_ORG_KEY"),
    )
    rate_limit: str = "60/minute"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    results_lookback_days: int = 365  # how far back new-league result files are ingested

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
