"""Shared helpers for ingest/db/api tests (not collected: no ``test_`` prefix)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matchpredict.contracts import MATCH_COLUMNS, Prediction
from matchpredict.db.session import create_session_factory

FIXTURES = Path(__file__).parent / "fixtures"


def memory_factory():
    return create_session_factory("sqlite:///:memory:")


def frame(rows: list[dict]) -> pd.DataFrame:
    """rows with any subset of MATCH_COLUMNS; the rest are NaN/None."""
    out = []
    for row in rows:
        full = {c: np.nan for c in MATCH_COLUMNS}
        full["time"] = None
        full.update(row)
        full["date"] = pd.Timestamp(full["date"])
        out.append(full)
    df = pd.DataFrame(out, columns=list(MATCH_COLUMNS))
    df["date"] = df["date"].astype("datetime64[ns]")
    return df


class FakeService:
    """In-memory PredictionService implementing the contracts Protocol."""

    def __init__(self, teams: dict[str, str] | None = None) -> None:
        self.model_version = "test-1"
        self.last_result_date: pd.Timestamp | None = pd.Timestamp("2026-10-01")
        self.teams = teams or {"Arsenal": "E0", "Chelsea": "E0", "Coventry": "E0", "Lyon": "F1"}
        self.calls: list[tuple] = []
        self.applied: list[pd.DataFrame] = []
        self.saved: list[Path] = []
        self.fail_for: set[str] = set()

    def predict(self, home, away, league, date, odds=None) -> Prediction:
        self.calls.append((home, away, league, date, odds))
        if home in self.fail_for:
            raise KeyError(home)
        return {
            "probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "expected_goals": {"home": 1.6, "away": 1.0},
            "over_2_5": 0.5,
            "btts": 0.45,
            "top_scores": [{"score": "1-0", "p": 0.12}, {"score": "1-1", "p": 0.11}],
            "model": "market" if odds else "base",
            "model_version": self.model_version,
        }

    def update_results(self, matches: pd.DataFrame) -> int:
        new = matches
        if self.last_result_date is not None:
            new = matches[matches["date"] > self.last_result_date]
        self.applied.append(new)
        if len(new):
            self.last_result_date = new["date"].max()
        return len(new)

    def known_teams(self) -> dict[str, str]:
        return dict(self.teams)

    def team_ratings(self, team: str) -> dict[str, float]:
        if team not in self.teams:
            raise KeyError(team)
        return {"elo": 1600.0, "pi_home": 0.5, "pi_away": 0.2, "form5": 9.0}

    def save(self, path: Path) -> None:
        self.saved.append(path)
