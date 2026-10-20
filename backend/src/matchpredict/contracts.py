"""Shared contracts between predictor (ML), ingest/db and the API.

Every component exchanges matches as a pandas DataFrame with MATCH_COLUMNS
(missing values = NaN). Dates are datetime64 (day precision); time is "HH:MM" or None.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import pandas as pd

MATCH_COLUMNS: tuple[str, ...] = (
    "date",  # datetime64[ns], match day
    "time",  # str "HH:MM" or None
    "league",  # division code, e.g. "E0", "SP1", "USA"
    "home",  # football-data.co.uk team name
    "away",
    "home_goals",  # float, NaN for upcoming fixtures
    "away_goals",
    "home_shots", "away_shots",
    "home_target", "away_target",  # shots on target
    "home_corners", "away_corners",
    "home_xg", "away_xg",
    "odds_home", "odds_draw", "odds_away",  # pre-match decimal odds (Bet365, else market average)
)

DEFAULT_ARTIFACT = "artifacts/model.joblib"  # relative to backend/


class Probs(TypedDict):
    home: float
    draw: float
    away: float


class ExpectedGoals(TypedDict):
    home: float
    away: float


class ScoreProb(TypedDict):
    score: str  # "2-1" (home-away)
    p: float


class Prediction(TypedDict):
    probs: Probs
    expected_goals: ExpectedGoals
    over_2_5: float
    btts: float
    top_scores: list[ScoreProb]  # 5 most likely, descending
    model: str  # "market" when odds were used, else "base"
    model_version: str


class PredictionService(Protocol):
    model_version: str
    last_result_date: pd.Timestamp | None  # latest finished match applied to ratings

    def predict(
        self,
        home: str,
        away: str,
        league: str,
        date: pd.Timestamp,
        odds: tuple[float, float, float] | None = None,
    ) -> Prediction: ...

    def update_results(self, matches: pd.DataFrame) -> int:
        """Apply finished matches (MATCH_COLUMNS) newer than last_result_date; return count applied."""
        ...

    def known_teams(self) -> dict[str, str]:
        """team name -> league code of its most recent match."""
        ...

    def team_ratings(self, team: str) -> dict[str, float]:
        """Current elo, pi_home, pi_away, form5 for a team (KeyError if unknown)."""
        ...
