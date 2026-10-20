"""Serving wrapper: loads the trained artifact and implements contracts.PredictionService."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from matchpredict.contracts import ExpectedGoals, Prediction, Probs, ScoreProb
from matchpredict.predictor.data import sort_matches
from matchpredict.predictor.features import assemble, valid_odds
from matchpredict.predictor.goals import (
    GoalsModel,
    btts,
    expected_goals,
    over_line,
    rescale_to_1x2,
    top_scores,
)
from matchpredict.predictor.model import OutcomeModel
from matchpredict.predictor.ratings import RatingsEngine

DECIMALS = 4
TOP_SCORES = 5


def _round_probs(probs: np.ndarray) -> Probs:
    home, draw = round(float(probs[0]), DECIMALS), round(float(probs[1]), DECIMALS)
    return Probs(home=home, draw=draw, away=round(1.0 - home - draw, DECIMALS))


def _league_last_dates(engine: RatingsEngine) -> dict[str, pd.Timestamp]:
    """Latest applied match date per league, recovered from each team's last match."""
    out: dict[str, pd.Timestamp] = {}
    for team, league in engine.team_league.items():
        last = engine.last_date.get(team)
        if last is not None:
            stamp = pd.Timestamp(last)
            if league not in out or stamp > out[league]:
                out[league] = stamp
    return out


class ModelService:
    """Holds models + live ratings state. Not thread-safe for concurrent update_results."""

    def __init__(self, artifact: dict) -> None:
        self.artifact = artifact
        self.model_version: str = artifact["model_version"]
        self.last_result_date: pd.Timestamp | None = artifact.get("last_result_date")
        self.engine: RatingsEngine = artifact["engine"]
        self.base: OutcomeModel = artifact["base"]
        self.market: OutcomeModel = artifact["market"]
        self.goals: GoalsModel = artifact["goals"]
        self.league_last_date: dict[str, pd.Timestamp] = dict(
            artifact.get("league_last_date") or _league_last_dates(self.engine)
        )

    # --------------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: str | Path) -> ModelService:
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or "engine" not in artifact:
            raise ValueError(f"{path} is not a MatchPredict model artifact")
        return cls(artifact)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"last_result_date": self.last_result_date, "league_last_date": dict(self.league_last_date)}
        joblib.dump({**self.artifact, **state}, path, compress=3)

    # ---------------------------------------------------------------- predicting
    def _require_team(self, team: str) -> None:
        if team not in self.engine.played:
            raise KeyError(f"unknown team: {team!r}")

    def _require_predictable(self, team: str, league: str) -> None:
        """Unseen teams are seeded from their league (engine seeding); unknown team + league -> KeyError."""
        if team not in self.engine.played and league not in self.engine.league_teams:
            raise KeyError(f"unknown team {team!r} in unknown league {league!r}")

    def predict(
        self,
        home: str,
        away: str,
        league: str,
        date: pd.Timestamp,
        odds: tuple[float, float, float] | None = None,
    ) -> Prediction:
        self._require_predictable(home, league)
        self._require_predictable(away, league)
        day = pd.Timestamp(date).to_datetime64()
        ratings = pd.DataFrame([self.engine.features_for(home, away, league, day)])
        use_market = valid_odds(odds)
        odds_row = np.array([odds if use_market else (np.nan, np.nan, np.nan)], dtype=float)
        row = assemble(ratings, pd.Series([league]), odds_row)
        model = self.market if use_market else self.base
        probs = model.predict_proba(row)[0]
        matrix = rescale_to_1x2(self.goals.score_matrices(row)[0], probs)
        xg_home, xg_away = expected_goals(matrix)
        return Prediction(
            probs=_round_probs(probs),
            expected_goals=ExpectedGoals(home=round(xg_home, 2), away=round(xg_away, 2)),
            over_2_5=round(over_line(matrix, 2.5), DECIMALS),
            btts=round(btts(matrix), DECIMALS),
            top_scores=[ScoreProb(score=s, p=round(p, DECIMALS)) for s, p in top_scores(matrix, TOP_SCORES)],
            model="market" if use_market else "base",
            model_version=self.model_version,
        )

    # ------------------------------------------------------------------ updating
    def update_results(self, matches: pd.DataFrame) -> int:
        """Apply finished matches newer than their league's last applied date; return count applied.

        Dedupe is per league so leagues whose history lags (or is backfilled later) still catch up.
        New matches are applied in one chronological pass (same-day batch semantics).
        """
        finished = matches.dropna(subset=["home_goals", "away_goals"])
        if finished.empty:
            return 0
        finished = finished.assign(date=pd.to_datetime(finished["date"]).astype("datetime64[ns]"))
        cutoff = finished["league"].map(self.league_last_date)
        finished = finished[cutoff.isna() | (finished["date"] > cutoff)]
        if finished.empty:
            return 0
        if "time" not in finished.columns:
            finished = finished.assign(time=None)
        ordered = sort_matches(finished)
        self.engine.build_features(ordered)
        for league, last in ordered.groupby("league")["date"].max().items():
            self.league_last_date[league] = pd.Timestamp(last)
        newest = pd.Timestamp(ordered["date"].max())
        self.last_result_date = newest if self.last_result_date is None else max(self.last_result_date, newest)
        return len(ordered)

    # ------------------------------------------------------------------- queries
    def known_teams(self) -> dict[str, str]:
        return dict(self.engine.team_league)

    def team_ratings(self, team: str) -> dict[str, float]:
        self._require_team(team)
        eng = self.engine
        league = eng.team_league.get(team, "")
        pi_home, pi_away = eng.pi.get(team, (0.0, 0.0))
        return {
            "elo": float(eng._elo_of(team, league)),
            "pi_home": float(pi_home),
            "pi_away": float(pi_away),
            "form5": float(sum(points for points, _ in eng.form.get(team, ()))),
            "attack": float(eng._rating_of(eng.attack, team, league)),
            "defence": float(eng._rating_of(eng.defence, team, league)),
            "played": float(eng.played.get(team, 0)),
        }
