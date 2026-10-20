"""One feature table for domestic and European matches.

Blocks (all strictly pre-match, day-batch semantics):

* global rating (``global_rating.GlobalRating``): cross-country Elo, country offsets,
  league strength, attack/defence λ and their Poisson 1X2;
* v1 domestic engine (``predictor.ratings``): per-league Elo, pi-ratings, attack/defence,
  form, rest, EWMAs of goals/shots/target/corners/xG. European results update only the
  team history part (form, rest, goal EWMAs); league-scale v1 features are NaN for UEFA rows;
* ClubElo as of the last snapshot strictly before the match day (optional block);
* context: competition / stage categoricals, neutral venue, second leg + first-leg goal diff;
* de-vigged odds (market variant only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matchpredict.ingest.leagues import EUROPEAN_LEAGUES
from matchpredict.predictor.features import (
    DERIVED_FEATURES as V1_DERIVED,
)
from matchpredict.predictor.features import (
    MARKET_ONLY,
    RATING_FEATURES,
    market_features,
    poisson_1x2,
)
from matchpredict.predictor.ratings import MatchResult, RatingsEngine
from matchpredict.predictor.v2.global_rating import ClubEloBook, GlobalRating

LEG_WINDOW_DAYS = 35
ODDS_COLUMNS = ["odds_home", "odds_draw", "odds_away"]

# v1 features that only make sense inside one domestic league
LEAGUE_SCALE = [
    "elo_home", "elo_away", "elo_diff", "elo_exp_home", "pi_home", "pi_away", "pi_diff", "pi_pred_gd",
    "league_home_rate", "league_draw_rate", "elo_fast_diff", "gr_lambda_home", "gr_lambda_away",
    "gr_att_home", "gr_def_home", "gr_att_away", "gr_def_away",
]  # fmt: skip
V1_FEATURES: list[str] = RATING_FEATURES + V1_DERIVED

GLOBAL_FEATURES: list[str] = [
    "g_elo_home", "g_elo_away", "g_elo_diff", "g_hfa", "g_exp_home", "g_off_home", "g_off_away",
    "g_lstr_home", "g_lstr_away", "g_lstr_diff", "g_games_home", "g_games_away",
    "g_seed_home", "g_seed_away",
]  # fmt: skip
GOAL_FEATURES: list[str] = [
    "g_lam_home", "g_lam_away", "g_att_home", "g_def_home", "g_att_away", "g_def_away",
    "g_p_home", "g_p_draw", "g_p_away",
]  # fmt: skip
CLUBELO_FEATURES: list[str] = ["ce_home", "ce_away", "ce_diff"]
CONTEXT_FEATURES: list[str] = ["is_euro", "neutral", "leg2", "agg_gd"]
CAT_FEATURES: list[str] = ["league", "stage"]
MARKET_BASELINE = ["mkt_home", "mkt_draw", "mkt_away"]


@dataclass(frozen=True)
class FeatureSet:
    """Which blocks a model uses (for ablations)."""

    v1: bool = True
    global_elo: bool = True
    global_goals: bool = True
    clubelo: bool = True
    context: bool = True

    def columns(self, market: bool = False) -> list[str]:
        cols: list[str] = []
        if self.v1:
            cols += V1_FEATURES
        if self.global_elo:
            cols += GLOBAL_FEATURES
        if self.global_goals:
            cols += GOAL_FEATURES
        if self.clubelo:
            cols += CLUBELO_FEATURES
        if self.context:
            cols += CONTEXT_FEATURES
        cols += CAT_FEATURES if self.context else ["league"]
        if market:
            cols += MARKET_ONLY
        return cols

    def cat_columns(self) -> list[str]:
        return list(CAT_FEATURES) if self.context else ["league"]


class DomesticEngine(RatingsEngine):
    """v1 engine; UEFA results only update form / rest / goal EWMAs (not league-scale ratings)."""

    def update(self, result: MatchResult) -> None:
        if result.league in EUROPEAN_LEAGUES:
            self._update_team_history(result)
            return
        super().update(result)

    def features_for(self, home: str, away: str, league: str, date) -> dict[str, float]:
        if league not in EUROPEAN_LEAGUES:
            return super().features_for(home, away, league, date)
        feats = super().features_for(home, away, self.team_league.get(home, league), date)
        for key in LEAGUE_SCALE:
            feats[key] = math.nan
        return feats


@dataclass
class LegBook:
    """Recent UEFA results to detect second legs (same pair, reversed venue, ≤35 days)."""

    recent: list[tuple[np.datetime64, str, str, str, float, float]] = field(default_factory=list)

    def context(self, home: str, away: str, league: str, stage: str, day) -> tuple[float, float]:
        if league not in EUROPEAN_LEAGUES or stage not in ("qual", "ko", "final"):
            return 0.0, 0.0
        day = np.datetime64(day, "D")
        for d, lg, h, a, hg, ag in reversed(self.recent):
            gap = (day - d) / np.timedelta64(1, "D")
            if gap > LEG_WINDOW_DAYS:
                break
            if gap > 0 and lg == league and h == away and a == home:
                return 1.0, float(np.clip(ag - hg, -3, 3))
        return 0.0, 0.0

    def add(self, day, league: str, home: str, away: str, hg: float, ag: float) -> None:
        if league in EUROPEAN_LEAGUES:
            self.recent.append((np.datetime64(day, "D"), league, home, away, hg, ag))
            horizon = np.datetime64(day, "D") - np.timedelta64(2 * LEG_WINDOW_DAYS, "D")
            while self.recent and self.recent[0][0] < horizon:
                self.recent.pop(0)


@dataclass
class FeatureState:
    """Everything needed to compute features for new fixtures (persisted in the artifact)."""

    rating: GlobalRating
    domestic: DomesticEngine
    legs: LegBook = field(default_factory=LegBook)
    clubelo: ClubEloBook | None = None


def _leg_features(matches: pd.DataFrame, legs: LegBook) -> pd.DataFrame:
    out = np.zeros((len(matches), 2))
    order = np.argsort(matches["date"].to_numpy(dtype="datetime64[D]"), kind="stable")
    cols = [matches[c].to_numpy() for c in ("date", "league", "home", "away", "home_goals", "away_goals", "stage")]
    pending: list[tuple] = []
    current = None
    for i in order:
        day = np.datetime64(cols[0][i], "D")
        if day != current:
            for item in pending:
                legs.add(*item)
            pending, current = [], day
        out[i] = legs.context(cols[2][i], cols[3][i], cols[1][i], str(cols[6][i]), day)
        if not (math.isnan(cols[4][i]) or math.isnan(cols[5][i])):
            pending.append((day, cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i]))
    for item in pending:
        legs.add(*item)
    return pd.DataFrame(out, columns=["leg2", "agg_gd"], index=matches.index)


def _clubelo_features(matches: pd.DataFrame, book: ClubEloBook | None) -> pd.DataFrame:
    n = len(matches)
    if book is None:
        return pd.DataFrame(np.full((n, 3), np.nan), columns=CLUBELO_FEATURES, index=matches.index)
    days = matches["date"].to_numpy(dtype="datetime64[D]")
    home = np.array([book.get(t, d) for t, d in zip(matches["home"], days, strict=True)], dtype=float)
    away = np.array([book.get(t, d) for t, d in zip(matches["away"], days, strict=True)], dtype=float)
    return pd.DataFrame({"ce_home": home, "ce_away": away, "ce_diff": home - away}, index=matches.index)


def assemble(
    matches: pd.DataFrame,
    global_feats: pd.DataFrame,
    v1_feats: pd.DataFrame,
    legs: pd.DataFrame,
    clubelo: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the blocks into one numeric table (+ categoricals + market columns)."""
    v1 = v1_feats.reindex(columns=RATING_FEATURES).astype(float)
    lam = v1[["gr_lambda_home", "gr_lambda_away"]].to_numpy()
    probs = np.full((len(v1), 3), np.nan)
    ok = np.isfinite(lam).all(axis=1)
    if ok.any():
        probs[ok] = poisson_1x2(lam[ok, 0], lam[ok, 1])
    for j, col in enumerate(V1_DERIVED):
        v1[col] = probs[:, j]
    g = global_feats.astype(float).copy()
    gp = poisson_1x2(g["g_lam_home"].to_numpy(), g["g_lam_away"].to_numpy())
    for j, col in enumerate(("g_p_home", "g_p_draw", "g_p_away")):
        g[col] = gp[:, j]
    table = pd.concat([v1, g, clubelo, legs], axis=1)
    table["is_euro"] = matches["league"].isin(EUROPEAN_LEAGUES).astype(float).to_numpy()
    table["neutral"] = matches["neutral"].astype(float).to_numpy() if "neutral" in matches else 0.0
    table["league"] = matches["league"].astype(str).to_numpy()
    table["stage"] = matches["stage"].astype(str).to_numpy() if "stage" in matches else "dom"
    odds = matches.reindex(columns=ODDS_COLUMNS).to_numpy(dtype=float, copy=True)
    euro = table["is_euro"].to_numpy() > 0
    odds[euro] = np.nan  # no reliable UEFA odds in our data
    market = market_features(odds)
    market.index = table.index
    return pd.concat([table, market], axis=1)


def build_table(matches: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    """Run every engine over ``matches`` (chronological, updating ``state``) -> feature table."""
    global_feats = state.rating.build_features(matches)
    v1_feats = state.domestic.build_features(matches)
    legs = _leg_features(matches, state.legs)
    clubelo = _clubelo_features(matches, state.clubelo)
    return assemble(matches, global_feats, v1_feats, legs, clubelo)


def fixture_row(
    state: FeatureState,
    home: str,
    away: str,
    league: str,
    day: pd.Timestamp,
    stage: str,
    neutral: bool,
    odds: tuple[float, float, float] | None,
) -> pd.DataFrame:
    """Feature row for one upcoming fixture (no state change)."""
    d = np.datetime64(pd.Timestamp(day).date(), "D")
    match = pd.DataFrame(
        [{"date": pd.Timestamp(day), "league": league, "home": home, "away": away, "stage": stage,
          "neutral": float(neutral), "home_goals": np.nan, "away_goals": np.nan,
          "odds_home": odds[0] if odds else np.nan, "odds_draw": odds[1] if odds else np.nan,
          "odds_away": odds[2] if odds else np.nan}]
    )  # fmt: skip
    g = pd.DataFrame([state.rating.features_for(home, away, league, d, stage, neutral)])
    v1 = pd.DataFrame([state.domestic.features_for(home, away, league, d)])
    leg2, agg = state.legs.context(home, away, league, stage, d)
    legs = pd.DataFrame([{"leg2": leg2, "agg_gd": agg}])
    clubelo = _clubelo_features(match, state.clubelo)
    return assemble(match, g, v1, legs, clubelo)
