"""Feature table: ratings-engine features + league categorical + de-vigged market odds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchpredict.predictor.goals import outcome_probs, score_matrix
from matchpredict.predictor.metrics import devig
from matchpredict.predictor.ratings import EWMA_KEYS, EngineConfig, RatingsEngine

ODDS_COLUMNS = ("odds_home", "odds_draw", "odds_away")
CAT_FEATURES: list[str] = ["league"]

_TEAM_KEYS = ["form5", "gd5", "rest", "played"] + [
    f"{key}_{direction}" for key in EWMA_KEYS for direction in ("for", "against")
]
RATING_FEATURES: list[str] = [
    "elo_home", "elo_away", "elo_diff", "elo_exp_home",
    "pi_home", "pi_away", "pi_diff", "pi_pred_gd",
    "league_home_rate", "league_draw_rate",
    "elo_fast_diff", "gr_lambda_home", "gr_lambda_away",
    "gr_att_home", "gr_def_home", "gr_att_away", "gr_def_away",
] + [f"{key}_{side}" for side in ("home", "away") for key in _TEAM_KEYS]
DERIVED_FEATURES: list[str] = ["gr_p_home", "gr_p_draw", "gr_p_away"]

BASE_FEATURES: list[str] = RATING_FEATURES + DERIVED_FEATURES + CAT_FEATURES
MARKET_ONLY: list[str] = ["mkt_home", "mkt_draw", "mkt_away", "mkt_logit_ha"]
MARKET_FEATURES: list[str] = BASE_FEATURES + MARKET_ONLY


def market_features(odds: np.ndarray) -> pd.DataFrame:
    """De-vigged implied probabilities (+ home-vs-away log ratio); NaN when odds are invalid."""
    probs = devig(np.asarray(odds, dtype=float).reshape(-1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        logit = np.log(probs[:, 0] / probs[:, 2])
    return pd.DataFrame(
        {"mkt_home": probs[:, 0], "mkt_draw": probs[:, 1], "mkt_away": probs[:, 2], "mkt_logit_ha": logit}
    )


def valid_odds(odds: tuple[float, float, float] | None) -> bool:
    if odds is None or len(odds) != 3:
        return False
    try:
        values = [float(o) for o in odds]
    except (TypeError, ValueError):
        return False
    return all(np.isfinite(v) and v > 1.0 for v in values)


_CHUNK = 50_000


def poisson_1x2(lam_home: np.ndarray, lam_away: np.ndarray) -> np.ndarray:
    """Independent-Poisson H/D/A probabilities, computed in chunks to bound memory."""
    parts = [
        outcome_probs(score_matrix(lam_home[i : i + _CHUNK], lam_away[i : i + _CHUNK], rho=0.0))
        for i in range(0, len(lam_home), _CHUNK)
    ]
    return np.vstack(parts) if parts else np.empty((0, 3))


def assemble(ratings: pd.DataFrame, league: pd.Series, odds: np.ndarray) -> pd.DataFrame:
    table = ratings.reindex(columns=RATING_FEATURES).astype(float)
    probs = poisson_1x2(table["gr_lambda_home"].to_numpy(), table["gr_lambda_away"].to_numpy())
    for i, col in enumerate(DERIVED_FEATURES):
        table[col] = probs[:, i]
    table["league"] = league.astype(str).to_numpy()
    market = market_features(odds)
    market.index = table.index
    return pd.concat([table, market], axis=1)


def build_feature_table(
    matches: pd.DataFrame, engine: RatingsEngine | None = None, config: EngineConfig | None = None
) -> tuple[pd.DataFrame, RatingsEngine]:
    """Run the ratings engine over `matches` (chronological pass) and return all model features."""
    engine = engine or RatingsEngine(config)
    ratings = engine.build_features(matches)
    odds = matches.reindex(columns=list(ODDS_COLUMNS)).to_numpy(dtype=float)
    return assemble(ratings, matches["league"], odds), engine
