"""Expected-goals model: Poisson GBMs for each side + Dixon-Coles low-score correction.

Score matrices are indexed [home_goals, away_goals] over 0..MAX_GOALS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

MAX_GOALS = 10
RHO_BOUNDS = (-0.25, 0.25)
_GOALS = np.arange(MAX_GOALS + 1)
_HOME_IDX, _AWAY_IDX = np.meshgrid(_GOALS, _GOALS, indexing="ij")
_LAMBDA_CLIP = (0.05, 6.0)


def dc_tau(hg: np.ndarray, ag: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles adjustment factor for each (home goals, away goals) observation."""
    tau = np.ones(len(hg), dtype=float)
    tau = np.where((hg == 0) & (ag == 0), 1.0 - lh * la * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1.0 + lh * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1.0 + la * rho, tau)
    return np.where((hg == 1) & (ag == 1), 1.0 - rho, tau)


def score_matrix(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """(n, G, G) Dixon-Coles score probabilities, renormalised over the truncated grid."""
    lh = np.asarray(lh, dtype=float)
    la = np.asarray(la, dtype=float)
    ph = poisson.pmf(_GOALS[None, :], lh[:, None])
    pa = poisson.pmf(_GOALS[None, :], la[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1.0 - lh * la * rho
    m[:, 0, 1] *= 1.0 + lh * rho
    m[:, 1, 0] *= 1.0 + la * rho
    m[:, 1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)


def fit_rho(hg: np.ndarray, ag: np.ndarray, lh: np.ndarray, la: np.ndarray) -> float:
    """MLE of rho given fitted Poisson means (the Poisson terms do not depend on rho)."""
    hg, ag = np.asarray(hg), np.asarray(ag)
    low = (hg <= 1) & (ag <= 1)
    hg, ag, lh, la = hg[low], ag[low], np.asarray(lh)[low], np.asarray(la)[low]

    def nll(rho: float) -> float:
        tau = dc_tau(hg, ag, lh, la, rho)
        return float(-np.log(np.clip(tau, 1e-9, None)).sum())

    return float(minimize_scalar(nll, bounds=RHO_BOUNDS, method="bounded").x)


def outcome_probs(matrix: np.ndarray) -> np.ndarray:
    """[P(home), P(draw), P(away)] from one (G, G) matrix or a stack (n, G, G)."""
    home = np.where(_HOME_IDX > _AWAY_IDX, matrix, 0.0).sum(axis=(-2, -1))
    draw = np.where(_HOME_IDX == _AWAY_IDX, matrix, 0.0).sum(axis=(-2, -1))
    away = np.where(_HOME_IDX < _AWAY_IDX, matrix, 0.0).sum(axis=(-2, -1))
    return np.stack([home, draw, away], axis=-1)


def over_line(matrix: np.ndarray, line: float = 2.5) -> float:
    return float(matrix[(_HOME_IDX + _AWAY_IDX) > line].sum())


def btts(matrix: np.ndarray) -> float:
    return float(matrix[(_HOME_IDX > 0) & (_AWAY_IDX > 0)].sum())


def expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    return float((matrix * _HOME_IDX).sum()), float((matrix * _AWAY_IDX).sum())


def top_scores(matrix: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    flat = matrix.ravel()
    order = np.argsort(-flat, kind="stable")[:k]
    return [(f"{i // (MAX_GOALS + 1)}-{i % (MAX_GOALS + 1)}", float(flat[i])) for i in order]


def rescale_to_1x2(matrix: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Scale the H/D/A regions of a (G, G) matrix so they sum to the given 1X2 vector."""
    current = outcome_probs(matrix)
    factors = np.divide(probs, current, out=np.zeros(3), where=current > 0)
    scale = np.select(
        [_HOME_IDX > _AWAY_IDX, _HOME_IDX == _AWAY_IDX],
        [factors[0], factors[1]],
        default=factors[2],
    )
    out = matrix * scale
    return out / out.sum()


@dataclass
class GoalsModel:
    """Two Poisson-loss gradient boosted regressors (home goals, away goals) + DC rho."""

    feature_cols: list[str]
    cat_cols: list[str] = field(default_factory=list)
    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    rho: float = 0.0
    categories: dict[str, list[str]] = field(default_factory=dict)
    used_cols: list[str] = field(default_factory=list)  # numeric columns with any data at fit time
    home_model: HistGradientBoostingRegressor | None = None
    away_model: HistGradientBoostingRegressor | None = None

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Plain float array (fast single-row predict): numeric columns + integer category codes."""
        parts = [X[self.used_cols].to_numpy(dtype=float)]
        for col in self.cat_cols:
            lookup = {name: float(i) for i, name in enumerate(self.categories[col])}
            codes = [lookup.get(str(v), np.nan) for v in X[col]]  # unseen -> NaN (missing)
            parts.append(np.asarray(codes, dtype=float).reshape(-1, 1))
        return np.hstack(parts)

    def _regressor(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=100,
            l2_regularization=1.0,
            categorical_features=[False] * len(self.used_cols) + [True] * len(self.cat_cols),
            early_stopping=False,
            random_state=0,
        )

    def fit(self, X: pd.DataFrame, home_goals: np.ndarray, away_goals: np.ndarray) -> GoalsModel:
        self.categories = {c: sorted(pd.unique(X[c].astype(str))) for c in self.cat_cols}
        self.used_cols = [c for c in self.feature_cols if X[c].notna().any()]
        features = self._matrix(X)
        self.home_model = self._regressor().fit(features, home_goals)
        self.away_model = self._regressor().fit(features, away_goals)
        lh, la = self.predict_lambdas(X)
        self.rho = fit_rho(np.asarray(home_goals), np.asarray(away_goals), lh, la)
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.home_model is None or self.away_model is None:
            raise RuntimeError("GoalsModel is not fitted")
        features = self._matrix(X)
        lh = np.clip(self.home_model.predict(features), *_LAMBDA_CLIP)
        la = np.clip(self.away_model.predict(features), *_LAMBDA_CLIP)
        return lh, la

    def score_matrices(self, X: pd.DataFrame) -> np.ndarray:
        lh, la = self.predict_lambdas(X)
        return score_matrix(lh, la, self.rho)

    def predict_1x2(self, X: pd.DataFrame) -> np.ndarray:
        return outcome_probs(self.score_matrices(X))
