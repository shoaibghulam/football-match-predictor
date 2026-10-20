"""v2 model members: weighted CatBoost 1X2 (optionally residual on odds) and a global-Elo logit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from matchpredict.predictor.model import softmax

SMALL_BATCH = 256
REFIT_ITERATION_FACTOR = 1.1


@dataclass
class CatModel:
    """CatBoost MultiClass, early-stopped on a held-out season; optional log-odds baseline."""

    feature_cols: list[str]
    cat_cols: list[str] = field(default_factory=list)
    baseline_cols: list[str] = field(default_factory=list)
    iterations: int = 3000
    learning_rate: float = 0.06
    depth: int = 6
    l2_leaf_reg: float = 5.0
    early_stopping_rounds: int = 100
    thread_count: int = -1
    best_iteration: int = 0
    model: CatBoostClassifier | None = None

    def _baseline(self, X: pd.DataFrame) -> np.ndarray | None:
        if not self.baseline_cols:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(X[self.baseline_cols].to_numpy(dtype=float))

    def usable(self, X: pd.DataFrame) -> np.ndarray:
        base = self._baseline(X)
        return np.ones(len(X), bool) if base is None else np.isfinite(base).all(axis=1)

    def _pool(self, X: pd.DataFrame, y=None, weight=None, with_baseline: bool = True) -> Pool:
        data = X[self.feature_cols].copy()
        for col in self.cat_cols:
            data[col] = data[col].astype(str)
        baseline = self._baseline(X) if with_baseline else None
        return Pool(data, label=y, cat_features=self.cat_cols, baseline=baseline, weight=weight)

    def _classifier(self, iterations: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            loss_function="MultiClass", iterations=iterations, learning_rate=self.learning_rate,
            depth=self.depth, l2_leaf_reg=self.l2_leaf_reg, thread_count=self.thread_count,
            random_seed=0, verbose=False, allow_writing_files=False,
        )  # fmt: skip

    def fit(self, X, y, X_val, y_val, weight=None, weight_val=None) -> CatModel:
        keep, keep_val = self.usable(X), self.usable(X_val)
        w = None if weight is None else np.asarray(weight)[keep]
        wv = None if weight_val is None else np.asarray(weight_val)[keep_val]
        self.model = self._classifier(self.iterations)
        self.model.fit(
            self._pool(X[keep], np.asarray(y)[keep], w),
            eval_set=self._pool(X_val[keep_val], np.asarray(y_val)[keep_val], wv),
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
        )
        self.best_iteration = int(self.model.get_best_iteration() or 0) + 1
        return self

    def refit_full(self, X, y, weight=None) -> CatModel:
        keep = self.usable(X)
        iterations = max(1, round(self.best_iteration * REFIT_ITERATION_FACTOR))
        refit = replace(self, iterations=iterations, model=None)
        refit.model = refit._classifier(iterations)
        w = None if weight is None else np.asarray(weight)[keep]
        refit.model.fit(refit._pool(X[keep], np.asarray(y)[keep], w))
        return refit

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CatModel is not fitted")
        threads = 1 if len(X) < SMALL_BATCH else self.thread_count
        raw = np.asarray(
            self.model.predict(self._pool(X, with_baseline=False), prediction_type="RawFormulaVal", thread_count=threads),
            dtype=float,
        )
        base = self._baseline(X)
        return softmax(raw if base is None else raw + base)


ELO_LR_FEATURES = ("logit", "abs_logit", "is_euro", "logit_x_euro", "neutral", "leg2", "agg_gd", "prov_home", "prov_away")


def elo_design(X: pd.DataFrame) -> pd.DataFrame:
    exp = X["g_exp_home"].clip(1e-4, 1 - 1e-4).to_numpy()
    logit = np.log(exp / (1 - exp))
    euro = X["is_euro"].to_numpy(dtype=float)
    return pd.DataFrame(
        {"logit": logit, "abs_logit": np.abs(logit), "is_euro": euro, "logit_x_euro": logit * euro,
         "neutral": X["neutral"].to_numpy(dtype=float), "leg2": X["leg2"].to_numpy(dtype=float),
         "agg_gd": X["agg_gd"].to_numpy(dtype=float),
         "prov_home": (X["g_seed_home"].to_numpy(dtype=float) > 0).astype(float),
         "prov_away": (X["g_seed_away"].to_numpy(dtype=float) > 0).astype(float)},
        index=X.index,
    )  # fmt: skip


@dataclass
class EloLR:
    """Multinomial logit on the global-Elo expectation (+ UEFA context). Ensemble member/baseline."""

    c: float = 1.0
    scaler: StandardScaler | None = None
    model: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, weight=None) -> EloLR:
        x = elo_design(X).to_numpy()
        self.scaler = StandardScaler().fit(x)
        self.model = LogisticRegression(C=self.c, max_iter=2000).fit(self.scaler.transform(x), y, sample_weight=weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("EloLR is not fitted")
        p = self.model.predict_proba(self.scaler.transform(elo_design(X).to_numpy()))
        return p[:, np.argsort(self.model.classes_)]
