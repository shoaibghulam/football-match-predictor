"""CatBoost 1X2 classifiers with temperature-scaling calibration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize_scalar

TEMPERATURE_BOUNDS = (0.05, 20.0)
SMALL_BATCH = 256
REFIT_ITERATION_FACTOR = 1.1  # more data -> slightly more trees when refitting on everything


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TemperatureScaler:
    """Single scalar T: p = softmax(logits / T), T chosen to minimise log loss."""

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, outcomes: np.ndarray) -> TemperatureScaler:
        rows = np.arange(len(outcomes))

        def nll(t: float) -> float:
            z = logits / t
            z = z - z.max(axis=1, keepdims=True)
            log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return float(-log_p[rows, outcomes].mean())

        self.temperature = float(minimize_scalar(nll, bounds=TEMPERATURE_BOUNDS, method="bounded").x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits / self.temperature)


@dataclass
class OutcomeModel:
    """CatBoost MultiClass on pre-match features; early-stopped and calibrated on a held-out season.

    With ``baseline_cols`` (probability columns, e.g. de-vigged odds) the trees learn a residual on
    top of log(baseline): logits = log(baseline) + trees(features). Rows without a valid baseline are
    dropped from training and cannot be predicted (callers fall back to a model without one).
    """

    feature_cols: list[str]
    cat_cols: list[str] = field(default_factory=list)
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 5.0
    early_stopping_rounds: int = 100
    thread_count: int = -1
    baseline_cols: list[str] = field(default_factory=list)
    temperature: float = 1.0
    best_iteration: int = 0
    model: CatBoostClassifier | None = None

    def _log_baseline(self, X: pd.DataFrame) -> np.ndarray | None:
        if not self.baseline_cols:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(X[self.baseline_cols].to_numpy(dtype=float))

    def _usable(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        base = self._log_baseline(X)
        if base is None:
            return X, np.asarray(y)
        keep = np.isfinite(base).all(axis=1)
        return X[keep], np.asarray(y)[keep]

    def _pool(self, X: pd.DataFrame, y: np.ndarray | None = None, with_baseline: bool = True) -> Pool:
        data = X[self.feature_cols].copy()
        for col in self.cat_cols:
            data[col] = data[col].astype(str)
        baseline = self._log_baseline(X) if with_baseline else None
        return Pool(data, label=y, cat_features=self.cat_cols, baseline=baseline)

    def _classifier(self, iterations: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            thread_count=self.thread_count,
            random_seed=0,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> OutcomeModel:
        """Train with early stopping on (X_val, y_val), then fit the temperature on the same season."""
        X, y = self._usable(X, y)
        X_val, y_val = self._usable(X_val, y_val)
        self.model = self._classifier(self.iterations)
        self.model.fit(
            self._pool(X, y),
            eval_set=self._pool(X_val, y_val),
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
        )
        self.best_iteration = int(self.model.get_best_iteration() or 0) + 1
        self.temperature = TemperatureScaler().fit(self.logits(X_val), y_val).temperature
        return self

    def refit_full(self, X: pd.DataFrame, y: np.ndarray) -> OutcomeModel:
        """New model trained on all rows with the tuned tree count, keeping the fitted temperature."""
        X, y = self._usable(X, y)
        iterations = max(1, round(self.best_iteration * REFIT_ITERATION_FACTOR))
        refit = replace(self, iterations=iterations, model=None)
        refit.model = refit._classifier(iterations)
        refit.model.fit(self._pool(X, y))
        return refit

    def logits(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("OutcomeModel is not fitted")
        threads = 1 if len(X) < SMALL_BATCH else self.thread_count  # no thread-pool overhead when serving
        raw = self.model.predict(
            self._pool(X, with_baseline=False), prediction_type="RawFormulaVal", thread_count=threads
        )
        raw = np.asarray(raw, dtype=float)
        base = self._log_baseline(X)
        return raw if base is None else raw + base

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return softmax(self.logits(X) / self.temperature)
