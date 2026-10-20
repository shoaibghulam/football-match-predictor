"""Log-linear pooling of 1X2 forecasts (weights + class bias fit by log loss) and helpers.

    log p_final ∝ Σ_i w_i · log p_i + b        (w_i ≥ 0, b = (b_home, b_draw, 0))

With a single member this is temperature scaling (T = 1/w) plus a draw/home bias, so the
pool doubles as the calibration step. Weights are fitted on a held-out (calibration)
season, never on test rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from matchpredict.predictor.metrics import log_loss

_EPS = 1e-9
BIAS_L2 = 1.0  # mild shrinkage of the class bias (per row-normalised objective)
MAX_WEIGHT = 5.0


def _logp(members: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.log(np.clip(p, _EPS, 1.0)) for p in members], axis=0)  # (k, n, 3)


def _combine(logs: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = np.tensordot(weights, logs, axes=1) + bias[None, :]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class LogPool:
    names: list[str]
    weights: list[float] = field(default_factory=list)
    bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    use_bias: bool = False  # bias fitted on one season overfits it (see docs/08-model-v2.md)

    def fit(self, members: list[np.ndarray], y: np.ndarray) -> LogPool:
        logs = _logp(members)
        k, n = logs.shape[0], logs.shape[1]
        rows = np.arange(n)
        y = np.asarray(y)

        def objective(theta: np.ndarray) -> float:
            w = theta[:k]
            b = np.array([theta[k], theta[k + 1], 0.0]) if self.use_bias else np.zeros(3)
            p = _combine(logs, w, b)
            nll = -np.log(np.clip(p[rows, y], _EPS, 1.0)).mean()
            return float(nll + (BIAS_L2 * (b**2).sum() / n if self.use_bias else 0.0))

        start = np.concatenate([np.full(k, 1.0 / k), np.zeros(2)])
        bounds = [(0.0, MAX_WEIGHT)] * k + [(-1.0, 1.0)] * 2
        res = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        self.weights = [float(v) for v in res.x[:k]]
        self.bias = [float(res.x[k]), float(res.x[k + 1]), 0.0] if self.use_bias else [0.0, 0.0, 0.0]
        return self

    def predict(self, members: list[np.ndarray]) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("LogPool is not fitted")
        return _combine(_logp(members), np.asarray(self.weights), np.asarray(self.bias))

    def describe(self) -> dict:
        return {"members": dict(zip(self.names, [round(w, 4) for w in self.weights], strict=True)),
                "bias": [round(b, 4) for b in self.bias]}  # fmt: skip


def fit_pool(names: list[str], members: list[np.ndarray], y: np.ndarray, use_bias: bool = False) -> LogPool:
    return LogPool(names=list(names), use_bias=use_bias).fit(members, y)


def pool_gain(pool: LogPool, members: list[np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Log loss of the pool and of each member (diagnostic)."""
    out = {name: log_loss(p, y) for name, p in zip(pool.names, members, strict=True)}
    out["pool"] = log_loss(pool.predict(members), y)
    return out


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, list[dict]]:
    """Per class: equal-width bins of predicted probability -> (n, mean predicted, observed)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: dict[str, list[dict]] = {}
    for k, label in enumerate(("home", "draw", "away")):
        p = probs[:, k]
        hit = (np.asarray(y) == k).astype(float)
        bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            mask = bins == b
            if mask.sum() == 0:
                continue
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "predicted": round(float(p[mask].mean()), 4), "observed": round(float(hit[mask].mean()), 4)})  # fmt: skip
        out[label] = rows
    return out
