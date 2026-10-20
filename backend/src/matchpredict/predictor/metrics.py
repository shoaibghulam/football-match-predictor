"""Probabilistic scoring rules for 3-way (home/draw/away) forecasts.

Outcome encoding everywhere: 0 = home win, 1 = draw, 2 = away win.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean ranked probability score (lower is better, 0 = perfect)."""
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[outcomes]
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(onehot, axis=1)
    return float((cum_diff[:, :-1] ** 2).sum(axis=1).mean() / (n_classes - 1))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    picked = probs[np.arange(len(outcomes)), outcomes]
    return float(-np.log(np.clip(picked, _EPS, 1.0)).mean())


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[outcomes]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == outcomes).mean())


def evaluate(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return {
        "n": len(outcomes),
        "rps": rps(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "brier": brier(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
    }


def devig(odds: np.ndarray) -> np.ndarray:
    """Decimal odds (n, 3) -> implied probabilities with the margin removed proportionally.

    Rows with any missing/invalid odd come back as all-NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / np.where(odds > 1.0, odds, np.nan)
        return inverse / inverse.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> list[float]:
    """Expected calibration error per class (one-vs-rest, equal-width probability bins)."""
    errors = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for k in range(probs.shape[1]):
        p = probs[:, k]
        hit = (outcomes == k).astype(float)
        bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        total = 0.0
        for b in range(n_bins):
            mask = bins == b
            if mask.any():
                total += mask.sum() * abs(p[mask].mean() - hit[mask].mean())
        errors.append(float(total / len(p)))
    return errors
