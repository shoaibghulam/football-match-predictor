"""Accuracy / scoring-rule aggregates for the Results page (pure functions over metric rows)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np

from matchpredict.predictor.metrics import brier, devig, log_loss, rps

OUTCOMES = ("H", "D", "A")
# calibration buckets on the probability of our favourite outcome (always >= 1/3)
BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0001),
)  # fmt: skip
DAILY_POINTS = 60  # last N match days in the daily chart


def outcome_codes(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win (metrics.py encoding)."""
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))


def row_rps(probs: Sequence[float], outcome: int) -> float:
    return rps(np.asarray([probs], dtype=float), np.asarray([outcome]))


def _acc(correct: int, n: int) -> float | None:
    return correct / n if n else None


def _bucket_label(lo: float, hi: float) -> str:
    return f"{round(lo * 100)}-{min(round(hi * 100), 100)}%"


def _group(keys: np.ndarray, hit: np.ndarray, extra: dict[str, np.ndarray] | None = None) -> list[dict]:
    out = []
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        n, correct = int(mask.sum()), int(hit[mask].sum())
        row: dict[str, Any] = {"key": key, "n": n, "correct": correct, "accuracy": _acc(correct, n)}
        for name, values in (extra or {}).items():
            row[name] = float(values[mask].mean())
        out.append(row)
    return out


def _by_outcome(codes: np.ndarray, hit: np.ndarray) -> dict[str, dict]:
    out = {}
    for k, label in enumerate(OUTCOMES):
        mask = codes == k
        n, correct = int(mask.sum()), int(hit[mask].sum())
        out[label] = {"n": n, "correct": correct, "accuracy": _acc(correct, n)}
    return out


def _calibration(top: np.ndarray, hit: np.ndarray) -> list[dict]:
    out = []
    for lo, hi in BUCKETS:
        mask = (top >= lo) & (top < hi)
        n = int(mask.sum())
        out.append({
            "bucket": _bucket_label(lo, hi), "lo": lo, "hi": min(hi, 1.0), "n": n,
            "accuracy": _acc(int(hit[mask].sum()), n),
            "avg_prob": float(top[mask].mean()) if n else None,
        })  # fmt: skip
    return out


def _bookmaker(probs: np.ndarray, y: np.ndarray, hit: np.ndarray, odds: np.ndarray) -> dict | None:
    book = devig(odds)
    ok = np.isfinite(book).all(axis=1)
    if not ok.any():
        return None
    b, yy = book[ok], y[ok]
    return {
        "n": int(ok.sum()),
        "accuracy": float((b.argmax(axis=1) == yy).mean()),
        "rps": rps(b, yy),
        "model_accuracy": float(hit[ok].mean()),  # our model on the same matches
        "model_rps": rps(probs[ok], yy),
    }


def _by_league(leagues: np.ndarray, hit: np.ndarray, row_scores: np.ndarray) -> list[dict]:
    rows = _group(leagues, hit, {"rps": row_scores})
    rows.sort(key=lambda r: (-r["n"], r["key"]))
    return [{"league": r.pop("key"), **r} for r in rows]


def _timeline(dates: Sequence[dt.date], hit: np.ndarray) -> tuple[list[dict], list[dict]]:
    days = np.array([d.isoformat() for d in dates])
    daily = [{"date": r.pop("key"), **r} for r in _group(days, hit)][-DAILY_POINTS:]
    months = np.array([d[:7] for d in days])
    monthly = [{"month": r.pop("key"), **r} for r in _group(months, hit)]
    return daily, monthly


def empty_summary() -> dict:
    return {
        "n": 0, "correct": 0, "accuracy": None, "rps": None, "log_loss": None, "brier": None,
        "by_outcome": {k: {"n": 0, "correct": 0, "accuracy": None} for k in OUTCOMES},
        "by_predicted": {k: {"n": 0, "correct": 0, "accuracy": None} for k in OUTCOMES},
        "by_league": [], "by_confidence_bucket": _calibration(np.array([]), np.array([], dtype=bool)),
        "bookmaker": None, "daily": [], "monthly": [], "first_date": None, "last_date": None,
    }  # fmt: skip


def summarise(rows: Sequence[Any]) -> dict:
    """Aggregate metric rows (prob_*, home/away goals, odds_*, league, date)."""
    if not rows:
        return empty_summary()
    probs = np.array([[r.prob_home, r.prob_draw, r.prob_away] for r in rows], dtype=float)
    y = outcome_codes(np.array([r.home_goals for r in rows]), np.array([r.away_goals for r in rows]))
    odds = np.array([[r.odds_home, r.odds_draw, r.odds_away] for r in rows], dtype=float)
    predicted = probs.argmax(axis=1)
    hit = predicted == y
    onehot = np.eye(3)[y]
    cum = np.cumsum(probs, axis=1) - np.cumsum(onehot, axis=1)
    row_scores = (cum[:, :-1] ** 2).sum(axis=1) / 2
    dates = [r.date for r in rows]
    daily, monthly = _timeline(dates, hit)
    return {
        "n": len(rows),
        "correct": int(hit.sum()),
        "accuracy": float(hit.mean()),
        "rps": float(row_scores.mean()),
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "by_outcome": _by_outcome(y, hit),
        "by_predicted": _by_outcome(predicted, hit),
        "by_league": _by_league(np.array([r.league for r in rows]), hit, row_scores),
        "by_confidence_bucket": _calibration(probs.max(axis=1), hit),
        "bookmaker": _bookmaker(probs, y, hit, np.nan_to_num(odds, nan=0.0)),
        "daily": daily,
        "monthly": monthly,
        "first_date": min(dates).isoformat(),
        "last_date": max(dates).isoformat(),
    }
