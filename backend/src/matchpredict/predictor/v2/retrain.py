"""Weekly v2 retrain (roadmap Phase 5) with a no-regression promotion check.

1. Train a candidate artifact in a subprocess (``v2.train --skip-eval``; keeps the current
   artifact's walk-forward gate decision) so the API process is not blocked.
2. Promote it only if its validation log loss (last 365 days, domestic and UEFA) is not worse
   than the current artifact's by more than ``settings.retrain_tolerance``.
3. On promotion: atomically replace ``model_v2.joblib``, drop the old state sidecar and
   reload the holders. Every run is appended to ``artifacts/retrain_log.jsonl``.

Disable with ``MP_RETRAIN_ENABLED=false``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib

from matchpredict.config import BACKEND_DIR, Settings
from matchpredict.loader import ServiceHolder, clear_v2_cache

log = logging.getLogger(__name__)

RETRAIN_TIMEOUT_S = 3 * 3600
CANDIDATE_NAME = "model_v2.candidate.joblib"
LOG_NAME = "retrain_log.jsonl"
SCOPES = (("domestic_log_loss", "domestic"), ("europe_log_loss", "europe"))


@dataclass
class RetrainResult:
    status: str  # skipped | failed | rejected | promoted
    reason: str = ""
    current: dict = field(default_factory=dict)
    candidate: dict = field(default_factory=dict)
    started_at: str = ""
    seconds: float = 0.0


def compare(current: dict, candidate: dict, tolerance: float) -> tuple[bool, str]:
    """Promote when no scope's validation log loss is worse than current + tolerance."""
    for key, label in SCOPES:
        old, new = current.get(key), candidate.get(key)
        if new is None or (isinstance(new, float) and math.isnan(new)):
            if old is not None and not (isinstance(old, float) and math.isnan(old)):
                return False, f"candidate has no {label} validation"
            continue
        if old is None or (isinstance(old, float) and math.isnan(old)):
            continue
        if new > old + tolerance:
            return False, f"{label} validation log loss {new:.4f} > {old:.4f} + {tolerance}"
    return True, "validation log loss not worse"


def _validation(path: Path) -> dict:
    return dict(joblib.load(path).get("validation") or {})


def _train_command(candidate: Path, current: Path) -> list[str]:
    return [sys.executable, "-m", "matchpredict.predictor.v2.train", "--skip-eval",
            "--out", str(candidate), "--gate-from", str(current)]  # fmt: skip


def _append_log(path: Path, result: RetrainResult) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(result), default=str) + "\n")


def weekly_retrain(
    settings: Settings,
    holders: Sequence[ServiceHolder] = (),
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> RetrainResult:
    """One retrain attempt; never raises. ``run`` is injectable for tests."""
    started = now()
    current = Path(settings.v2_artifact_path)
    candidate = current.with_name(CANDIDATE_NAME)
    result = RetrainResult(status="skipped", started_at=started.isoformat())
    try:
        if not settings.retrain_enabled:
            result.reason = "disabled (MP_RETRAIN_ENABLED=false)"
            return result
        if not current.exists():
            result.reason = "no v2 artifact yet (run predictor.v2.train once)"
            return result
        proc = run(_train_command(candidate, current), cwd=str(BACKEND_DIR), capture_output=True,
                   text=True, timeout=RETRAIN_TIMEOUT_S)  # fmt: skip
        if proc.returncode != 0 or not candidate.exists():
            result.status, result.reason = "failed", (proc.stderr or "")[-2000:]
            return result
        result.current, result.candidate = _validation(current), _validation(candidate)
        ok, reason = compare(result.current, result.candidate, settings.retrain_tolerance)
        result.reason = reason
        if not ok:
            result.status = "rejected"
            candidate.unlink(missing_ok=True)
            return result
        os.replace(candidate, current)
        current.with_name(current.name + ".state").unlink(missing_ok=True)
        clear_v2_cache()
        for holder in holders:
            holder.reload()
        result.status = "promoted"
        return result
    except Exception as exc:
        log.exception("weekly retrain failed")
        result.status, result.reason = "failed", f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result.seconds = round((now() - started).total_seconds(), 1)
        log.info("weekly retrain: %s (%s)", result.status, result.reason)
        if result.status != "skipped" or settings.retrain_enabled:
            try:
                _append_log(current.with_name(LOG_NAME), result)
            except OSError:
                log.exception("could not write the retrain log")
