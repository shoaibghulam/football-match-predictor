"""Loads the prediction service lazily and shares it safely between threads.

The real implementation is ``matchpredict.predictor.service.ModelService``; it is
imported inside :func:`load_service` so the API and ingest work (degraded) when
the artifact or the module is not available yet.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from matchpredict.contracts import Prediction, PredictionService

log = logging.getLogger(__name__)

PREDICTION_CACHE_SIZE = 4096
OddsTriple = tuple[float, float, float] | None


def load_service(path: Path) -> PredictionService | None:
    """Return a loaded ModelService, or None when the artifact/module is missing or broken."""
    if not path.exists():
        log.warning("model artifact not found at %s; predictions disabled", path)
        return None
    try:
        from matchpredict.predictor.service import ModelService

        service = ModelService.load(path)
    except Exception:
        log.exception("failed to load model artifact %s", path)
        return None
    log.info("loaded model %s from %s", getattr(service, "model_version", "?"), path)
    return service


def load_europe_service(path: Path) -> PredictionService | None:
    """Return the European (UEFA / cross-league) service, or None when unavailable."""
    if not path.exists():
        log.warning("europe artifact not found at %s; UEFA predictions disabled", path)
        return None
    try:
        from matchpredict.predictor.europe import EuropeService

        service = EuropeService.load(path)
    except Exception:
        log.exception("failed to load europe artifact %s", path)
        return None
    log.info("loaded europe model %s from %s", service.model_version, path)
    return service


# --- model v2 (gate-aware) ---------------------------------------------------------

_V2_LOCK = threading.Lock()
_V2_CACHE: dict[Path, PredictionService] = {}
V2_SCOPES = ("domestic", "europe")


def load_v2_service(path: Path) -> PredictionService | None:
    """The v2 service (one shared instance per artifact path), or None when missing/broken."""
    path = Path(path)
    with _V2_LOCK:
        if path in _V2_CACHE:
            return _V2_CACHE[path]
        if not path.exists():
            log.info("v2 artifact not found at %s; using v1 models", path)
            return None
        try:
            from matchpredict.predictor.v2.service import V2Service

            service = V2Service.load(path)
        except Exception:
            log.exception("failed to load v2 artifact %s", path)
            return None
        _V2_CACHE[path] = service
        log.info("loaded v2 model %s (gate %s)", service.model_version, getattr(service, "gate", {}))
        return service


def clear_v2_cache() -> None:
    with _V2_LOCK:
        _V2_CACHE.clear()


def gated_loader(
    scope: str, v2_path: Path, fallback: Callable[[Path], PredictionService | None]
) -> Callable[[Path], PredictionService | None]:
    """Loader that serves v2 for ``scope`` when its artifact passed the promotion gate, else v1."""
    if scope not in V2_SCOPES:
        raise ValueError(f"unknown scope {scope!r}")

    def load(path: Path) -> PredictionService | None:
        v2 = load_v2_service(v2_path)
        if v2 is not None and bool(getattr(v2, "gate", {}).get(scope)):
            log.info("%s predictions: v2 %s", scope, v2.model_version)
            return v2
        if v2 is not None:
            log.info("%s predictions: v1 (v2 %s did not pass the %s gate)", scope, v2.model_version, scope)
        return fallback(path)

    return load


def domestic_holder(settings) -> ServiceHolder:
    loader = gated_loader("domestic", settings.v2_artifact_path, load_service) if settings.v2_enabled else load_service
    return ServiceHolder(artifact_path=settings.artifact_path, loader=loader)


def europe_holder(settings) -> ServiceHolder:
    loader = (
        gated_loader("europe", settings.v2_artifact_path, load_europe_service)
        if settings.v2_enabled
        else load_europe_service
    )
    return ServiceHolder(artifact_path=settings.europe_artifact_path, loader=loader)


def artifact_path_of(service: PredictionService, default: Path) -> Path:
    """Where a service persists its state (v2 keeps its own path; v1 uses the configured one)."""
    own = getattr(service, "artifact_path", None)
    return Path(own) if own else Path(default)


class ServiceHolder:
    """Holds the (optional) service, a lock for mutation, and a prediction cache."""

    def __init__(
        self,
        service: PredictionService | None = None,
        artifact_path: Path | None = None,
        loader: Callable[[Path], PredictionService | None] | None = None,
    ) -> None:
        self._service = service
        self.artifact_path = artifact_path
        self._loader = loader
        self.lock = threading.RLock()
        self._cache: OrderedDict[tuple, Prediction] = OrderedDict()

    @property
    def service(self) -> PredictionService | None:
        return self._service

    @property
    def loaded(self) -> bool:
        return self._service is not None

    def ensure_loaded(self) -> PredictionService | None:
        """Try loading from ``artifact_path`` if nothing is loaded yet."""
        with self.lock:
            if self._service is None and self.artifact_path is not None:
                self._service = (self._loader or load_service)(self.artifact_path)
                self._cache.clear()
            return self._service

    def invalidate(self) -> None:
        with self.lock:
            self._cache.clear()

    def reload(self) -> PredictionService | None:
        """Drop the current service and load again from ``artifact_path`` (after a retrain)."""
        with self.lock:
            self._service = None
            self._cache.clear()
        return self.ensure_loaded()

    def predict(
        self,
        home: str,
        away: str,
        league: str,
        day: pd.Timestamp,
        odds: OddsTriple = None,
        stage: str | None = None,
    ) -> Prediction:
        service = self._service
        if service is None:
            raise RuntimeError("model not loaded")
        key = (home, away, league, day, odds, stage, service.model_version)
        with self.lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
            if stage is not None:
                prediction = service.predict(home, away, league, day, odds, stage=stage)  # type: ignore[call-arg]
            else:
                prediction = service.predict(home, away, league, day, odds)
            self._cache[key] = prediction
            if len(self._cache) > PREDICTION_CACHE_SIZE:
                self._cache.popitem(last=False)
            return prediction
