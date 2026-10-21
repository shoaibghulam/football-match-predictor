"""Install the pre-trained models shipped in ``backend/models/`` into the artifacts directory.

The image carries trained artifacts so a fresh clone needs no training. At start-up every
shipped file that is *missing* from the artifacts location is copied in; existing files are
never overwritten (the daily ingest and the weekly retrain keep updating that copy). A state
sidecar (``model_v2.joblib.state``) is only installed together with its model, so a
shipped sidecar can never be paired with a different, user-trained model.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from matchpredict.config import Settings

log = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".state"
NOT_ARTIFACTS = {".gz"}  # e.g. the backtest bundle: imported into the DB, not an artifact


def targets(settings: Settings) -> dict[str, Path]:
    """Shipped file name -> where the API expects it (others go next to ``artifact_path``)."""
    v2 = Path(settings.v2_artifact_path)
    return {
        "model.joblib": Path(settings.artifact_path),
        "europe.joblib": Path(settings.europe_artifact_path),
        "model_v2.joblib": v2,
        "model_v2.joblib" + SIDECAR_SUFFIX: v2.with_name(v2.name + SIDECAR_SUFFIX),
    }


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".installing")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def install_shipped_models(source_dir: Path, settings: Settings) -> list[Path]:
    """Copy missing shipped artifacts into place; return the files installed. Never raises."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        log.info("no shipped models at %s", source_dir)
        return []
    where = targets(settings)
    default_dir = Path(settings.artifact_path).parent
    shipped = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix not in NOT_ARTIFACTS
    )  # fmt: skip
    installed: list[Path] = []
    installed_names: set[str] = set()
    # models first, so a sidecar can check whether its model was installed in this run
    for src in sorted(shipped, key=lambda p: p.name.endswith(SIDECAR_SUFFIX)):
        dst = where.get(src.name, default_dir / src.name)
        if dst.exists():
            continue
        if src.name.endswith(SIDECAR_SUFFIX) and src.name.removesuffix(SIDECAR_SUFFIX) not in installed_names:
            continue  # its model is a different (existing) file
        try:
            _copy(src, dst)
        except OSError:
            log.exception("could not install shipped model %s -> %s", src, dst)
            continue
        installed.append(dst)
        installed_names.add(src.name)
    if installed:
        log.info("installed shipped models: %s", ", ".join(p.name for p in installed))
    return installed
