"""Export per-match walk-forward (out-of-sample) v2 predictions into ``backtest_predictions``.

CLI (from ``backend/``)::

    uv run python -m matchpredict.predictor.backtest_export                 # 2022/23 .. 2026/27
    uv run python -m matchpredict.predictor.backtest_export --seasons 2025  # one season only

For each test season N it reuses ``v2.train.run_fold``: members are trained on matches before
1 Jul of season N-1, early-stopped / pooled / calibrated on season N-1, and season N is scored —
so every stored probability comes from a model that never saw that season. The served
probability is the v2 market ensemble when the match has valid pre-match odds (domestic), else
the v2 base ensemble (no odds / UEFA), exactly as ``metrics_v2.json`` reports ``v2_served``.
Each season is written in its own short transaction (replacing that season's previous rows).
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import delete, insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from matchpredict.config import get_settings
from matchpredict.db.results_models import BacktestPrediction
from matchpredict.db.session import SessionFactory, create_session_factory
from matchpredict.predictor.data import sort_matches
from matchpredict.predictor.v2 import train as v2_train
from matchpredict.predictor.v2.data import (
    FRAME_COLUMNS,
    build_canonicaliser,
    canonicalise,
    dedupe_european,
    read_db,
    standardise,
)
from matchpredict.predictor.v2.global_rating import ClubEloBook
from matchpredict.predictor.v2.names import Canonicaliser

log = logging.getLogger("matchpredict.backtest_export")

DEFAULT_SEASONS: tuple[int, ...] = (*v2_train.TEST_SEASONS, 2026)  # 2026/27 = current season to date
WRITE_ATTEMPTS = 5
WRITE_RETRY_SECONDS = 3.0
CHUNK = 2000


def season_label(season: int) -> str:
    return f"{season}/{(season + 1) % 100:02d}"


def model_version(season: int) -> str:
    return f"v2-backtest-{season_label(season)}"


def load_frame_with_ids(database_url: str) -> tuple[pd.DataFrame, np.ndarray, Canonicaliser, ClubEloBook | None]:
    """Same frame as ``v2.data.load_matches`` plus the aligned ``matches.id`` of every row."""
    settings = get_settings()
    raw = read_db(database_url)
    names = build_canonicaliser(standardise(raw), settings.openfootball_dir, v2_train.ELO_CSV)
    frame = dedupe_european(canonicalise(standardise(raw), names))
    frame = frame[frame["home"] != frame["away"]]
    frame = sort_matches(frame[[*FRAME_COLUMNS, "id"]])
    book = ClubEloBook.from_csv(v2_train.ELO_CSV) if v2_train.ELO_CSV.exists() else None
    return frame[list(FRAME_COLUMNS)], frame["id"].to_numpy(dtype=np.int64), names, book


def fold_rows(fold: v2_train.FoldResult, match_ids: np.ndarray) -> list[dict]:
    """Rows for ``backtest_predictions`` from one walk-forward fold."""
    market = fold.probs["v2_market"]
    served = fold.probs["v2_served"]
    used_market = np.isfinite(market).all(axis=1)
    version = model_version(fold.season)
    return [
        {
            "match_id": int(match_ids[i]),
            "season": fold.season,
            "model": "market" if used_market[k] else "base",
            "model_version": version,
            "prob_home": float(served[k, 0]),
            "prob_draw": float(served[k, 1]),
            "prob_away": float(served[k, 2]),
        }
        for k, i in enumerate(fold.idx)
    ]


def _replace(session: Session, season: int, rows: Sequence[dict]) -> None:
    ids = [r["match_id"] for r in rows]
    session.execute(delete(BacktestPrediction).where(BacktestPrediction.season == season))
    for start in range(0, len(ids), CHUNK):  # a match id may have moved season after a data fix
        session.execute(delete(BacktestPrediction).where(BacktestPrediction.match_id.in_(ids[start:start + CHUNK])))
    for start in range(0, len(rows), CHUNK):
        session.execute(insert(BacktestPrediction), list(rows[start:start + CHUNK]))
    session.commit()


def write_season(factory: SessionFactory, season: int, rows: Sequence[dict]) -> int:
    """Replace one season's rows in a single short transaction (retried while the DB is locked)."""
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            with factory() as session:
                _replace(session, season, rows)
            return len(rows)
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == WRITE_ATTEMPTS:
                raise
            log.warning("database locked, retrying season %s (%d/%d)", season, attempt, WRITE_ATTEMPTS)
            time.sleep(WRITE_RETRY_SECONDS * attempt)
    return 0  # pragma: no cover - loop always returns or raises


def export(database_url: str, seasons: Sequence[int], config: v2_train.TrainConfig | None = None) -> dict[int, int]:
    config = config or v2_train.TrainConfig()
    started = time.perf_counter()
    frame, match_ids, names, book = load_frame_with_ids(database_url)
    ds = v2_train.prepare(frame, names, config, book)
    factory = create_session_factory(database_url)
    written: dict[int, int] = {}
    for season in seasons:
        if not (ds.season == season).any():
            log.warning("no finished matches in season %s, skipped", season_label(season))
            continue
        fold = v2_train.run_fold(ds, season, config)
        written[season] = write_season(factory, season, fold_rows(fold, match_ids))
        log.info("season %s: stored %d backtest predictions (%.0fs elapsed)",
                 season_label(season), written[season], time.perf_counter() - started)  # fmt: skip
    return written


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Store walk-forward out-of-sample v2 predictions")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS),
                        help="test season start years, e.g. 2024 for 2024/25")  # fmt: skip
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # most recent season first, so an interrupted run still leaves the freshest record
    written = export(args.database_url or get_settings().database_url, sorted(set(args.seasons), reverse=True))
    log.info("done: %s", {season_label(s): n for s, n in written.items()})


if __name__ == "__main__":  # pragma: no cover
    main()
