"""Seed the database from the history CSV.

Usage: ``uv run python -m matchpredict.db.seed [--csv PATH]``
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from matchpredict.config import get_settings
from matchpredict.db import repository
from matchpredict.db.session import SessionFactory, create_session_factory

log = logging.getLogger(__name__)
SOURCE = "history"


def seed(session_factory: SessionFactory, csv_path: Path) -> repository.UpsertStats:
    """Load ``csv_path``: bulk insert when the table is empty, else upsert."""
    from matchpredict.predictor.data import load_history  # shared history loader

    frame = load_history(csv_path)
    log.info("loaded %d history rows from %s", len(frame), csv_path)
    with session_factory() as session:
        if repository.count_matches(session) == 0:
            return repository.UpsertStats(
                inserted=repository.bulk_insert_matches(session, frame, SOURCE)
            )
        return repository.upsert_matches(session, frame, SOURCE)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Seed the MatchPredict DB from Matches.csv")
    parser.add_argument("--csv", type=Path, default=settings.history_csv)
    parser.add_argument("--database-url", default=settings.database_url)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not args.csv.exists():
        raise SystemExit(f"history CSV not found: {args.csv}")
    started = time.perf_counter()
    factory = create_session_factory(args.database_url)
    stats = seed(factory, args.csv)
    with factory() as session:
        total = repository.count_matches(session)
    log.info(
        "seed done in %.1fs: inserted=%d updated=%d total=%d",
        time.perf_counter() - started,
        stats.inserted,
        stats.updated,
        total,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
