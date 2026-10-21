"""Ship the walk-forward backtest (``backtest_predictions``) with the code.

The backtest takes ~30 min to compute, so its rows are exported to a compact file
(``backend/models/backtest_predictions.csv.gz``) and imported on a fresh database. Rows are
keyed by ``(league, date, home, away)`` — never by ``matches.id``, which differs between
databases — and joined to the local ``matches`` table on import:

1. exact key;
2. same league / date and one team equal (a club spelled differently, e.g. "Nottm Forest"
   in the Kaggle history vs "Nott'm Forest" on football-data.co.uk), when unique;
3. the same with the day and month swapped (some Kaggle rows of the extra leagues) or the
   date ±1 day (late kick-offs published on the next UK day).

Rows without a match are skipped (count logged). Import is idempotent: it is skipped when the
table already holds rows for every model version in the file.

CLI: ``uv run python -m matchpredict.db.backtest_bundle export|import [--path P]``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, insert, select

from matchpredict.config import BACKEND_DIR, get_settings
from matchpredict.db.models import Base, Match
from matchpredict.db.results_models import BacktestPrediction
from matchpredict.db.session import SessionFactory, create_session_factory

log = logging.getLogger(__name__)

DEFAULT_PATH = BACKEND_DIR / "models" / "backtest_predictions.csv.gz"
KEY = ["league", "date", "home", "away"]
COLUMNS = [*KEY, "season", "model", "model_version", "prob_home", "prob_draw", "prob_away"]
PROBS = ["prob_home", "prob_draw", "prob_away"]
CHUNK = 5000


@dataclass(frozen=True)
class ImportResult:
    status: str  # skipped | imported | empty
    rows: int = 0
    joined: int = 0
    unmatched: int = 0


def export_bundle(session_factory: SessionFactory, path: Path = DEFAULT_PATH) -> int:
    """Write the backtest joined to its match keys; returns the number of rows."""
    cols = [Match.league, Match.date, Match.home, Match.away, BacktestPrediction.season, BacktestPrediction.model,
            BacktestPrediction.model_version, *(getattr(BacktestPrediction, c) for c in PROBS)]  # fmt: skip
    stmt = select(*cols).join(Match, Match.id == BacktestPrediction.match_id)
    with session_factory() as session:
        frame = pd.DataFrame(session.execute(stmt).all(), columns=COLUMNS)
    frame[PROBS] = frame[PROBS].round(4)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["season", *KEY]).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    log.info("exported %d backtest rows to %s (%d bytes)", len(frame), path, path.stat().st_size)
    return len(frame)


def read_bundle(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"league": str, "home": str, "away": str, "model": str, "model_version": str})
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"backtest bundle is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d").dt.date
    return frame[COLUMNS]


def _matches_frame(session_factory: SessionFactory, leagues: list[str], lo, hi) -> pd.DataFrame:
    stmt = select(Match.id, Match.league, Match.date, Match.home, Match.away).where(
        Match.league.in_(leagues), Match.date.between(lo, hi)
    )
    with session_factory() as session:
        return pd.DataFrame(session.execute(stmt).all(), columns=["match_id", *KEY])


def _swap_day_month(day) -> object | None:
    try:
        return day.replace(month=day.day, day=day.month)
    except ValueError:
        return None


def _unique_on(matches: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return matches.drop_duplicates(cols, keep=False)[[*cols, "match_id"]]


def join_matches(bundle: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """bundle + ``match_id`` (NaN when no match); see the module docstring for the rules."""
    out = bundle.assign(match_id=pd.NA, _date=bundle["date"])
    variants = [("exact", lambda d: d), ("swapped", _swap_day_month),
                ("next day", lambda d: d + dt.timedelta(days=1)), ("previous day", lambda d: d - dt.timedelta(days=1))]  # fmt: skip
    lookups = [KEY, ["league", "date", "home"], ["league", "date", "away"]]
    for _name, shift in variants:
        for cols in lookups:
            todo = out["match_id"].isna()
            if not todo.any():
                return out.drop(columns="_date")
            probe = out.loc[todo].copy()
            probe["date"] = probe["_date"].map(shift)
            probe = probe.dropna(subset=["date"])
            hit = probe.reset_index().merge(_unique_on(matches, cols), on=cols, how="inner", suffixes=("", "_m"))
            if hit.empty:
                continue
            out.loc[hit["index"].to_numpy(), "match_id"] = hit["match_id_m"].to_numpy()
    return out.drop(columns="_date")


def import_bundle(session_factory: SessionFactory, path: Path = DEFAULT_PATH) -> ImportResult:
    """Load the shipped backtest into an empty / outdated table (idempotent)."""
    path = Path(path)
    if not path.exists():
        log.info("no backtest bundle at %s", path)
        return ImportResult("empty")
    bundle = read_bundle(path)
    if bundle.empty:
        return ImportResult("empty")
    versions = sorted(bundle["model_version"].unique())
    with session_factory() as session:
        Base.metadata.create_all(session.get_bind(), tables=[BacktestPrediction.__table__])
        present = set(session.scalars(select(BacktestPrediction.model_version).distinct()))
    if set(versions) <= present:
        log.info("backtest already imported (%d model versions)", len(versions))
        return ImportResult("skipped", rows=len(bundle))
    matches = _matches_frame(session_factory, sorted(bundle["league"].unique()),
                             min(bundle["date"]) - dt.timedelta(days=40), max(bundle["date"]) + dt.timedelta(days=40))  # fmt: skip
    joined = join_matches(bundle, matches).dropna(subset=["match_id"])
    joined = joined.drop_duplicates("match_id", keep="first")
    rows = [
        {"match_id": int(r.match_id), "season": int(r.season), "model": r.model, "model_version": r.model_version,
         "prob_home": float(r.prob_home), "prob_draw": float(r.prob_draw), "prob_away": float(r.prob_away)}
        for r in joined.itertuples(index=False)
    ]  # fmt: skip
    with session_factory() as session:
        session.execute(delete(BacktestPrediction).where(BacktestPrediction.model_version.in_(versions)))
        ids = [r["match_id"] for r in rows]
        for i in range(0, len(ids), CHUNK):  # match_id is unique: drop rows of other versions too
            session.execute(delete(BacktestPrediction).where(BacktestPrediction.match_id.in_(ids[i : i + CHUNK])))
        for i in range(0, len(rows), CHUNK):
            session.execute(insert(BacktestPrediction), rows[i : i + CHUNK])
        session.commit()
        total = session.scalar(select(func.count()).select_from(BacktestPrediction))
    result = ImportResult("imported", rows=len(bundle), joined=len(rows), unmatched=len(bundle) - len(rows))
    log.info("imported backtest: %d of %d rows joined to matches (%d unmatched, table now %d)",
             result.joined, result.rows, result.unmatched, total)  # fmt: skip
    return result


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export / import the shipped backtest predictions")
    parser.add_argument("command", choices=("export", "import"))
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--database-url", default=settings.database_url)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    factory = create_session_factory(args.database_url)
    if args.command == "export":
        print(json.dumps({"rows": export_bundle(factory, args.path)}))  # CLI output
    else:
        print(json.dumps(asdict(import_bundle(factory, args.path))))  # CLI output


if __name__ == "__main__":  # pragma: no cover
    main()
