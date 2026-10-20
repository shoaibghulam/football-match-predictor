"""Unified match frame for v2: every finished domestic + UEFA match in the DB, deduplicated.

Columns: ``contracts.MATCH_COLUMNS`` + ``stage`` (``dom`` / ``qual`` / ``league`` / ``ko`` /
``final``), ``neutral`` (finals), ``comp`` (``DOM`` / ``UCL`` / ``UEL`` / ``UECL``), ``source``.
European team names are folded onto canonical names (``names.Canonicaliser``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES
from matchpredict.predictor.data import sort_matches
from matchpredict.predictor.v2.names import Canonicaliser

log = logging.getLogger(__name__)

EXTRA = ("stage", "neutral", "comp", "source")
FRAME_COLUMNS: tuple[str, ...] = (*MATCH_COLUMNS, *EXTRA)
SOURCE_PRIORITY = {"openfootball": 0, "football-data.org": 1, "thesportsdb": 2}
NEAR_DUPLICATE_DAYS = 1
NUMERIC = [c for c in MATCH_COLUMNS if c not in ("date", "time", "league", "home", "away")]


def read_db(database_url: str, include_scheduled: bool = False) -> pd.DataFrame:
    """Raw rows from the ``matches`` table (read-only query; no DDL is executed)."""
    engine = create_engine(database_url)
    status = "" if include_scheduled else "WHERE status = 'finished'"
    cols = ", ".join([*MATCH_COLUMNS, "stage", "source", "status", "id"])
    try:
        with engine.connect() as conn:
            frame = pd.read_sql(text(f"SELECT {cols} FROM matches {status}"), conn)
    finally:
        engine.dispose()
    return frame


def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    """Types + ``comp`` / ``stage`` / ``neutral`` columns for a raw DB / ingest frame."""
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    for col in NUMERIC:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    if "time" not in frame.columns:
        frame["time"] = None
    european = frame["league"].isin(EUROPEAN_LEAGUES)
    stage = frame["stage"] if "stage" in frame.columns else pd.Series([None] * len(frame), index=frame.index)
    frame["stage"] = np.where(european, stage.fillna("league").astype(str), "dom")
    frame["neutral"] = (frame["stage"] == "final").astype(float)
    frame["comp"] = np.where(european, frame["league"], "DOM")
    if "source" not in frame.columns:
        frame["source"] = "ingest"
    frame["source"] = frame["source"].fillna("ingest").astype(str)
    return frame


def canonicalise(frame: pd.DataFrame, names: Canonicaliser) -> pd.DataFrame:
    out = frame.copy()
    out["home"] = [names.canonical(n) for n in out["home"]]
    out["away"] = [names.canonical(n) for n in out["away"]]
    return out


def dedupe_european(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop UEFA rows stored twice (two sources / ±1 day), keeping the higher-priority source."""
    euro = frame["comp"] != "DOM"
    if not euro.any():
        return frame
    part = frame[euro].copy()
    part["_prio"] = part["source"].map(SOURCE_PRIORITY).fillna(9)
    part = part.sort_values(["_prio", "date"], kind="mergesort")
    part = part.drop_duplicates(["league", "date", "home", "away"], keep="first")
    keep: list[int] = []
    seen: dict[tuple[str, str, str], list[np.datetime64]] = {}
    for idx, league, home, away, day in zip(
        part.index, part["league"], part["home"], part["away"], part["date"].to_numpy(), strict=True
    ):
        key = (league, home, away)
        days = seen.setdefault(key, [])
        if any(abs((day - d) / np.timedelta64(1, "D")) <= NEAR_DUPLICATE_DAYS for d in days):
            continue
        days.append(day)
        keep.append(idx)
    dropped = int(euro.sum()) - len(keep)
    if dropped:
        log.info("dropped %d duplicate UEFA rows", dropped)
    return pd.concat([frame[~euro], part.loc[keep].drop(columns="_prio")])


def domestic_leagues(frame: pd.DataFrame) -> dict[str, str]:
    """team -> its most frequent domestic league (defines the club's country)."""
    dom = frame[frame["league"].map(lambda lg: lg not in EUROPEAN_LEAGUES)]
    sides = pd.concat(
        [dom[["home", "league"]].set_axis(["team", "league"], axis=1),
         dom[["away", "league"]].set_axis(["team", "league"], axis=1)]
    )  # fmt: skip
    if sides.empty:
        return {}
    counts = sides.groupby(["team", "league"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).drop_duplicates("team")
    return dict(zip(best["team"], best["league"], strict=True))


def source_countries(openfootball_dir: Path | None, elo_csv: Path | None) -> dict[str, str | None]:
    """Raw / ClubElo club name -> country code, best effort (missing files are skipped)."""
    out: dict[str, str | None] = {}
    if elo_csv is not None and Path(elo_csv).exists():
        elo = pd.read_csv(elo_csv, usecols=["club", "country"]).drop_duplicates("club", keep="last")
        out.update(zip(elo["club"], elo["country"], strict=True))
    if openfootball_dir is not None and Path(openfootball_dir).exists():
        try:
            from matchpredict.ingest import openfootball

            raw = openfootball.load_cached(Path(openfootball_dir))
            for side in ("home", "away"):
                out.update(zip(raw[f"{side}_raw"], raw[f"{side}_country"], strict=True))
        except Exception:
            log.exception("could not read openfootball cache for club countries")
    return out


def build_canonicaliser(
    frame: pd.DataFrame, openfootball_dir: Path | None = None, elo_csv: Path | None = None
) -> Canonicaliser:
    euro = frame[frame["league"].isin(EUROPEAN_LEAGUES)]
    names = list(euro["home"]) + list(euro["away"])
    return Canonicaliser.build(domestic_leagues(frame), names, source_countries(openfootball_dir, elo_csv))


def prepare_frame(raw: pd.DataFrame, names: Canonicaliser) -> pd.DataFrame:
    """standardise -> canonical names -> UEFA dedupe -> chronological order."""
    frame = dedupe_european(canonicalise(standardise(raw), names))
    frame = frame[frame["home"] != frame["away"]]
    return sort_matches(frame[list(FRAME_COLUMNS)])


def load_matches(
    database_url: str, openfootball_dir: Path | None = None, elo_csv: Path | None = None
) -> tuple[pd.DataFrame, Canonicaliser]:
    raw = read_db(database_url)
    names = build_canonicaliser(standardise(raw), openfootball_dir, elo_csv)
    frame = prepare_frame(raw, names)
    log.info("loaded %d finished matches (%d UEFA)", len(frame), int((frame["comp"] != "DOM").sum()))
    return frame, names
