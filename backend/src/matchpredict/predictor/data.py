"""Load the football-data.co.uk style history CSV into the shared MATCH_COLUMNS frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matchpredict.contracts import MATCH_COLUMNS

SEASON_START_MONTH = 7  # seasons run Jul 1 .. Jun 30

RAW_TO_CONTRACT: dict[str, str] = {
    "MatchDate": "date",
    "Division": "league",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHome": "home_goals",
    "FTAway": "away_goals",
    "HomeShots": "home_shots",
    "AwayShots": "away_shots",
    "HomeTarget": "home_target",
    "AwayTarget": "away_target",
    "HomeCorners": "home_corners",
    "AwayCorners": "away_corners",
    "OddHome": "odds_home",
    "OddDraw": "odds_draw",
    "OddAway": "odds_away",
}


def _normalise_time(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().split(":")
    if len(parts) < 2:
        return None
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}"


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological order: date, then kickoff time (unknown times first), stable."""
    time_key = df["time"].fillna("")
    order = np.lexsort((time_key.to_numpy(dtype=str), df["date"].to_numpy()))
    return df.iloc[order].reset_index(drop=True)


def load_history(path: str | Path) -> pd.DataFrame:
    """Read Matches.csv -> DataFrame with exactly MATCH_COLUMNS, finished matches only."""
    raw = pd.read_csv(path, low_memory=False)
    missing = {"MatchDate", "Division", "HomeTeam", "AwayTeam", "FTHome", "FTAway"} - set(raw.columns)
    if missing:
        raise ValueError(f"history file is missing columns: {sorted(missing)}")
    df = pd.DataFrame({dst: raw[src] if src in raw.columns else np.nan for src, dst in RAW_TO_CONTRACT.items()})
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    raw_time = raw["MatchTime"] if "MatchTime" in raw.columns else pd.Series([None] * len(raw))
    df["time"] = pd.Series([_normalise_time(v) for v in raw_time], index=raw.index, dtype=object)
    df["home_xg"] = np.nan
    df["away_xg"] = np.nan
    numeric = [c for c in MATCH_COLUMNS if c not in ("date", "time", "league", "home", "away")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.dropna(subset=["home_goals", "away_goals", "home", "away", "date"])
    df = df[list(MATCH_COLUMNS)]
    return sort_matches(df)


def outcomes(df: pd.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    diff = df["home_goals"].to_numpy(dtype=float) - df["away_goals"].to_numpy(dtype=float)
    return np.where(diff > 0, 0, np.where(diff == 0, 1, 2)).astype(int)


def season_of(dates: pd.Series) -> np.ndarray:
    """Season label = calendar year in which the season starts (2024 -> 2024/25)."""
    d = pd.to_datetime(dates)
    return np.where(d.dt.month >= SEASON_START_MONTH, d.dt.year, d.dt.year - 1).astype(int)


def season_start(season: int) -> pd.Timestamp:
    return pd.Timestamp(year=season, month=SEASON_START_MONTH, day=1)
