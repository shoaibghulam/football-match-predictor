"""football-data.co.uk client: fixtures and current-season results.

All parsers return DataFrames with exactly ``contracts.MATCH_COLUMNS``.
Odds are pre-match only (Bet365, else market average) — closing odds
(``...C...`` columns) are never used.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.ingest.http import Fetcher, FetchError, fetch_bytes
from matchpredict.ingest.leagues import (
    COUNTRY_TO_CODE,
    LEAGUES,
    MAIN_DIVISIONS,
    NEW_LEAGUE_FILES,
    TEAM_ALIASES,
)

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk"
FIXTURES_URL = f"{BASE_URL}/fixtures.csv"
NEW_FIXTURES_URL = f"{BASE_URL}/new_league_fixtures.csv"
REQUEST_DELAY = 0.5  # seconds between sequential downloads (be polite)

_ODDS_SOURCES = (("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"))
_FLOAT_COLS = [c for c in MATCH_COLUMNS if c not in ("date", "time", "league", "home", "away")]


@dataclass
class FetchResult:
    frame: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def season_code(day: date) -> str:
    """Season folder for a date: seasons start in July (2026-10-20 -> "2627")."""
    start = day.year if day.month >= 7 else day.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def results_url(div: str, season: str) -> str:
    return f"{BASE_URL}/mmz4281/{season}/{div}.csv"


def new_league_url(code: str) -> str:
    return f"{BASE_URL}/new/{NEW_LEAGUE_FILES[code]}.csv"


def empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in MATCH_COLUMNS})
    frame["date"] = pd.Series(dtype="datetime64[ns]")
    for col in ("time", "league", "home", "away"):
        frame[col] = pd.Series(dtype="object")
    return frame[list(MATCH_COLUMNS)]


def read_csv_bytes(content: bytes | str) -> pd.DataFrame:
    """Read a football-data CSV, tolerating a UTF-8 BOM and latin-1 bytes."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
    else:
        text = content.lstrip("﻿")
    if not text.strip():
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True, on_bad_lines="skip")
    raw.columns = [str(c).strip().lstrip("﻿") for c in raw.columns]
    return raw


def parse_dates(values: pd.Series) -> pd.Series:
    """dd/mm/YYYY with a dd/mm/yy fallback; unparsable -> NaT. Day precision."""
    s = values.astype("string").str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return out.fillna(short).dt.normalize().astype("datetime64[ns]")


def parse_times(values: pd.Series | None, n: int) -> list[str | None]:
    if values is None:
        return [None] * n
    out: list[str | None] = []
    for v in values.tolist():
        if not isinstance(v, str) or ":" not in v:
            out.append(None)
            continue
        hh, mm = v.strip().split(":")[:2]
        out.append(f"{int(hh):02d}:{int(mm):02d}" if hh.isdigit() and mm.isdigit() else None)
    return out


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[col], errors="coerce").astype("float64")


def pick_odds(raw: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per row: first complete pre-match triple among Bet365, Avg (never closing)."""
    nan = pd.Series(np.nan, index=raw.index, dtype="float64")
    home, draw, away = nan.copy(), nan.copy(), nan.copy()
    for h, d, a in _ODDS_SOURCES:
        oh, od, oa = _num(raw, h), _num(raw, d), _num(raw, a)
        valid = oh.gt(1) & od.gt(1) & oa.gt(1) & home.isna()
        home = home.mask(valid, oh)
        draw = draw.mask(valid, od)
        away = away.mask(valid, oa)
    return home, draw, away


def apply_team_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename teams whose current spelling differs from the training history."""
    for league, aliases in TEAM_ALIASES.items():
        mask = frame["league"] == league
        if mask.any():
            for col in ("home", "away"):
                frame.loc[mask, col] = frame.loc[mask, col].replace(aliases)
    return frame


def finalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean, canonicalise names, dedupe and order a MATCH_COLUMNS frame."""
    frame = frame.dropna(subset=["date", "home", "away", "league"])
    frame = frame[(frame["home"] != "") & (frame["away"] != "")]
    frame = frame.drop_duplicates(subset=["league", "date", "home", "away"], keep="last")
    frame = frame.copy()
    for col in _FLOAT_COLS:
        frame[col] = frame[col].astype("float64")
    for col in ("time", "league", "home", "away"):
        frame[col] = frame[col].astype(object).where(frame[col].notna(), None)
    frame = apply_team_aliases(frame)
    return frame[list(MATCH_COLUMNS)].sort_values(["date", "league", "home"]).reset_index(drop=True)


def _text(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw.columns:
        return pd.Series(None, index=raw.index, dtype="object")
    return raw[col].astype("string").str.strip().astype("object")


def parse_main_csv(content: bytes | str) -> pd.DataFrame:
    """Parse ``fixtures.csv`` or ``mmz4281/{season}/{DIV}.csv`` (Div/HomeTeam/... layout)."""
    raw = read_csv_bytes(content)
    if raw.empty or "Div" not in raw.columns:
        return empty_frame()
    if "HomeTeam" not in raw.columns and {"HT", "AT"} <= set(raw.columns):
        raw = raw.rename(columns={"HT": "HomeTeam", "AT": "AwayTeam"})  # early-2000s files (e.g. G1)
    if "HomeTeam" not in raw.columns or "AwayTeam" not in raw.columns:
        return empty_frame()
    odds = pick_odds(raw)
    frame = pd.DataFrame(
        {
            "date": parse_dates(raw["Date"]),
            "time": parse_times(raw.get("Time"), len(raw)),
            "league": _text(raw, "Div"),
            "home": _text(raw, "HomeTeam"),
            "away": _text(raw, "AwayTeam"),
            "home_goals": _num(raw, "FTHG"),
            "away_goals": _num(raw, "FTAG"),
            "home_shots": _num(raw, "HS"),
            "away_shots": _num(raw, "AS"),
            "home_target": _num(raw, "HST"),
            "away_target": _num(raw, "AST"),
            "home_corners": _num(raw, "HC"),
            "away_corners": _num(raw, "AC"),
            "home_xg": _num(raw, "HxG"),
            "away_xg": _num(raw, "AxG"),
            "odds_home": odds[0],
            "odds_draw": odds[1],
            "odds_away": odds[2],
        }
    )
    frame = frame[frame["league"].isin(list(LEAGUES))]
    return finalize_frame(frame)


def parse_new_league_csv(content: bytes | str) -> pd.DataFrame:
    """Parse ``new_league_fixtures.csv`` or ``new/{CODE}.csv`` (Country/League/Home/Away layout).

    Result files only carry closing odds, so their odds stay NaN.
    """
    raw = read_csv_bytes(content)
    if raw.empty or not {"Country", "Date", "Home", "Away"} <= set(raw.columns):
        return empty_frame()
    country = _text(raw, "Country")
    unknown = sorted(set(country.dropna()) - set(COUNTRY_TO_CODE))
    if unknown:
        log.info("skipping unknown countries: %s", ", ".join(unknown))
    odds = pick_odds(raw)
    frame = pd.DataFrame(
        {
            "date": parse_dates(raw["Date"]),
            "time": parse_times(raw.get("Time"), len(raw)),
            "league": country.map(COUNTRY_TO_CODE),
            "home": _text(raw, "Home"),
            "away": _text(raw, "Away"),
            "home_goals": _num(raw, "HG"),
            "away_goals": _num(raw, "AG"),
        }
    )
    for col in ("home_shots", "away_shots", "home_target", "away_target",
                "home_corners", "away_corners", "home_xg", "away_xg"):  # fmt: skip
        frame[col] = np.nan
    frame["odds_home"], frame["odds_draw"], frame["odds_away"] = odds
    return finalize_frame(frame)


def _download(
    url: str, fetch: Fetcher, result: FetchResult, parser: Callable[[bytes], pd.DataFrame]
) -> pd.DataFrame | None:
    try:
        frame = parser(fetch(url))
    except FetchError as exc:
        log.warning("download failed: %s", exc)
        result.errors.append(str(exc))
        return None
    except (ValueError, KeyError) as exc:
        log.warning("parse failed for %s: %s", url, exc)
        result.errors.append(f"parse error for {url}: {exc}")
        return None
    result.sources.append(url)
    return frame


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return empty_frame()
    return finalize_frame(pd.concat(parts, ignore_index=True))


def fetch_fixtures(
    fetch: Fetcher = fetch_bytes, sleep: Callable[[float], None] = time.sleep
) -> FetchResult:
    """Upcoming fixtures (next round) for main and extra leagues."""
    result = FetchResult(frame=empty_frame())
    main = _download(FIXTURES_URL, fetch, result, parse_main_csv)
    sleep(REQUEST_DELAY)
    extra = _download(NEW_FIXTURES_URL, fetch, result, parse_new_league_csv)
    frame = _concat([main, extra])
    result.frame = frame[frame["home_goals"].isna()].reset_index(drop=True)
    return result


def fetch_results(
    today: date,
    *,
    divisions: Iterable[str] = MAIN_DIVISIONS,
    new_leagues: Iterable[str] = tuple(NEW_LEAGUE_FILES),
    lookback_days: int = 365,
    since: Mapping[str, date] | None = None,
    fetch: Fetcher = fetch_bytes,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Finished matches of the current season (main) and recent ones (extra leagues).

    Extra-league files hold every season, so rows are kept from the earlier of
    ``since[league]`` (e.g. the DB's latest result there) and ``today - lookback_days``.
    """
    since = since or {}
    result = FetchResult(frame=empty_frame())
    season = season_code(today)
    frames: list[pd.DataFrame | None] = []
    for div in divisions:
        frames.append(_download(results_url(div, season), fetch, result, parse_main_csv))
        sleep(REQUEST_DELAY)
    default_start = today - timedelta(days=lookback_days)
    for code in new_leagues:
        frame = _download(new_league_url(code), fetch, result, parse_new_league_csv)
        if frame is not None:
            start = min(since.get(code, default_start), default_start)
            frames.append(frame[frame["date"] >= pd.Timestamp(start)])
        sleep(REQUEST_DELAY)
    frame = _concat(frames)
    result.frame = frame[frame["home_goals"].notna() & frame["away_goals"].notna()].reset_index(
        drop=True
    )
    return result
