"""Optional football-data.org v4 client: extends the fixture horizon to 14 days.

Only used when a key is configured (``MP_FOOTBALL_DATA_ORG_KEY`` or
``FOOTBALL_DATA_ORG_KEY``). Team names are mapped to football-data.co.uk names
with an alias table, then fuzzy matching against the known teams of the league.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.ingest.football_data_uk import (
    FetchResult,
    empty_frame,
    finalize_frame,
)

log = logging.getLogger(__name__)

API_URL = "https://api.football-data.org/v4/matches"
MAX_RANGE_DAYS = 10  # API limit for dateFrom/dateTo
REQUEST_INTERVAL = 6.5  # free tier: 10 requests / minute
MATCH_THRESHOLD = 85.0
UK_TZ = ZoneInfo("Europe/London")  # football-data.co.uk times are UK local time

COMPETITIONS: dict[str, str] = {
    "PL": "E0", "ELC": "E1", "BL1": "D1", "SA": "I1", "PD": "SP1",
    "FL1": "F1", "DED": "N1", "PPL": "P1", "BSA": "BRA",
}  # fmt: skip

# football-data.org name -> football-data.co.uk name (checked before fuzzy matching)
ALIASES: dict[str, str] = {
    "Manchester United FC": "Man United",
    "Manchester City FC": "Man City",
    "Wolverhampton Wanderers FC": "Wolves",
    "Nottingham Forest FC": "Nott'm Forest",
    "Tottenham Hotspur FC": "Tottenham",
    "Brighton & Hove Albion FC": "Brighton",
    "Newcastle United FC": "Newcastle",
    "West Ham United FC": "West Ham",
    "Sheffield Wednesday FC": "Sheffield Weds",
    "Sheffield United FC": "Sheffield United",
    "Queens Park Rangers FC": "QPR",
    "West Bromwich Albion FC": "West Brom",
    "Leicester City FC": "Leicester",
    "Leeds United FC": "Leeds",
    "FC Bayern München": "Bayern Munich",
    "Borussia Mönchengladbach": "M'gladbach",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "1. FC Köln": "FC Koln",
    "FC St. Pauli 1910": "St Pauli",
    "1. FSV Mainz 05": "Mainz",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "VfL Wolfsburg": "Wolfsburg",
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    "SSC Napoli": "Napoli",
    "AS Roma": "Roma",
    "Hellas Verona FC": "Verona",
    "Club Atlético de Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Betis Balompié": "Betis",
    "RC Celta de Vigo": "Celta",
    "Real Sociedad de Fútbol": "Sociedad",
    "Rayo Vallecano de Madrid": "Vallecano",
    "RCD Espanyol de Barcelona": "Espanol",
    "Deportivo Alavés": "Alaves",
    "Paris Saint-Germain FC": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Saint-Étienne": "St Etienne",
    "Stade Rennais FC 1901": "Rennes",
    "PSV": "PSV Eindhoven",
    "AFC Ajax": "Ajax",
    "Feyenoord Rotterdam": "Feyenoord",
    "AZ": "AZ Alkmaar",
    "Sporting Clube de Portugal": "Sp Lisbon",
    "Sport Lisboa e Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting Clube de Braga": "Sp Braga",
    "Vitória SC": "Guimaraes",
}

_STOPWORDS = re.compile(r"\b(fc|afc|cf|sc|ac|ssc|club|de|cd|rc|sv|vfl|vfb|fk)\b")


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = _STOPWORDS.sub(" ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def map_team(
    names: Iterable[str | None], candidates: list[str], threshold: float = MATCH_THRESHOLD
) -> str | None:
    """Map API name variants (name, shortName, tla) to a known football-data.co.uk team."""
    variants = [n for n in names if n]
    known = set(candidates)
    for name in variants:
        alias = ALIASES.get(name)
        if alias in known or (alias and not candidates):
            return alias
        if name in known:
            return name
    if not candidates:
        return None
    normed = {normalise(c): c for c in candidates}
    best: tuple[str, float] | None = None
    for name in variants:
        hit = process.extractOne(normalise(name), list(normed), scorer=fuzz.token_sort_ratio)
        if hit and (best is None or hit[1] > best[1]):
            best = (normed[hit[0]], hit[1])
    if best and best[1] >= threshold:
        return best[0]
    return None


def _date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=MAX_RANGE_DAYS - 1), end)
        chunks.append((cur, stop))
        cur = stop + timedelta(days=1)
    return chunks


def _row(match: Mapping, league: str, home: str, away: str) -> dict:
    kickoff = datetime.fromisoformat(match["utcDate"]).astimezone(UK_TZ)
    row: dict = {c: np.nan for c in MATCH_COLUMNS}
    row.update(
        date=pd.Timestamp(kickoff.date()),
        time=kickoff.strftime("%H:%M"),
        league=league,
        home=home,
        away=away,
    )
    return row


def parse_matches(payload: Mapping, known: Mapping[str, list[str]]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    misses: list[str] = []
    for match in payload.get("matches", []):
        if match.get("status") not in ("SCHEDULED", "TIMED"):
            continue
        league = COMPETITIONS.get(match.get("competition", {}).get("code", ""))
        if league is None:
            continue
        sides = []
        for side in ("homeTeam", "awayTeam"):
            team = match.get(side) or {}
            names = (team.get("name"), team.get("shortName"), team.get("tla"))
            mapped = map_team(names, known.get(league, []))
            if mapped is None:
                misses.append(f"{league}:{team.get('name')}")
            sides.append(mapped)
        if None not in sides:
            rows.append(_row(match, league, sides[0], sides[1]))
    return rows, misses


def fetch_org_fixtures(
    api_key: str,
    start: date,
    end: date,
    known: Mapping[str, list[str]],
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Scheduled matches between ``start`` and ``end`` for the mapped competitions."""
    result = FetchResult(frame=empty_frame())
    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    rows: list[dict] = []
    try:
        for i, (lo, hi) in enumerate(_date_chunks(start, end)):
            if i:
                sleep(REQUEST_INTERVAL)
            params = {
                "dateFrom": lo.isoformat(),
                "dateTo": hi.isoformat(),
                "competitions": ",".join(COMPETITIONS),
            }
            try:
                resp = client.get(API_URL, params=params, headers={"X-Auth-Token": api_key})
                resp.raise_for_status()
                payload = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("football-data.org request failed: %s", exc)
                result.errors.append(f"football-data.org {lo}..{hi}: {exc}")
                continue
            got, misses = parse_matches(payload, known)
            if misses:
                log.info("football-data.org unmapped teams: %s", ", ".join(sorted(set(misses))))
            rows.extend(got)
            result.sources.append(f"{API_URL}?dateFrom={lo}&dateTo={hi}")
    finally:
        if owns:
            client.close()
    if rows:
        result.frame = finalize_frame(pd.DataFrame(rows))
    return result


# --- UEFA Champions League (free tier: competition "CL") ------------------------------

COMPETITION_URL = "https://api.football-data.org/v4/competitions/{code}/matches"
EUROPEAN_COMPETITIONS: dict[str, str] = {"CL": "UCL"}  # EL/ECL are not on the free tier
_ORG_LIVE = {"IN_PLAY", "PAUSED", "LIVE", "EXTRA_TIME", "PENALTY_SHOOTOUT"}
_ORG_FINISHED = {"FINISHED", "AWARDED"}
_ORG_SCHEDULED = {"SCHEDULED", "TIMED"}


def org_status(raw: str | None) -> str | None:
    """football-data.org status -> scheduled | live | finished (None = skip)."""
    if raw in _ORG_LIVE:
        return "live"
    if raw in _ORG_FINISHED:
        return "finished"
    if raw in _ORG_SCHEDULED:
        return "scheduled"
    return None  # POSTPONED, SUSPENDED, CANCELLED


def org_stage(raw: str | None) -> str:
    text = (raw or "").upper()
    if "QUALIF" in text or "PRELIM" in text:
        return "qual"
    if text in ("LEAGUE_STAGE", "GROUP_STAGE", "REGULAR_SEASON"):
        return "league"
    return "final" if text == "FINAL" else "ko"


def _org_score(match: Mapping, status: str) -> tuple[float, float]:
    if status == "scheduled":
        return np.nan, np.nan
    score = match.get("score") or {}
    part = score.get("regularTime") if status == "finished" else None
    part = part or score.get("fullTime") or {}
    home, away = part.get("home"), part.get("away")
    if home is None or away is None:
        return (0.0, 0.0) if status == "live" else (np.nan, np.nan)
    return float(home), float(away)


def parse_european_matches(payload: Mapping, mapper) -> tuple[list[dict], list[str]]:
    """CL matches (any status) -> rows with MATCH_COLUMNS + status/minute/stage/unmapped."""
    rows: list[dict] = []
    misses: list[str] = []
    for match in payload.get("matches", []):
        status = org_status(match.get("status"))
        code = (match.get("competition") or {}).get("code") or "CL"
        league = EUROPEAN_COMPETITIONS.get(code)
        if status is None or league is None or not match.get("utcDate"):
            continue
        sides, ok = [], True
        for side in ("homeTeam", "awayTeam"):
            team = match.get(side) or {}
            names = [n for n in (team.get("name"), team.get("shortName")) if n]
            if not names:
                ok = False
                break
            hit = next((mapper.map(n) for n in names if mapper.map(n)), None)
            if hit is None:
                misses.append(names[0])
                hit, _ = mapper.resolve(names[0])
            sides.append((hit, hit not in mapper.unmapped))
        if not ok:
            continue
        hg, ag = _org_score(match, status)
        if status == "finished" and np.isnan(hg):
            continue
        row = _row(match, league, sides[0][0], sides[1][0])
        minute = match.get("minute")
        row.update(
            home_goals=hg,
            away_goals=ag,
            status=status,
            minute=int(minute) if status == "live" and str(minute or "").isdigit() else None,
            stage=org_stage(match.get("stage")),
            unmapped=not (sides[0][1] and sides[1][1]),
        )
        rows.append(row)
    return rows, misses


def fetch_org_european(
    api_key: str,
    start: date,
    end: date,
    mapper,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Champions League matches (scheduled, live and finished) between ``start`` and ``end``."""
    result = FetchResult(frame=empty_frame())
    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    rows: list[dict] = []
    try:
        for i, (lo, hi) in enumerate(_date_chunks(start, end)):
            for j, code in enumerate(EUROPEAN_COMPETITIONS):
                if i or j:
                    sleep(REQUEST_INTERVAL)
                url = COMPETITION_URL.format(code=code)
                params = {"dateFrom": lo.isoformat(), "dateTo": hi.isoformat()}
                try:
                    resp = client.get(url, params=params, headers={"X-Auth-Token": api_key})
                    resp.raise_for_status()
                    payload = resp.json()
                except (httpx.HTTPError, ValueError) as exc:
                    log.warning("football-data.org %s request failed: %s", code, exc)
                    result.errors.append(f"football-data.org {code} {lo}..{hi}: {exc}")
                    continue
                got, misses = parse_european_matches(payload, mapper)
                if misses:
                    log.info("football-data.org %s unmapped: %s", code, ", ".join(sorted(set(misses))))
                rows.extend(got)
                result.sources.append(f"{url}?dateFrom={lo}&dateTo={hi}")
    finally:
        if owns:
            client.close()
    if rows:
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        result.frame = frame.drop_duplicates(subset=["league", "date", "home", "away"], keep="last")
    return result
