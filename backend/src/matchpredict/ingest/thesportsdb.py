"""TheSportsDB (free key ``123``) — no-key fallback for UEFA fixtures, results and live scores.

The free tier is limited (``eventsnextleague`` / ``eventspastleague`` return one event,
``eventsseason`` the first 15), but ``eventsround`` returns a whole round. We read the
round numbers of the last and next events and then fetch those rounds. Times are UTC
and are converted to UK local time (the DB convention).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.db.models import STATUS_FINISHED, STATUS_LIVE, STATUS_SCHEDULED
from matchpredict.ingest.http import Fetcher, FetchError, fetch_bytes
from matchpredict.ingest.teamnames import TeamNameMapper

log = logging.getLogger(__name__)

BASE_URL = "https://www.thesportsdb.com/api/v1/json"
FREE_KEY = "123"
REQUEST_INTERVAL = 2.5  # free tier allows ~30 requests / minute
UK_TZ = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")

LEAGUE_IDS: dict[str, int] = {"UCL": 4480, "UEL": 4481, "UECL": 5071}

_LIVE = {"1H", "2H", "HT", "ET", "BT", "P", "LIVE", "INT", "BREAK", "IN PLAY", "PEN LIVE"}
_FINISHED = {"FT", "AET", "PEN", "AP", "MATCH FINISHED", "FINISHED", "AWD", "WO"}
_SKIP = {"PST", "CANC", "ABD", "SUSP", "POSTPONED", "CANCELLED", "ABANDONED"}

FRAME_COLUMNS = (*MATCH_COLUMNS, "status", "minute", "stage", "unmapped")


@dataclass
class TsdbResult:
    frame: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    requests: int = 0


def empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in FRAME_COLUMNS})
    frame["date"] = pd.Series(dtype="datetime64[ns]")
    return frame


def map_status(raw: str | None) -> str | None:
    text = (raw or "").strip().upper()
    if text in _SKIP:
        return None
    if text in _FINISHED:
        return STATUS_FINISHED
    if text in _LIVE or re.fullmatch(r"\d{1,3}('|\+\d+)?'?", text):
        return STATUS_LIVE
    return STATUS_SCHEDULED


def parse_minute(*values: object) -> int | None:
    """First value that looks like a match minute ("67", "45+2'"); "2H" is not one."""
    for value in values:
        hit = re.fullmatch(r"\s*(\d{1,3})\s*(\+\s*\d+)?\s*'?\s*", str(value or ""))
        if hit:
            return int(hit.group(1))
    return None


def _num(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan


def _kickoff(event: Mapping) -> datetime | None:
    stamp = event.get("strTimestamp")
    try:
        if stamp:
            parsed = datetime.fromisoformat(str(stamp))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        day = event.get("dateEvent")
        if not day:
            return None
        clock = (event.get("strTime") or "00:00:00")[:8]
        return datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=UTC)
    except ValueError:
        return None


def stage_of(event: Mapping, kickoff: datetime) -> str:
    try:
        rnd = int(event.get("intRound") or 0)
    except (TypeError, ValueError):
        rnd = 0
    if kickoff.month in (6, 7, 8):
        return "qual"
    if rnd >= 100 or kickoff.month in (2, 3, 4, 5):
        return "final" if rnd == 200 else "ko"
    return "league"


def parse_events(events: list[Mapping], league: str, mapper: TeamNameMapper) -> list[dict]:
    rows: list[dict] = []
    for event in events or []:
        status = map_status(event.get("strStatus"))
        kickoff = _kickoff(event)
        if status is None or kickoff is None:
            continue
        home, home_ok = mapper.resolve(str(event.get("strHomeTeam") or "").strip())
        away, away_ok = mapper.resolve(str(event.get("strAwayTeam") or "").strip())
        if not home or not away:
            continue
        local = kickoff.astimezone(UK_TZ)
        hg, ag = _num(event.get("intHomeScore")), _num(event.get("intAwayScore"))
        if status == STATUS_SCHEDULED:
            hg = ag = np.nan
        if status != STATUS_SCHEDULED and (np.isnan(hg) or np.isnan(ag)):
            if status == STATUS_FINISHED:
                continue
            hg, ag = 0.0, 0.0
        row = {c: np.nan for c in MATCH_COLUMNS}
        row.update(
            date=pd.Timestamp(local.date()),
            time=local.strftime("%H:%M"),
            league=league,
            home=home,
            away=away,
            home_goals=hg,
            away_goals=ag,
            status=status,
            minute=parse_minute(event.get("strProgress"), event.get("strStatus"))
            if status == STATUS_LIVE
            else None,
            stage=stage_of(event, kickoff),
            unmapped=not (home_ok and away_ok),
        )
        rows.append(row)
    return rows


class TsdbClient:
    def __init__(
        self,
        key: str = FREE_KEY,
        *,
        fetch: Fetcher = fetch_bytes,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.key = key
        self.fetch = fetch
        self.sleep = sleep
        self.requests = 0

    def get(self, path: str) -> list[Mapping]:
        if self.requests:
            self.sleep(REQUEST_INTERVAL)
        self.requests += 1
        content = self.fetch(f"{BASE_URL}/{self.key}/{path}")
        payload = json.loads(content.decode("utf-8"))  # always UTF-8 (avoids mojibake)
        events = payload.get("events") if isinstance(payload, dict) else None
        return list(events or [])


def _round_and_season(events: list[Mapping]) -> tuple[int | None, str | None]:
    for event in events:
        try:
            return int(event.get("intRound")), event.get("strSeason")
        except (TypeError, ValueError):
            continue
    return None, None


def fetch_competition(
    client: TsdbClient, league: str, mapper: TeamNameMapper, rounds_only: set[int] | None = None
) -> tuple[list[dict], list[str]]:
    """Events of the last and next round of one competition."""
    league_id = LEAGUE_IDS[league]
    errors: list[str] = []
    rounds: dict[int, str] = {}
    events: list[Mapping] = []
    for path in (f"eventspastleague.php?id={league_id}", f"eventsnextleague.php?id={league_id}"):
        try:
            got = client.get(path)
        except (FetchError, ValueError) as exc:
            errors.append(f"thesportsdb {league} {path}: {exc}")
            continue
        events.extend(got)
        rnd, season = _round_and_season(got)
        if rnd is not None and season:
            rounds[rnd] = season
    for rnd, season in sorted(rounds.items()):
        if rounds_only is not None and rnd not in rounds_only:
            continue
        try:
            events.extend(client.get(f"eventsround.php?id={league_id}&r={rnd}&s={season}"))
        except (FetchError, ValueError) as exc:
            errors.append(f"thesportsdb {league} round {rnd}: {exc}")
    return parse_events(events, league, mapper), errors


def fetch_european(
    mapper: TeamNameMapper,
    leagues: tuple[str, ...] = tuple(LEAGUE_IDS),
    *,
    fetch: Fetcher = fetch_bytes,
    sleep: Callable[[float], None] = time.sleep,
    key: str = FREE_KEY,
) -> TsdbResult:
    client = TsdbClient(key, fetch=fetch, sleep=sleep)
    rows: list[dict] = []
    errors: list[str] = []
    for league in leagues:
        got, errs = fetch_competition(client, league, mapper)
        rows.extend(got)
        errors.extend(errs)
    frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS)) if rows else empty_frame()
    if rows:
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        frame = frame.drop_duplicates(subset=["league", "date", "home", "away"], keep="last")
    return TsdbResult(frame=frame.reset_index(drop=True), errors=errors, requests=client.requests)
