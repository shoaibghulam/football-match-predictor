"""openfootball/champions-league (CC0, Football.TXT) downloader and parser.

Files per season folder ``2011-12`` … ``2025-26``: ``cl.txt`` (Champions League),
``clq.txt`` (CL qualifiers), ``el.txt``/``elq.txt`` (Europa League), ``conf.txt``/
``confq.txt`` (Conference League). Only the files that exist are downloaded; they
are cached under ``backend/data/openfootball/<season>/<file>``.

Scores: the 90-minute score is used. ``3-2 a.e.t. (3-0, 1-0)`` -> 3-0 and
``4-3 pen. 1-1 a.e.t. (1-1, 0-1)`` -> 1-1. Cancelled/awarded/unplayed matches are skipped.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from matchpredict.ingest.http import Fetcher, FetchError, NotFoundError, fetch_bytes

log = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/openfootball/champions-league/master"
FIRST_SEASON = 2011
FILES: dict[str, tuple[str, str]] = {  # file stem -> (league code, stage family)
    "cl": ("UCL", "main"),
    "clq": ("UCL", "qual"),
    "el": ("UEL", "main"),
    "elq": ("UEL", "qual"),
    "conf": ("UECL", "main"),
    "confq": ("UECL", "qual"),
}
REQUEST_DELAY = 0.2

_MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), start=1
)}  # fmt: skip
_DATE_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$"
)
_MATCH_RE = re.compile(
    r"^\s*(?:(?P<time>\d{1,2}[:.]\d{2})\s+)?"
    r"(?P<home>\S.*?)\s*\((?P<hc>[A-Z]{3})\)\s+v\s+"
    r"(?P<away>\S.*?)\s*\((?P<ac>[A-Z]{3})\)\s*(?P<result>.*)$"
)
_SCORE_RE = re.compile(r"(\d+)-(\d+)")
_HEADER_RE = re.compile(r"^=\s*.*?(\d{4})/(\d{2})\s*$")
_SECTION_RE = re.compile(r"^\s*▪\s*(?P<name>.+?)\s*$")

COLUMNS = (
    "date", "time", "league", "season", "stage", "home_raw", "home_country",
    "away_raw", "away_country", "home_goals", "away_goals", "neutral",
)  # fmt: skip


@dataclass(frozen=True)
class SourceFile:
    season: str  # "2025-26"
    stem: str  # "cl"

    @property
    def url(self) -> str:
        return f"{RAW_BASE}/{self.season}/{self.stem}.txt"

    @property
    def league(self) -> str:
        return FILES[self.stem][0]


def season_folder(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def seasons_until(today: date) -> list[str]:
    last = today.year if today.month >= 7 else today.year - 1
    return [season_folder(y) for y in range(FIRST_SEASON, last + 1)]


def classify_stage(section: str, family: str) -> str:
    """Section heading -> "qual" | "league" (group / league phase) | "ko" | "final"."""
    text = section.lower()
    if family == "qual" or re.search(r"\bround\s*\d|\d\.\s*round|qualif", text):
        return "qual"
    if "final" in text and not re.search(r"semi|quarter|viertel|halb|achtel|sechzehntel|round of", text):
        return "final"
    if re.search(r"group|gruppe|league|matchday", text) and "playoff" not in text:
        return "league"
    return "ko"


def parse_result(text: str) -> tuple[int, int] | None:
    """90-minute score from a result string, or None when not played / not usable."""
    if "[" in text or not text.strip():
        return None  # [cancelled], [awarded], ...
    paren = re.search(r"\(([^)]*)\)", text)
    extra_time = "a.e.t." in text or "pen." in text
    if extra_time:
        if paren is None:
            return None
        hit = _SCORE_RE.search(paren.group(1))
    else:
        hit = _SCORE_RE.search(text)
    if hit is None:
        return None
    return int(hit.group(1)), int(hit.group(2))


def _resolve_date(day: re.Match, start_year: int, previous: date | None) -> date | None:
    """Explicit year wins; otherwise the season year closest to the previous date line."""
    month, dom = _MONTHS[day.group(1)], int(day.group(2))
    if day.group(3):
        return date(int(day.group(3)), month, dom) if _valid(int(day.group(3)), month, dom) else None
    options = [date(y, month, dom) for y in (start_year, start_year + 1) if _valid(y, month, dom)]
    if not options:
        return None  # invalid date line: following matches are skipped
    if previous is None:
        return options[0] if month >= 7 or len(options) == 1 else options[-1]
    return min(options, key=lambda d: abs((d - previous).days))


def _valid(year: int, month: int, dom: int) -> bool:
    try:
        date(year, month, dom)
    except ValueError:
        return False
    return True


def parse_text(text: str, league: str, family: str, season: str) -> pd.DataFrame:
    """Parse one Football.TXT file into rows with COLUMNS (raw team names)."""
    start_year = int(season[:4])
    current: date | None = None
    section = ""
    last_time: str | None = None
    rows: list[dict] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("="):
            header = _HEADER_RE.match(line)
            if header:
                start_year = int(header.group(1))
            continue
        sec = _SECTION_RE.match(line)
        if sec:
            section = sec.group("name")
            last_time = None
            continue
        day = _DATE_RE.match(line)
        if day:
            current = _resolve_date(day, start_year, current)
            last_time = None
            continue
        match = _MATCH_RE.match(line)
        if match is None or current is None:
            continue
        if match.group("time"):
            last_time = match.group("time").replace(".", ":").zfill(5)
        score = parse_result(match.group("result"))
        if score is None:
            continue
        stage = classify_stage(section, family)
        rows.append(
            {
                "date": pd.Timestamp(current),
                "time": last_time,
                "league": league,
                "season": season,
                "stage": stage,
                "home_raw": match.group("home").strip(),
                "home_country": match.group("hc"),
                "away_raw": match.group("away").strip(),
                "away_country": match.group("ac"),
                "home_goals": float(score[0]),
                "away_goals": float(score[1]),
                "neutral": stage == "final",
            }
        )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame


def candidate_files(seasons: Iterable[str]) -> list[SourceFile]:
    return [SourceFile(s, stem) for s in seasons for stem in FILES]


def download(
    cache_dir: Path,
    today: date,
    *,
    fetch: Fetcher = fetch_bytes,
    refresh_current: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Download missing files into ``cache_dir`` (the current season is re-fetched)."""
    seasons = seasons_until(today)
    paths: list[Path] = []
    missing_marker = ".missing"
    for src in candidate_files(seasons):
        path = cache_dir / src.season / f"{src.stem}.txt"
        marker = path.with_suffix(missing_marker)
        current = src.season == seasons[-1]
        if path.exists() and not (current and refresh_current):
            paths.append(path)
            continue
        if marker.exists() and not current:
            continue
        try:
            content = fetch(src.url)
        except NotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("404\n", encoding="utf-8")
            continue
        except FetchError as exc:
            log.warning("openfootball download failed for %s: %s", src.url, exc)
            if path.exists():
                paths.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths.append(path)
        sleep(REQUEST_DELAY)
    return paths


def load_cached(cache_dir: Path) -> pd.DataFrame:
    """Parse every cached file into one chronological frame."""
    frames = []
    for path in sorted(cache_dir.glob("*/*.txt")):
        stem = path.stem
        if stem not in FILES:
            continue
        league, family = FILES[stem]
        text = path.read_bytes().decode("utf-8", errors="replace")
        frames.append(parse_text(text, league, family, path.parent.name))
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "time"], na_position="first", kind="mergesort").reset_index(
        drop=True
    )
