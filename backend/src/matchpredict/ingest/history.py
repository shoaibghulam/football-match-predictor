"""Match-history bootstrap from football-data.co.uk (no ``data/raw/Matches.csv`` needed).

Downloads every season since 2000/01 of the 22 main divisions
(``mmz4281/{YYZZ}/{DIV}.csv``) plus the 16 extra-country files (``new/{CODE}.csv``),
caches them under ``history_cache_dir`` and upserts the finished matches. The Kaggle
``Matches.csv`` the models were trained on is built from the same files, so team names
match the shipped models. Odds follow the ingest rule (Bet365 pre-match, else the market
average; closing odds never).

Idempotent and resumable: cached past-season files are never downloaded again, missing
files (the site answers ``300`` for divisions that did not exist yet) are remembered, and
a ``.complete`` marker makes later starts skip the whole step.

CLI: ``uv run python -m matchpredict.ingest.history [--force]``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import pandas as pd

from matchpredict.config import get_settings
from matchpredict.db import repository
from matchpredict.db.session import SessionFactory, create_session_factory
from matchpredict.ingest import football_data_uk as fduk
from matchpredict.ingest.http import USER_AGENT, FetchError, fetch_with_curl
from matchpredict.ingest.leagues import MAIN_DIVISIONS, NEW_LEAGUE_FILES

log = logging.getLogger(__name__)

# The www host answers every request with a redirect to the bare domain.
HISTORY_BASE = "https://football-data.co.uk"
FIRST_SEASON = 2000  # 2000/01, like Matches.csv
REQUEST_DELAY = 0.1  # seconds between downloads (sequential, polite)
RETRIES = 3
BACKOFF_SECONDS = 1.5
TIMEOUT = 30.0
COMPLETE_MARKER = ".complete"
MISSING_SUFFIX = ".missing"
SOURCE = "history"
_MISSING_STATUS = {300, 404, 410}  # file does not exist for that season

HistoryFetcher = Callable[[str], bytes | None]  # None = the file does not exist
Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class HistoryFile:
    url: str
    path: Path  # cache location
    kind: str  # "main" (Div/HomeTeam layout) | "new" (Country/Home layout)
    refresh: bool  # current season / all-season files: downloaded again on every run


@dataclass
class HistoryResult:
    status: str = "skipped"  # skipped | complete | partial | failed
    files: int = 0
    downloaded: int = 0
    cached: int = 0
    missing: int = 0
    rows: int = 0
    inserted: int = 0
    updated: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# --- plan ---------------------------------------------------------------------------


def season_codes(today: dt.date, first: int = FIRST_SEASON) -> list[str]:
    """"0001" … the season containing ``today`` (seasons start in July)."""
    current = today.year if today.month >= 7 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, current + 1)]


def plan_files(
    cache_dir: Path,
    today: dt.date,
    divisions: Iterable[str] = MAIN_DIVISIONS,
    new_leagues: Mapping[str, str] = NEW_LEAGUE_FILES,
    first: int = FIRST_SEASON,
) -> list[HistoryFile]:
    seasons = season_codes(today, first)
    files = [
        HistoryFile(
            f"{HISTORY_BASE}/mmz4281/{s}/{div}.csv",
            cache_dir / "mmz4281" / s / f"{div}.csv",
            "main",
            refresh=s == seasons[-1],
        )
        for s in seasons
        for div in divisions
    ]
    files += [
        HistoryFile(f"{HISTORY_BASE}/new/{code}.csv", cache_dir / "new" / f"{code}.csv", "new", refresh=True)
        for code in new_leagues.values()
    ]
    return files


# --- download -----------------------------------------------------------------------


def looks_like_csv(content: bytes | None, kind: str) -> bool:
    """A real data file (not the HTML page the site serves for unknown files)."""
    if not content:
        return False
    head = content[:200].lstrip(b"\xef\xbb\xbf").lstrip()
    first_line = head.split(b"\n", 1)[0]
    return first_line.startswith(b"Div,") if kind == "main" else first_line.startswith(b"Country,")


class HttpHistoryFetcher:
    """One keep-alive client, retries, and a sticky ``curl`` fallback for TLS problems."""

    def __init__(self, client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep,
                 curl: Callable[[str], bytes] = fetch_with_curl) -> None:  # fmt: skip
        self._client = client or httpx.Client(
            timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._sleep = sleep
        self._curl = curl
        self.use_curl = False  # switched on once httpx fails but curl works

    def __call__(self, url: str) -> bytes | None:
        if self.use_curl:
            return self._curl(url)
        last: Exception | None = None
        for attempt in range(1, RETRIES + 1):
            try:
                resp = self._client.get(url)
            except (httpx.HTTPError, OSError) as exc:  # transport / TLS
                last = exc
            else:
                if resp.status_code in _MISSING_STATUS:
                    return None
                if resp.status_code == 200:
                    return resp.content
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise FetchError(f"HTTP {resp.status_code} for {url}")
                last = FetchError(f"HTTP {resp.status_code} for {url}")
            if attempt < RETRIES:
                self._sleep(BACKOFF_SECONDS * attempt)
        log.warning("httpx failed for %s (%s); trying curl", url, last)
        content = self._curl(url)  # raises FetchError when curl fails too
        self.use_curl = True
        return content

    def close(self) -> None:
        self._client.close()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def _missing_marker(path: Path) -> Path:
    return path.with_name(path.name + MISSING_SUFFIX)


def download(
    files: Iterable[HistoryFile],
    fetch: HistoryFetcher,
    result: HistoryResult,
    *,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
    fresh: dict[str, bytes] | None = None,
    workers: int = 1,
) -> None:
    """Fill the cache; ``fresh`` (optional) receives the bytes of every refreshed current file.

    ``workers`` > 1 keeps that many keep-alive downloads in flight (each still pausing
    ``REQUEST_DELAY`` between its requests); counters are only touched by the caller's thread.
    """
    files = list(files)
    result.files = len(files)
    todo: list[HistoryFile] = []
    for item in files:
        if not item.refresh and (item.path.exists() or _missing_marker(item.path).exists()):
            result.cached += item.path.exists()
            result.missing += not item.path.exists()
        else:
            todo.append(item)
    done = len(files) - len(todo)
    if progress is not None:
        progress(done, len(files))

    def job(item: HistoryFile) -> tuple[HistoryFile, str, bytes | str | None]:
        outcome = _download_one(item, fetch)
        sleep(REQUEST_DELAY)
        return (item, *outcome)

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="history") as pool:
        for item, outcome, payload in pool.map(job, todo):
            if outcome == "error":
                result.errors.append(str(payload))
            elif outcome == "missing":
                result.missing += 1
            else:
                result.downloaded += 1
                if fresh is not None and item.refresh and isinstance(payload, bytes):
                    fresh[item.url] = payload  # current files only (the first ingest reuses them)
            done += 1
            if progress is not None:
                progress(done, len(files))


def _download_one(item: HistoryFile, fetch: HistoryFetcher) -> tuple[str, bytes | str | None]:
    """("ok", content) | ("missing", None) | ("error", message); writes the cache file."""
    try:
        content = fetch(item.url)
    except FetchError as exc:
        log.warning("history download failed: %s", exc)
        return "error", str(exc)
    if not looks_like_csv(content, item.kind):
        if not item.refresh:  # past seasons never appear later
            _write_atomic(_missing_marker(item.path), b"")
        return "missing", None
    _write_atomic(item.path, content)
    return "ok", content


# --- parse & store ------------------------------------------------------------------


def parse_cached(files: Iterable[HistoryFile]) -> pd.DataFrame:
    """All cached files -> finished matches as one MATCH_COLUMNS frame."""
    frames: list[pd.DataFrame] = []
    for item in files:
        if not item.path.exists():
            continue
        parser = fduk.parse_main_csv if item.kind == "main" else fduk.parse_new_league_csv
        try:
            frames.append(parser(item.path.read_bytes()))
        except (ValueError, KeyError) as exc:
            log.warning("could not parse %s: %s", item.path, exc)
    parts = [f for f in frames if not f.empty]
    if not parts:
        return fduk.empty_frame()
    frame = fduk.finalize_frame(pd.concat(parts, ignore_index=True))
    finished = frame["home_goals"].notna() & frame["away_goals"].notna()
    return frame[finished].reset_index(drop=True)


def store(session_factory: SessionFactory, frame: pd.DataFrame) -> repository.UpsertStats:
    """Bulk insert into an empty table, else idempotent upsert."""
    with session_factory() as session:
        if repository.count_matches(session) == 0:
            return repository.UpsertStats(inserted=repository.bulk_insert_matches(session, frame, SOURCE))
        return repository.upsert_matches(session, frame, SOURCE)


def is_complete(cache_dir: Path) -> bool:
    return (Path(cache_dir) / COMPLETE_MARKER).exists()


def mark_complete(cache_dir: Path, info: Mapping[str, object]) -> None:
    payload = {**info, "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    _write_atomic(Path(cache_dir) / COMPLETE_MARKER, json.dumps(payload, default=str).encode())


def seed_from_web(
    session_factory: SessionFactory,
    cache_dir: Path,
    *,
    today: dt.date | None = None,
    fetch: HistoryFetcher | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
    fresh: dict[str, bytes] | None = None,
    force: bool = False,
    workers: int = 1,
) -> HistoryResult:
    """Download (or reuse) the history files and store them; never raises."""
    started = time.perf_counter()
    cache_dir = Path(cache_dir)
    result = HistoryResult()
    if is_complete(cache_dir) and not force:
        log.info("match history already seeded (%s)", cache_dir / COMPLETE_MARKER)
        return result
    files = plan_files(cache_dir, today or dt.datetime.now(dt.UTC).date())
    owned = fetch is None
    fetcher = fetch or HttpHistoryFetcher(sleep=sleep)
    try:
        download(files, fetcher, result, sleep=sleep, progress=progress, fresh=fresh, workers=workers)
        frame = parse_cached(files)
        stats = store(session_factory, frame)
        result.rows, result.inserted, result.updated = len(frame), stats.inserted, stats.updated
        result.status = "partial" if result.errors else "complete"
        if not result.errors and len(frame):
            mark_complete(cache_dir, {"source": "football-data.co.uk", "rows": len(frame)})
    except Exception as exc:
        log.exception("history seed failed")
        result.status = "failed"
        result.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if owned and isinstance(fetcher, HttpHistoryFetcher):
            fetcher.close()
        result.seconds = round(time.perf_counter() - started, 1)
    log.info("history seed %s: files=%d downloaded=%d cached=%d missing=%d rows=%d inserted=%d "
             "updated=%d errors=%d in %.0fs", result.status, result.files, result.downloaded, result.cached,
             result.missing, result.rows, result.inserted, result.updated, len(result.errors),
             result.seconds)  # fmt: skip
    return result


# --- reuse of fresh downloads by the first ingest ------------------------------------


def _url_key(url: str) -> str:
    return url.replace("://www.", "://", 1)


def memo_fetcher(fresh: Mapping[str, bytes], fallback: Callable[[str], bytes]) -> Callable[[str], bytes]:
    """A Fetcher serving files downloaded moments ago (www / bare host alike), else ``fallback``."""
    cache = {_url_key(url): content for url, content in fresh.items()}

    def fetch(url: str) -> bytes:
        hit = cache.get(_url_key(url))
        return hit if hit is not None else fallback(url)

    return fetch


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Seed match history from football-data.co.uk")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--cache-dir", type=Path, default=settings.history_cache_dir)
    parser.add_argument("--force", action="store_true", help="run even when already complete")
    parser.add_argument("--workers", type=int, default=settings.history_workers)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = seed_from_web(create_session_factory(args.database_url), args.cache_dir, force=args.force,
                           workers=args.workers)
    print(json.dumps(asdict(result), indent=2, default=str))  # CLI output
    if result.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
