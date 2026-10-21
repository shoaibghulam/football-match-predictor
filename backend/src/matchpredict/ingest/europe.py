"""UEFA competitions ingest: history seed, daily refresh and live refresh.

Sources (team names mapped to canonical DB / ClubElo names; unmapped names are
stored raw with ``unmapped=True``):

* football-data.org ``CL`` (needs ``FOOTBALL_DATA_ORG_KEY``) — fixtures, live
  scores (IN_PLAY / PAUSED) and results of the Champions League;
* TheSportsDB (free key) — UEL / UECL always, and UCL when no key is set;
* openfootball — 2011-12 … history, seeded once with ``seed``.

CLI: ``uv run python -m matchpredict.ingest.europe seed|refresh|live``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from matchpredict.config import Settings, get_settings
from matchpredict.db import repository
from matchpredict.db.models import STATUS_LIVE, Match
from matchpredict.db.session import SessionFactory, create_session_factory
from matchpredict.ingest import football_data_org, openfootball, thesportsdb
from matchpredict.ingest.http import Fetcher, fetch_bytes
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES, LEAGUES
from matchpredict.ingest.teamnames import COUNTRY_CODES, TeamNameMapper, log_misses
from matchpredict.loader import ServiceHolder, artifact_path_of

log = logging.getLogger(__name__)

UK_TZ = ZoneInfo("Europe/London")
REFRESH_BACK_DAYS = 3
REFRESH_AHEAD_DAYS = 7
SOURCE_ORG = "football-data.org"
SOURCE_TSDB = "thesportsdb"
SOURCE_OPENFOOTBALL = "openfootball"


@dataclass
class EuropeSummary:
    status: str = "ok"
    rows: int = 0
    inserted: int = 0
    updated: int = 0
    live: int = 0
    applied_results: int = 0
    sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False


# --- names -------------------------------------------------------------------------


def build_mapper(session_factory: SessionFactory, europe: ServiceHolder | None = None) -> TeamNameMapper:
    """Candidates: DB domestic teams (country from league) + the European model's clubs."""
    cands: dict[str, str | None] = {}
    domestic: set[str] = set()
    service = europe.ensure_loaded() if europe is not None else None
    with session_factory() as session:
        for team, league in repository.teams(session, window_days=800):
            if league in EUROPEAN_LEAGUES:
                continue
            country = COUNTRY_CODES.get(LEAGUES[league].country) if league in LEAGUES else None
            cands.setdefault(team, country)
            domestic.add(team)
    covered: set[str] = set()
    if service is not None:
        clubs, covered = _model_clubs(service)
        for team, country in clubs.items():
            cands.setdefault(team, country)
    mapper = TeamNameMapper(cands)
    # clubs known only by their raw European-source name stay flagged as unmapped
    mapper.unmapped.update(t for t in cands if t not in domestic and t not in covered and service is not None)
    if service is not None:
        mapper.aliases = {**mapper.aliases, **_normalised_aliases(getattr(service, "aliases", {}))}
    return mapper


def _model_clubs(service) -> tuple[dict[str, str | None], set[str]]:
    """Clubs (-> country) the European-capable model rates: v1 EuropeService (`book`) or V2Service."""
    book = getattr(service, "book", None)
    if book is not None:
        return dict(book.country), set(book.covered)
    rating = getattr(getattr(service, "state", None), "rating", None)
    if rating is not None:
        clubs = {team: rating.country(team) for team in rating.r}
        return clubs, set(clubs)
    return {}, set()


def _normalised_aliases(aliases: dict[str, str]) -> dict[str, str]:
    from matchpredict.ingest.teamnames import normalise

    return {normalise(raw): canon for raw, canon in aliases.items() if normalise(raw)}


# --- europe model updates ----------------------------------------------------------


def apply_to_europe(
    session_factory: SessionFactory, europe: ServiceHolder | None, settings: Settings
) -> int:
    """Feed finished domestic + European results newer than the model's last result."""
    service = europe.ensure_loaded() if europe is not None else None
    if service is None:
        return 0
    with europe.lock:  # type: ignore[union-attr]
        last = service.last_result_date
        lookback = int(getattr(service, "update_lookback_days", 1))
        after = (last - pd.Timedelta(days=lookback)).date() if last is not None else None
        with session_factory() as session:
            frame = repository.finished_frame(session, after=after, include_europe=True)
        applied = int(service.update_results(frame)) if not frame.empty else 0
        if applied:
            europe.invalidate()  # type: ignore[union-attr]
            service.save(artifact_path_of(service, settings.europe_artifact_path))
    return applied


# --- refresh -----------------------------------------------------------------------


def _upsert(session_factory: SessionFactory, frame: pd.DataFrame, source: str, summary: EuropeSummary) -> None:
    if frame.empty:
        return
    with session_factory() as session:
        stats = repository.upsert_matches(session, frame, source=source)
    summary.rows += len(frame)
    summary.inserted += stats.inserted
    summary.updated += stats.updated
    summary.live += int((frame.get("status") == STATUS_LIVE).sum()) if "status" in frame else 0
    summary.sources.append(source)


def refresh_european(
    session_factory: SessionFactory,
    settings: Settings | None = None,
    europe: ServiceHolder | None = None,
    *,
    today: dt.date | None = None,
    leagues: tuple[str, ...] = EUROPEAN_LEAGUES,
    fetch: Fetcher = fetch_bytes,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    back_days: int = REFRESH_BACK_DAYS,
    ahead_days: int = REFRESH_AHEAD_DAYS,
) -> EuropeSummary:
    """Fetch UEFA fixtures / live scores / results and upsert them; never raises."""
    settings = settings or get_settings()
    today = today or dt.datetime.now(UK_TZ).date()
    summary = EuropeSummary()
    try:
        mapper = build_mapper(session_factory, europe)
        tsdb_leagues = list(leagues)
        key = settings.football_data_org_key
        if key and "UCL" in leagues:
            org = football_data_org.fetch_org_european(
                key, today - dt.timedelta(days=back_days), today + dt.timedelta(days=ahead_days),
                mapper, client=client, sleep=sleep,
            )  # fmt: skip
            summary.errors.extend(org.errors)
            if not org.errors:
                tsdb_leagues.remove("UCL")
            _upsert(session_factory, org.frame, SOURCE_ORG, summary)
        if tsdb_leagues:
            tsdb = thesportsdb.fetch_european(
                mapper, tuple(tsdb_leagues), fetch=fetch, sleep=sleep, key=settings.thesportsdb_key
            )
            summary.errors.extend(tsdb.errors)
            _upsert(session_factory, tsdb.frame, SOURCE_TSDB, summary)
        log_misses(mapper, "uefa refresh")
        summary.applied_results = apply_to_europe(session_factory, europe, settings)
        if europe is not None:
            europe.invalidate()
            from matchpredict.ingest.run import freeze_snapshots

            freeze_snapshots(session_factory, None, europe, settings)
    except Exception as exc:
        log.exception("european refresh failed")
        summary.errors.append(f"{type(exc).__name__}: {exc}")
    if europe is not None:
        europe.invalidate()
    summary.status = "partial" if summary.errors else "ok"
    return summary


# --- live --------------------------------------------------------------------------


def kickoff_utc(match: Match) -> dt.datetime | None:
    if not match.time:
        return None
    hh, mm = (int(p) for p in match.time.split(":"))
    return dt.datetime.combine(match.date, dt.time(hh, mm), tzinfo=UK_TZ).astimezone(dt.UTC)


def active_leagues(
    session_factory: SessionFactory, now: dt.datetime, window_hours: float
) -> tuple[str, ...]:
    """UEFA leagues with a match in progress or kicking off within ±window of ``now``."""
    window = dt.timedelta(hours=window_hours)
    day = now.astimezone(UK_TZ).date()
    with session_factory() as session:
        matches = repository.kickoffs_between(session, day - dt.timedelta(days=1), day + dt.timedelta(days=1))
    active = set()
    for m in matches:
        if m.league not in EUROPEAN_LEAGUES:
            continue
        ko = kickoff_utc(m)
        if m.status == STATUS_LIVE or (ko is not None and abs(ko - now) <= window):
            active.add(m.league)
    return tuple(lg for lg in EUROPEAN_LEAGUES if lg in active)


def refresh_live(
    session_factory: SessionFactory,
    settings: Settings | None = None,
    europe: ServiceHolder | None = None,
    *,
    now: dt.datetime | None = None,
    fetch: Fetcher = fetch_bytes,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> EuropeSummary:
    """Poll scores only while a UEFA match is live or kicks off within ±3 h (1 request / league)."""
    settings = settings or get_settings()
    now = now or dt.datetime.now(dt.UTC)
    leagues = active_leagues(session_factory, now, settings.live_window_hours)
    if not leagues:
        return EuropeSummary(status="ok", skipped=True)
    return refresh_european(
        session_factory, settings, europe,
        today=now.astimezone(UK_TZ).date(), leagues=leagues, fetch=fetch, client=client,
        sleep=sleep, back_days=1, ahead_days=1,
    )  # fmt: skip


# --- history seed ------------------------------------------------------------------


def history_frame(raw: pd.DataFrame, mapper: TeamNameMapper) -> pd.DataFrame:
    """openfootball rows -> MATCH_COLUMNS frame (+ stage, unmapped, status)."""
    from matchpredict.contracts import MATCH_COLUMNS

    rows = []
    for r in raw.itertuples(index=False):
        home, home_ok = mapper.resolve(r.home_raw, r.home_country)
        away, away_ok = mapper.resolve(r.away_raw, r.away_country)
        row = dict.fromkeys(MATCH_COLUMNS)
        row.update(
            date=r.date, time=r.time, league=r.league, home=home, away=away,
            home_goals=r.home_goals, away_goals=r.away_goals, stage=r.stage,
            unmapped=not (home_ok and away_ok), status="finished",
        )  # fmt: skip
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame


def seed_history(
    session_factory: SessionFactory,
    settings: Settings | None = None,
    europe: ServiceHolder | None = None,
    *,
    today: dt.date | None = None,
    download: bool = True,
    fetch: Fetcher = fetch_bytes,
) -> repository.UpsertStats:
    settings = settings or get_settings()
    today = today or dt.date.today()
    if download:
        openfootball.download(settings.openfootball_dir, today, fetch=fetch)
    raw = openfootball.load_cached(settings.openfootball_dir)
    mapper = build_mapper(session_factory, europe)
    frame = history_frame(raw, mapper)
    log_misses(mapper, "openfootball seed")
    with session_factory() as session:
        stats = repository.upsert_matches(session, frame, source=SOURCE_OPENFOOTBALL)
    log.info("seeded %d European matches (inserted=%d updated=%d)", len(frame), stats.inserted, stats.updated)
    return stats


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    settings = get_settings()
    parser = argparse.ArgumentParser(description="UEFA competitions ingest")
    parser.add_argument("command", choices=("seed", "refresh", "live"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    factory = create_session_factory(settings.database_url)
    from matchpredict.loader import europe_holder

    europe = europe_holder(settings)
    if args.command == "seed":
        stats = seed_history(factory, settings, europe, download=not args.no_download)
        print(json.dumps(asdict(stats)))  # CLI output
        return
    run = refresh_european if args.command == "refresh" else refresh_live
    summary = run(factory, settings, europe)
    print(json.dumps(asdict(summary), indent=2))  # CLI output


if __name__ == "__main__":  # pragma: no cover
    main()
