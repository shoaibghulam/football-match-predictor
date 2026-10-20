"""Data-access functions over the ``matches`` and ``ingest_runs`` tables."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, insert, or_, select, union_all, update
from sqlalchemy.orm import Session

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.db.models import (
    EXTRA_COLUMNS,
    FLOAT_COLUMNS,
    INT_COLUMNS,
    STATUS_FINISHED,
    STATUS_LIVE,
    STATUS_SCHEDULED,
    STATUSES,
    VALUE_COLUMNS,
    IngestRun,
    Match,
    PredictionSnapshot,
    utcnow,
)
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES

CHUNK = 5000
TEAM_WINDOW_DAYS = 400  # a team is "current" if it played within this window of the latest match

Key = tuple[str, dt.date, str, str]


@dataclass(frozen=True)
class UpsertStats:
    inserted: int = 0
    updated: int = 0

    def __add__(self, other: UpsertStats) -> UpsertStats:
        return UpsertStats(self.inserted + other.inserted, self.updated + other.updated)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    return value


def frame_to_records(df: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    """MATCH_COLUMNS frame -> list of column dicts for the matches table (deduplicated)."""
    if df.empty:
        return []
    frame = df.dropna(subset=["date", "league", "home", "away"])
    frame = frame.drop_duplicates(subset=["league", "date", "home", "away"], keep="last")
    dates = pd.to_datetime(frame["date"]).dt.date.tolist()
    records: list[dict[str, Any]] = []
    cols = {c: frame[c].tolist() for c in MATCH_COLUMNS if c != "date"}
    extras = {c: frame[c].tolist() for c in EXTRA_COLUMNS if c in frame.columns}
    explicit = frame["status"].tolist() if "status" in frame.columns else None
    for i, day in enumerate(dates):
        rec: dict[str, Any] = {"date": day, "source": source}
        for col, values in cols.items():
            rec[col] = _clean(values[i])
        for col in INT_COLUMNS:
            if rec[col] is not None:
                rec[col] = round(float(rec[col]))
        for col in FLOAT_COLUMNS:
            if rec[col] is not None:
                rec[col] = float(rec[col])
        has_score = rec["home_goals"] is not None and rec["away_goals"] is not None
        rec["status"] = STATUS_FINISHED if has_score else STATUS_SCHEDULED
        for col, values in extras.items():
            rec[col] = _clean(values[i])
        status = explicit[i] if explicit is not None else None
        if status in STATUSES:
            rec["status"] = status
        if rec.get("minute") is not None:
            rec["minute"] = round(float(rec["minute"]))
        if rec.get("unmapped") is not None:
            rec["unmapped"] = bool(rec["unmapped"])
        rec["updated_at"] = utcnow()
        records.append(rec)
    return records


def _key(rec: dict[str, Any] | Match) -> Key:
    if isinstance(rec, dict):
        return (rec["league"], rec["date"], rec["home"], rec["away"])
    return (rec.league, rec.date, rec.home, rec.away)


def _existing(session: Session, records: Sequence[dict[str, Any]]) -> dict[Key, dict[str, Any]]:
    leagues = sorted({r["league"] for r in records})
    lo, hi = min(r["date"] for r in records), max(r["date"] for r in records)
    cols = [Match.id, Match.league, Match.date, Match.home, Match.away, Match.status]
    cols += [getattr(Match, c) for c in (*VALUE_COLUMNS, *EXTRA_COLUMNS)]
    out: dict[Key, dict[str, Any]] = {}
    for i in range(0, len(leagues), 500):
        stmt = select(*cols).where(
            Match.league.in_(leagues[i : i + 500]), Match.date.between(lo, hi)
        )
        for row in session.execute(stmt).mappings():
            out[(row["league"], row["date"], row["home"], row["away"])] = dict(row)
    return out


_STATUS_RANK = {STATUS_SCHEDULED: 0, STATUS_LIVE: 1, STATUS_FINISHED: 2}


def _merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any] | None:
    """Changes to apply: new non-null values win; status only moves forward
    (scheduled -> live -> finished) and an older status never overwrites a newer score."""
    changes: dict[str, Any] = {}
    old_rank = _STATUS_RANK.get(old["status"], 0)
    new_rank = _STATUS_RANK.get(new["status"], 0)
    stale = new_rank < old_rank
    for col in VALUE_COLUMNS:
        if stale and col in ("home_goals", "away_goals"):
            continue
        if new[col] is not None and new[col] != old[col]:
            changes[col] = new[col]
    for col in EXTRA_COLUMNS:
        if new.get(col) is not None and new[col] != old.get(col) and not (stale and col == "minute"):
            changes[col] = new[col]
    if new_rank > old_rank:
        changes["status"] = new["status"]
    if new_rank == 2 and old.get("minute") is not None:
        changes["minute"] = None
    if not changes:
        return None
    changes["source"] = new["source"]
    changes["updated_at"] = utcnow()
    return {"id": old["id"], **changes}


def upsert_matches(session: Session, df: pd.DataFrame, source: str = "ingest") -> UpsertStats:
    """Insert new matches and update existing ones (keyed by league, date, home, away)."""
    records = frame_to_records(df, source)
    if not records:
        return UpsertStats()
    existing = _existing(session, records)
    inserts: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    for rec in records:
        old = existing.get(_key(rec))
        if old is None:
            inserts.append(rec)
            continue
        change = _merge(old, rec)
        if change is not None:
            updates.append(change)
    for i in range(0, len(inserts), CHUNK):
        session.execute(insert(Match), inserts[i : i + CHUNK])
    # Bulk UPDATE by primary key; group by key set so every executemany has uniform params.
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for change in updates:
        groups.setdefault(tuple(sorted(change)), []).append(change)
    for rows in groups.values():
        session.execute(update(Match), rows)
    session.commit()
    return UpsertStats(inserted=len(inserts), updated=len(updates))


def bulk_insert_matches(session: Session, df: pd.DataFrame, source: str) -> int:
    """Fast path for empty tables: plain chunked INSERT without lookups."""
    records = frame_to_records(df, source)
    for i in range(0, len(records), CHUNK):
        session.execute(insert(Match), records[i : i + CHUNK])
    session.commit()
    return len(records)


def count_matches(session: Session, status: str | None = None, since: dt.date | None = None) -> int:
    stmt = select(func.count(Match.id))
    if status is not None:
        stmt = stmt.where(Match.status == status)
    if since is not None:
        stmt = stmt.where(Match.date >= since)
    return int(session.scalar(stmt) or 0)


def prune_stale_fixtures(session: Session, before: dt.date) -> int:
    """Delete scheduled/live rows dated before ``before`` (moved, postponed or never closed)."""
    result = session.execute(
        delete(Match).where(Match.status.in_((STATUS_SCHEDULED, STATUS_LIVE)), Match.date < before)
    )
    session.commit()
    return int(result.rowcount or 0)


def fixtures_between(
    session: Session,
    start: dt.date,
    end: dt.date,
    league: str | None = None,
    team: str | None = None,
) -> list[Match]:
    stmt = select(Match).where(Match.date.between(start, end))
    if league:
        stmt = stmt.where(Match.league == league)
    if team:
        stmt = stmt.where(or_(Match.home == team, Match.away == team))
    stmt = stmt.order_by(Match.date, Match.time.is_(None), Match.time, Match.league, Match.home)
    return list(session.scalars(stmt))


def recent_results(session: Session, team: str, n: int = 10) -> list[Match]:
    stmt = (
        select(Match)
        .where(Match.status == STATUS_FINISHED, or_(Match.home == team, Match.away == team))
        .order_by(Match.date.desc(), Match.time.desc())
        .limit(n)
    )
    return list(session.scalars(stmt))


def latest_finished_by_league(session: Session) -> dict[str, dt.date]:
    stmt = (
        select(Match.league, func.max(Match.date))
        .where(Match.status == STATUS_FINISHED)
        .group_by(Match.league)
    )
    return {league: day for league, day in session.execute(stmt)}


def latest_match_date(session: Session) -> dt.date | None:
    return session.scalar(select(func.max(Match.date)))


def teams(
    session: Session,
    league: str | None = None,
    q: str | None = None,
    window_days: int = TEAM_WINDOW_DAYS,
) -> list[tuple[str, str]]:
    """Current teams as (name, league of most recent match), sorted by name."""
    latest = latest_match_date(session)
    if latest is None:
        return []
    since = latest - dt.timedelta(days=window_days)
    home = select(Match.home.label("team"), Match.league, Match.date).where(Match.date >= since)
    away = select(Match.away.label("team"), Match.league, Match.date).where(Match.date >= since)
    sub = union_all(home, away).subquery()
    stmt = select(sub.c.team, sub.c.league, func.max(sub.c.date)).group_by(sub.c.team, sub.c.league)
    if q:
        stmt = stmt.where(sub.c.team.ilike(f"%{_escape_like(q)}%", escape="\\"))
    latest_league: dict[str, tuple[bool, dt.date, str]] = {}
    for team, lg, day in session.execute(stmt):
        rank = (lg not in EUROPEAN_LEAGUES, day)  # a domestic league wins over UEFA
        if team not in latest_league or rank > latest_league[team][:2]:
            latest_league[team] = (*rank, lg)
    rows = [(t, lg) for t, (_, _, lg) in latest_league.items() if league is None or lg == league]
    return sorted(rows, key=lambda r: r[0].lower())


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def team_exists(session: Session, name: str) -> bool:
    stmt = select(Match.id).where(or_(Match.home == name, Match.away == name)).limit(1)
    return session.scalar(stmt) is not None


def leagues(session: Session) -> list[str]:
    return sorted(session.scalars(select(Match.league).distinct()))


def finished_frame(
    session: Session, after: dt.date | None = None, include_europe: bool = False
) -> pd.DataFrame:
    """Finished matches (strictly after ``after``) as a MATCH_COLUMNS frame, chronological.

    UEFA competitions are excluded unless ``include_europe`` (the domestic model never
    sees them); with ``include_europe`` a ``stage`` column is added.
    """
    columns = list(MATCH_COLUMNS) + (["stage"] if include_europe else [])
    stmt = select(*[getattr(Match, c) for c in columns]).where(Match.status == STATUS_FINISHED)
    if not include_europe:
        stmt = stmt.where(Match.league.not_in(EUROPEAN_LEAGUES))
    if after is not None:
        stmt = stmt.where(Match.date > after)
    rows = session.execute(stmt.order_by(Match.date, Match.time)).all()
    frame = pd.DataFrame(rows, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    for col in (*INT_COLUMNS, *FLOAT_COLUMNS):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def live_matches(session: Session) -> list[Match]:
    stmt = select(Match).where(Match.status == STATUS_LIVE).order_by(Match.date, Match.time)
    return list(session.scalars(stmt))


def kickoffs_between(session: Session, start: dt.date, end: dt.date) -> list[Match]:
    """Scheduled or live matches dated within [start, end] (for the live-refresh window)."""
    stmt = select(Match).where(
        Match.status.in_((STATUS_SCHEDULED, STATUS_LIVE)), Match.date.between(start, end)
    )
    return list(session.scalars(stmt.order_by(Match.date, Match.time)))


def known_teams_by_league(session: Session) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, lg in teams(session):
        out.setdefault(lg, []).append(name)
    return out


# --- prediction snapshots --------------------------------------------------------


def snapshots_for(session: Session, match_ids: Sequence[int]) -> dict[int, PredictionSnapshot]:
    out: dict[int, PredictionSnapshot] = {}
    ids = sorted(set(match_ids))
    for i in range(0, len(ids), 500):
        stmt = select(PredictionSnapshot).where(PredictionSnapshot.match_id.in_(ids[i : i + 500]))
        out.update({s.match_id: s for s in session.scalars(stmt)})
    return out


def upsert_snapshots(session: Session, rows: Sequence[dict[str, Any]], now: dt.datetime | None = None) -> int:
    """Insert or refresh pre-kickoff snapshots (one per match); returns rows written.

    Each row: match_id, model_version, prediction (dict, PredictionOut shape), kickoff (UTC or None).
    Callers only pass fixtures that have not kicked off, so a stored snapshot freezes at kickoff."""
    if not rows:
        return 0
    existing = snapshots_for(session, [r["match_id"] for r in rows])
    if now is not None and now.tzinfo is not None:
        now = now.astimezone(dt.UTC).replace(tzinfo=None)
    now = now or utcnow()
    for row in rows:
        pred = row["prediction"]
        values = {
            "model_version": str(row["model_version"])[:40],
            "prob_home": float(pred["probs"]["home"]),
            "prob_draw": float(pred["probs"]["draw"]),
            "prob_away": float(pred["probs"]["away"]),
            "xg_home": float(pred["expected_goals"]["home"]),
            "xg_away": float(pred["expected_goals"]["away"]),
            "payload": json.dumps(pred, default=str),
            "kickoff": row.get("kickoff"),
            "updated_at": now,
        }
        snap = existing.get(row["match_id"])
        if snap is None:
            session.add(PredictionSnapshot(match_id=row["match_id"], created_at=now, **values))
        else:
            for key, value in values.items():
                setattr(snap, key, value)
    session.commit()
    return len(rows)


# --- ingest runs -----------------------------------------------------------------


def start_run(session: Session) -> IngestRun:
    run = IngestRun(status="running", started_at=utcnow())
    session.add(run)
    session.commit()
    return run


def finish_run(session: Session, run: IngestRun, status: str, **fields: Any) -> IngestRun:
    run.status = status
    run.finished_at = utcnow()
    for name, value in fields.items():
        setattr(run, name, value)
    session.add(run)
    session.commit()
    return run


def last_successful_run(session: Session) -> IngestRun | None:
    stmt = (
        select(IngestRun)
        .where(IngestRun.status.in_(("success", "partial")))
        .order_by(IngestRun.finished_at.desc())
        .limit(1)
    )
    return session.scalar(stmt)

