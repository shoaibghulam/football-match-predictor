"""Frozen pre-match predictions.

A fixture's prediction is stored in ``prediction_snapshots`` when it is first priced before
kickoff and refreshed (at most every ``snapshot_refresh_minutes``, or when the model version
changes) until kickoff. Live and finished matches are then served from the frozen snapshot,
so they show what the model said *before* the match, not a re-price with post-match ratings.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy.orm import Session

from matchpredict.api.predictions import Models, predict_match
from matchpredict.api.schemas import PredictionOut
from matchpredict.db import repository
from matchpredict.db.models import STATUS_SCHEDULED, Match, PredictionSnapshot
from matchpredict.db.session import SessionFactory
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES

log = logging.getLogger(__name__)

UK_TZ = ZoneInfo("Europe/London")
DEFAULT_REFRESH_MINUTES = 30


def kickoff_utc(match: Match) -> dt.datetime | None:
    """Kickoff as naive UTC (None when the time is unknown)."""
    if not match.time:
        return None
    hh, mm = (int(p) for p in match.time.split(":"))
    local = dt.datetime.combine(match.date, dt.time(hh, mm), tzinfo=UK_TZ)
    return local.astimezone(dt.UTC).replace(tzinfo=None)


def _naive_utc(now: dt.datetime) -> dt.datetime:
    return now.astimezone(dt.UTC).replace(tzinfo=None) if now.tzinfo else now


def is_pre_kickoff(match: Match, now: dt.datetime) -> bool:
    """True while a scheduled fixture has not kicked off (unknown time: before its UK match day)."""
    if match.status != STATUS_SCHEDULED:
        return False
    ko = kickoff_utc(match)
    utc_now = _naive_utc(now)
    if ko is None:
        uk_today = utc_now.replace(tzinfo=dt.UTC).astimezone(UK_TZ).date()
        return uk_today < match.date
    return utc_now < ko


def needs_refresh(
    snapshot: PredictionSnapshot | None, prediction: PredictionOut, now: dt.datetime, refresh_minutes: int
) -> bool:
    if snapshot is None:
        return True
    if snapshot.model_version != prediction.model_version:
        return True
    return _naive_utc(now) - snapshot.updated_at >= dt.timedelta(minutes=refresh_minutes)


def frozen(snapshot: PredictionSnapshot) -> PredictionOut | None:
    try:
        return PredictionOut.model_validate(json.loads(snapshot.payload))
    except Exception:  # noqa: BLE001 - a corrupt row falls back to live pricing
        log.warning("unreadable prediction snapshot for match %s", snapshot.match_id)
        return None


def snapshot_row(match: Match, prediction: PredictionOut) -> dict:
    return {
        "match_id": match.id,
        "model_version": prediction.model_version,
        "prediction": prediction.model_dump(),
        "kickoff": kickoff_utc(match),
    }


def write_snapshots(session: Session, rows: list[dict], now: dt.datetime | None = None) -> int:
    """Best effort: a failed snapshot write must never fail the request / ingest."""
    if not rows:
        return 0
    try:
        return repository.upsert_snapshots(session, rows, now)
    except Exception:
        session.rollback()
        log.exception("could not write %d prediction snapshots", len(rows))
        return 0


def freeze_upcoming(
    session_factory: SessionFactory,
    models: Models,
    now: dt.datetime,
    days_ahead: int = 7,
    refresh_minutes: int = DEFAULT_REFRESH_MINUTES,
) -> int:
    """Price every not-yet-started fixture in the next ``days_ahead`` days and store snapshots."""
    if not (models.domestic.loaded or models.europe.loaded):
        return 0
    uk_today = _naive_utc(now).replace(tzinfo=dt.UTC).astimezone(UK_TZ).date()
    with session_factory() as session:
        matches = [
            m for m in repository.fixtures_between(session, uk_today, uk_today + dt.timedelta(days=days_ahead))
            if is_pre_kickoff(m, now) and (models.domestic.loaded or m.league in EUROPEAN_LEAGUES)
        ]  # fmt: skip
        existing = repository.snapshots_for(session, [m.id for m in matches])
        rows = list(_rows_to_write(models, matches, existing, now, refresh_minutes))
        written = write_snapshots(session, rows, now)
    if written:
        log.info("froze %d pre-match predictions", written)
    return written


def _rows_to_write(
    models: Models,
    matches: Iterable[Match],
    existing: dict[int, PredictionSnapshot],
    now: dt.datetime,
    refresh_minutes: int,
) -> Iterable[dict]:
    for m in matches:
        odds = (m.odds_home, m.odds_draw, m.odds_away)
        triple = None if any(v is None or v <= 1.0 for v in odds) else odds
        pred = predict_match(models, m.home, m.away, m.league, pd.Timestamp(m.date), triple, stage=m.stage)
        if pred is not None and needs_refresh(existing.get(m.id), pred, now, refresh_minutes):
            yield snapshot_row(m, pred)
