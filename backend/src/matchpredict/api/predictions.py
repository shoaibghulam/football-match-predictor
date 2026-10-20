"""Which model prices a match, plus "featured" ranking (UEFA + big domestic games).

* UEFA competitions (UCL / UEL / UECL) -> the European model.
* Domestic fixtures -> the domestic model; the European model is the fallback when the
  two teams' leagues are in different countries or the domestic model cannot price it.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from matchpredict.api.schemas import PredictionOut
from matchpredict.db.models import Match
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES, LEAGUES
from matchpredict.loader import ServiceHolder

log = logging.getLogger(__name__)

FEATURED_MIN_ELO = 1780.0  # both clubs at least this strong (ClubElo scale) -> "big match"
FEATURED_LIMIT = 12
FEATURED_DOMESTIC_MIN_ELO = 1600.0  # weaker domestic games are never "featured"
# Leagues whose clubs are on the ClubElo (European) scale.
NON_EUROPEAN_COUNTRIES = frozenset({"Argentina", "Brazil", "China", "Japan", "Mexico", "USA"})

OddsTriple = tuple[float, float, float] | None


@dataclass(frozen=True)
class Models:
    domestic: ServiceHolder
    europe: ServiceHolder


def _country(league: str | None) -> str | None:
    return LEAGUES[league].country if league in LEAGUES else None


def cross_country(holder: ServiceHolder, home: str, away: str) -> bool:
    service = holder.service
    if service is None:
        return False
    known = service.known_teams()
    lh, la = known.get(home), known.get(away)
    if lh is None or la is None:
        return False
    return _country(lh) != _country(la)


def _europe(
    models: Models, home: str, away: str, league: str, day: pd.Timestamp, stage: str | None = None
) -> PredictionOut | None:
    if not models.europe.loaded:
        return None
    try:
        uefa_stage = stage if league in EUROPEAN_LEAGUES else None
        return PredictionOut.model_validate(models.europe.predict(home, away, league, day, None, uefa_stage))
    except Exception as exc:  # noqa: BLE001 - unknown team etc.
        log.info("europe prediction unavailable for %s v %s (%s): %s", home, away, league, exc)
        return None


def predict_match(
    models: Models,
    home: str,
    away: str,
    league: str,
    day: pd.Timestamp,
    odds: OddsTriple,
    stage: str | None = None,
) -> PredictionOut | None:
    if league in EUROPEAN_LEAGUES:
        return _europe(models, home, away, league, day, stage)
    if models.domestic.loaded and not cross_country(models.domestic, home, away):
        try:
            return PredictionOut.model_validate(models.domestic.predict(home, away, league, day, odds))
        except Exception as exc:  # noqa: BLE001 - one bad fixture must not fail the list
            log.warning("prediction failed for %s v %s (%s): %s", home, away, league, exc)
    return _europe(models, home, away, league, day)


def strength(models: Models, team: str) -> float | None:
    service = models.europe.service
    if service is None:
        return None
    try:
        return service.strength(team)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


def is_big_match(models: Models, match: Match) -> bool:
    if match.league in EUROPEAN_LEAGUES:
        return True
    h, a = strength(models, match.home), strength(models, match.away)
    return h is not None and a is not None and min(h, a) >= FEATURED_MIN_ELO


def rank_featured(models: Models, live: list[Match], day_matches: list[Match]) -> list[Match]:
    """Live first, then UEFA matches of the day, then the strongest domestic pairings."""
    seen: set[int] = set()
    out: list[Match] = []

    def push(items: list[Match]) -> None:
        for m in items:
            if m.id not in seen and len(out) < FEATURED_LIMIT:
                seen.add(m.id)
                out.append(m)

    push(sorted(live, key=lambda m: (m.league not in EUROPEAN_LEAGUES, m.date, m.time or "")))
    push([m for m in day_matches if m.league in EUROPEAN_LEAGUES])
    scored = []
    for m in day_matches:
        if m.league in EUROPEAN_LEAGUES or _country(m.league) in NON_EUROPEAN_COUNTRIES:
            continue
        h, a = strength(models, m.home), strength(models, m.away)
        if h is not None and a is not None and min(h, a) >= FEATURED_DOMESTIC_MIN_ELO:
            scored.append((h + a, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    push([m for _, m in scored])
    return out


HALF = 45
HALF_TIME_BREAK = 15
FULL_TIME = 90


def estimate_minute(kickoff: dt.datetime | None, now: dt.datetime) -> int | None:
    """Rough match minute from the kick-off time (15-minute half-time break)."""
    if kickoff is None:
        return None
    elapsed = int((now - kickoff).total_seconds() // 60)
    if elapsed < 0:
        return None
    if elapsed <= HALF:
        return max(elapsed, 1)
    if elapsed <= HALF + HALF_TIME_BREAK:
        return HALF
    return min(elapsed - HALF_TIME_BREAK, FULL_TIME)
