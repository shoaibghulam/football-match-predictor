"""/api/v1 routes (contract: docs/04-api.md)."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Iterator
from typing import Annotated
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, Depends, Path, Query, Request
from sqlalchemy.orm import Session

from matchpredict.api.errors import APIError
from matchpredict.api.predictions import (
    Models,
    cross_country,
    estimate_minute,
    is_big_match,
    predict_match,
    rank_featured,
)
from matchpredict.api.schemas import (
    FixtureOut,
    FormResult,
    Health,
    LeagueOut,
    LiveInfo,
    Odds,
    PredictResponse,
    Score,
    SetupOut,
    TeamForm,
    TeamOut,
)
from matchpredict.api.snapshots import (
    frozen,
    is_pre_kickoff,
    needs_refresh,
    snapshot_row,
    write_snapshots,
)
from matchpredict.db import repository
from matchpredict.db.models import (
    STATUS_FINISHED,
    STATUS_LIVE,
    STATUS_SCHEDULED,
    Match,
    PredictionSnapshot,
)
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES, LEAGUES
from matchpredict.loader import ServiceHolder

log = logging.getLogger(__name__)

MAX_RANGE_DAYS = 14
FORM_RESULTS = 10
UK_TZ = ZoneInfo("Europe/London")
TEAM_NAME = r"^[\w .'&()/\-]+$"

router = APIRouter()


# --- dependencies ----------------------------------------------------------------


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_holder(request: Request) -> ServiceHolder:
    return request.app.state.holder


def get_models(request: Request) -> Models:
    return Models(domestic=request.app.state.holder, europe=request.app.state.europe)


def get_today(request: Request) -> dt.date:
    clock: Callable[[], dt.date] = request.app.state.today
    return clock()


SessionDep = Annotated[Session, Depends(get_session)]
HolderDep = Annotated[ServiceHolder, Depends(get_holder)]
ModelsDep = Annotated[Models, Depends(get_models)]
TodayDep = Annotated[dt.date, Depends(get_today)]
LeagueQuery = Annotated[str | None, Query(min_length=1, max_length=8, pattern=r"^[A-Z0-9]+$")]
TeamQuery = Annotated[str, Query(min_length=1, max_length=64, pattern=TEAM_NAME)]


def _require_model(holder: ServiceHolder) -> None:
    if not holder.loaded:
        raise APIError(503, "MODEL_NOT_LOADED", "the prediction model is not loaded yet")


def _check_league(league: str | None) -> None:
    if league is not None and league not in LEAGUES:
        raise APIError(404, "LEAGUE_NOT_FOUND", f"unknown league '{league}'")


def _resolve(name: str, known: dict[str, str]) -> str | None:
    if name in known:
        return name
    lowered = name.casefold()
    return next((k for k in known if k.casefold() == lowered), None)


# --- endpoints -------------------------------------------------------------------


@router.get("/health", response_model=Health)
def health(
    session: SessionDep, holder: HolderDep, models: ModelsDep, today: TodayDep, request: Request
) -> Health:
    service = holder.service
    europe = models.europe.service
    last_run = repository.last_successful_run(session)
    last_result = getattr(service, "last_result_date", None) if service else None
    return Health(
        status="ok" if service is not None else "degraded",
        model_loaded=service is not None,
        model_version=getattr(service, "model_version", None) if service else None,
        last_result_date=pd.Timestamp(last_result).date() if last_result is not None else None,
        last_ingest=last_run.finished_at if last_run else None,
        matches=repository.count_matches(session, STATUS_FINISHED),
        fixtures=repository.count_matches(session, STATUS_SCHEDULED, since=today),
        europe_model_loaded=europe is not None,
        europe_model_version=getattr(europe, "model_version", None) if europe else None,
        football_data_org_key_configured=bool(request.app.state.settings.football_data_org_key),
        live_matches=len(repository.live_matches(session)),
        **_v2_health(service, europe),
        setup=_setup_health(request),
    )


def _setup_health(request: Request) -> SetupOut:
    tracker = getattr(request.app.state, "setup", None)
    return SetupOut(**tracker.as_dict()) if tracker is not None else SetupOut()


def _v2_health(domestic: object | None, europe: object | None) -> dict:
    scopes = [
        scope for scope, svc in (("domestic", domestic), ("europe", europe))
        if svc is not None and str(getattr(svc, "model_version", "")).startswith("v2")
    ]  # fmt: skip
    v2 = domestic if "domestic" in scopes else europe if scopes else None
    return {
        "model_v2_loaded": v2 is not None,
        "model_v2_version": getattr(v2, "model_version", None) if v2 is not None else None,
        "model_v2_scopes": scopes,
    }


@router.get("/leagues", response_model=list[LeagueOut])
def leagues() -> list[LeagueOut]:
    return [LeagueOut(code=c, name=lg.name, country=lg.country) for c, lg in LEAGUES.items()]


@router.get("/teams", response_model=list[TeamOut])
def teams(
    session: SessionDep,
    holder: HolderDep,
    league: LeagueQuery = None,
    q: Annotated[str | None, Query(min_length=2, max_length=64)] = None,
) -> list[TeamOut]:
    _check_league(league)
    rows = dict(repository.teams(session, league=league, q=q))
    service = holder.service
    if service is not None:
        needle = q.casefold() if q else None
        for name, lg in service.known_teams().items():
            if name in rows or (league and lg != league) or (needle and needle not in name.casefold()):
                continue
            rows[name] = lg
    return [TeamOut(name=n, league=lg) for n, lg in sorted(rows.items(), key=lambda r: r[0].lower())]


def _kickoff(match: Match) -> dt.datetime | None:
    if not match.time:
        return None
    hh, mm = (int(p) for p in match.time.split(":"))
    return dt.datetime.combine(match.date, dt.time(hh, mm), tzinfo=UK_TZ)


def _odds(match: Match) -> tuple[float, float, float] | None:
    triple = (match.odds_home, match.odds_draw, match.odds_away)
    if any(v is None or v <= 1.0 for v in triple):
        return None
    return triple  # type: ignore[return-value]


def _live_info(match: Match, score: Score | None, now: dt.datetime) -> LiveInfo | None:
    if match.status != STATUS_LIVE or score is None:
        return None
    if match.minute is not None:
        return LiveInfo(minute=match.minute, score=score)
    return LiveInfo(minute=estimate_minute(_kickoff(match), now), minute_estimated=True, score=score)


def _fixture_out(
    models: Models,
    match: Match,
    now: dt.datetime,
    featured: bool | None = None,
    snapshot: PredictionSnapshot | None = None,
) -> FixtureOut:
    status = match.status if match.status in (STATUS_FINISHED, STATUS_LIVE) else STATUS_SCHEDULED
    has_score = (
        status != STATUS_SCHEDULED and match.home_goals is not None and match.away_goals is not None
    )
    score = Score(home=match.home_goals, away=match.away_goals) if has_score else None
    odds = _odds(match)
    day = pd.Timestamp(match.date)
    prediction, frozen_at = None, None
    if status != STATUS_SCHEDULED and snapshot is not None:
        prediction = frozen(snapshot)
        frozen_at = snapshot.updated_at.replace(tzinfo=dt.UTC) if prediction is not None else None
    if prediction is None:
        prediction = predict_match(models, match.home, match.away, match.league, day, odds, stage=match.stage)
    return FixtureOut(
        id=match.id,
        league=match.league,
        date=match.date,
        time=match.time,
        kickoff=_kickoff(match),
        home=match.home,
        away=match.away,
        status=status,
        score=score,
        odds=Odds(home=odds[0], draw=odds[1], away=odds[2]) if odds else None,
        prediction=prediction,
        prediction_frozen_at=frozen_at,
        competition_type="europe" if match.league in EUROPEAN_LEAGUES else "domestic",
        stage=match.stage,
        featured=is_big_match(models, match) if featured is None else featured,
        live=_live_info(match, score, now),
    )


@router.get("/fixtures", response_model=list[FixtureOut])
def fixtures(
    request: Request,
    session: SessionDep,
    models: ModelsDep,
    today: TodayDep,
    date: dt.date | None = None,
    date_to: dt.date | None = None,
    league: LeagueQuery = None,
    team: Annotated[str | None, Query(min_length=1, max_length=64, pattern=TEAM_NAME)] = None,
) -> list[FixtureOut]:
    start = date or today
    end = date_to or start
    if end < start:
        raise APIError(422, "INVALID_DATE_RANGE", "date_to must not be before date")
    if (end - start).days > MAX_RANGE_DAYS:
        raise APIError(
            422, "DATE_RANGE_TOO_WIDE", f"date_to must be at most {MAX_RANGE_DAYS} days after date"
        )
    _check_league(league)
    matches = repository.fixtures_between(session, start, end, league=league, team=team)
    return _fixtures_out(request, session, models, matches)


def _fixtures_out(
    request: Request, session: Session, models: Models, matches: list[Match], featured: bool | None = None
) -> list[FixtureOut]:
    """Fixture items; live/finished use frozen snapshots, pre-kickoff ones refresh their snapshot."""
    now = request.app.state.now()
    snapshots = repository.snapshots_for(session, [m.id for m in matches]) if matches else {}
    refresh = request.app.state.settings.snapshot_refresh_minutes
    out, pending = [], []
    for m in matches:
        item = _fixture_out(models, m, now, featured=featured, snapshot=snapshots.get(m.id))
        pre = item.prediction is not None and is_pre_kickoff(m, now)
        if pre and needs_refresh(snapshots.get(m.id), item.prediction, now, refresh):
            pending.append(snapshot_row(m, item.prediction))
        out.append(item)
    write_snapshots(session, pending, now)
    return out


@router.get("/featured", response_model=list[FixtureOut])
def featured(
    request: Request,
    session: SessionDep,
    models: ModelsDep,
    today: TodayDep,
    date: dt.date | None = None,
) -> list[FixtureOut]:
    """Live matches, then UEFA matches of the day, then the strongest domestic pairings (max 12)."""
    day = date or today
    live = [m for m in repository.live_matches(session) if abs((m.date - day).days) <= 1]
    ranked = rank_featured(models, live, repository.fixtures_between(session, day, day))
    return _fixtures_out(request, session, models, ranked, featured=True)


@router.get("/predict", response_model=PredictResponse)
def predict(
    holder: HolderDep,
    models: ModelsDep,
    today: TodayDep,
    home: TeamQuery,
    away: TeamQuery,
    league: LeagueQuery = None,
) -> PredictResponse:
    _check_league(league)
    if home.casefold() == away.casefold():
        raise APIError(422, "SAME_TEAM", "home and away must be different teams")
    europe = models.europe
    if league in EUROPEAN_LEAGUES:
        return _predict_europe(europe, home, away, league, today)
    _require_model(holder)
    known = holder.service.known_teams()  # type: ignore[union-attr]
    resolved = {}
    for side, name in (("home", home), ("away", away)):
        found = _resolve(name, known)
        if found is None:
            if europe.loaded:  # e.g. a club outside the domestic model's leagues
                return _predict_europe(europe, home, away, "UCL", today)
            raise APIError(404, "TEAM_NOT_FOUND", f"unknown team '{name}'")
        resolved[side] = found
    if league is None and europe.loaded and cross_country(holder, resolved["home"], resolved["away"]):
        return _predict_europe(europe, resolved["home"], resolved["away"], "UCL", today)
    lg = league or known[resolved["home"]]
    pred = holder.predict(resolved["home"], resolved["away"], lg, pd.Timestamp(today))
    return PredictResponse(home=resolved["home"], away=resolved["away"], league=lg, **pred)


def _predict_europe(
    europe: ServiceHolder, home: str, away: str, league: str, today: dt.date
) -> PredictResponse:
    service = europe.service
    if service is None:
        raise APIError(503, "MODEL_NOT_LOADED", "the European model is not loaded yet")
    resolved = {}
    for side, name in (("home", home), ("away", away)):
        found = service.resolve(name)  # type: ignore[attr-defined]
        if found is None:
            raise APIError(404, "TEAM_NOT_FOUND", f"unknown team '{name}'")
        resolved[side] = found
    if resolved["home"] == resolved["away"]:
        raise APIError(422, "SAME_TEAM", "home and away must be different teams")
    pred = europe.predict(resolved["home"], resolved["away"], league, pd.Timestamp(today))
    return PredictResponse(home=resolved["home"], away=resolved["away"], league=league, **pred)


def _form_row(team: str, match: Match) -> FormResult:
    at_home = match.home == team
    gf, ga = (match.home_goals, match.away_goals) if at_home else (match.away_goals, match.home_goals)
    result = "W" if gf > ga else "D" if gf == ga else "L"
    return FormResult(
        date=match.date,
        league=match.league,
        home=match.home,
        away=match.away,
        home_goals=match.home_goals,
        away_goals=match.away_goals,
        venue="H" if at_home else "A",
        opponent=match.away if at_home else match.home,
        goals_for=gf,
        goals_against=ga,
        result=result,
    )


@router.get("/teams/{name}/form", response_model=TeamForm)
def team_form(
    session: SessionDep,
    holder: HolderDep,
    name: Annotated[str, Path(min_length=1, max_length=64, pattern=TEAM_NAME)],
) -> TeamForm:
    service = holder.service
    known = service.known_teams() if service is not None else {}
    team = _resolve(name, known) or name
    in_db = repository.team_exists(session, team)
    if not in_db and team not in known:
        raise APIError(404, "TEAM_NOT_FOUND", f"unknown team '{name}'")
    results = repository.recent_results(session, team, FORM_RESULTS)
    ratings = None
    if service is not None:
        try:
            ratings = {k: float(v) for k, v in service.team_ratings(team).items()}
        except KeyError:
            ratings = None
    league = known.get(team) or (results[0].league if results else None)
    return TeamForm(
        team=team, league=league, results=[_form_row(team, m) for m in results], ratings=ratings
    )
