"""/api/v1/results — our predictions vs the actual results (live record and backtest).

``source=live`` is the honest record: pre-kickoff predictions frozen by the app
(``prediction_snapshots``). ``source=backtest`` is the walk-forward out-of-sample record
(``backtest_predictions``). ``source=all`` lists both, each row labelled with its source.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from matchpredict.api import results_stats
from matchpredict.api.errors import APIError
from matchpredict.db import results_repository as rr
from matchpredict.ingest.leagues import LEAGUES

MAX_RANGE_DAYS = 366
MAX_PAGE_SIZE = 100
UK_TZ = ZoneInfo("Europe/London")
TEAM_PATTERN = r"^[\w .'&()/\-]+$"
OutcomeCode = Literal["H", "D", "A"]

router = APIRouter(tags=["results"])


# --- schemas ---------------------------------------------------------------------


class ResultProbs(BaseModel):
    home: float
    draw: float
    away: float


class ResultScore(BaseModel):
    home: int
    away: int


class ResultItem(BaseModel):
    id: str  # "<source>-<match id>"
    match_id: int
    source: Literal["live", "backtest"]
    status: Literal["finished", "pending"]
    date: dt.date
    kickoff: dt.datetime | None  # timezone-aware (Europe/London); null if time unknown
    league: str
    home: str
    away: str
    score: ResultScore | None
    actual: OutcomeCode | None
    predicted: OutcomeCode  # argmax of probs
    probs: ResultProbs
    prob_predicted: float
    prob_of_actual: float | None
    correct: bool | None
    rps: float | None
    model: str | None  # "market" (odds used) / "base"
    model_version: str
    frozen_at: dt.datetime | None  # live only: UTC time the pre-kickoff prediction was last refreshed


class ResultsPage(BaseModel):
    items: list[ResultItem]
    total: int
    page: int
    page_size: int


class OutcomeStat(BaseModel):
    n: int
    correct: int
    accuracy: float | None


class LeagueStat(BaseModel):
    league: str
    n: int
    correct: int
    accuracy: float | None
    rps: float


class BucketStat(BaseModel):
    bucket: str  # "50-60%"
    lo: float
    hi: float
    n: int
    accuracy: float | None  # how often our favourite outcome happened
    avg_prob: float | None  # the average probability we gave it


class BookmakerStat(BaseModel):
    n: int  # matches with valid pre-match odds (de-vigged proportionally)
    accuracy: float
    rps: float
    model_accuracy: float  # our model on the same matches
    model_rps: float


class DayStat(BaseModel):
    date: dt.date
    n: int
    correct: int
    accuracy: float | None


class MonthStat(BaseModel):
    month: str  # "2025-08"
    n: int
    correct: int
    accuracy: float | None


class ResultsSummary(BaseModel):
    source: Literal["live", "backtest", "all"]
    n: int
    correct: int
    accuracy: float | None
    rps: float | None
    log_loss: float | None
    brier: float | None
    by_outcome: dict[OutcomeCode, OutcomeStat]  # keyed by the actual result
    by_predicted: dict[OutcomeCode, OutcomeStat]  # keyed by our predicted outcome
    by_league: list[LeagueStat]
    by_confidence_bucket: list[BucketStat]
    bookmaker: BookmakerStat | None
    daily: list[DayStat]  # last 60 match days
    monthly: list[MonthStat]
    first_date: dt.date | None
    last_date: dt.date | None
    pending: int  # live predictions awaiting a result (same filters)
    available: dict[str, int]  # unfiltered finished counts per source: {"live": n, "backtest": n}


# --- dependencies ----------------------------------------------------------------


def _session(request: Request) -> Any:
    with request.app.state.session_factory() as session:
        yield session


SessionDep = Annotated[Session, Depends(_session)]


def _filters(
    source: Literal["live", "backtest", "all"] = "live",
    status: Literal["finished", "pending"] = "finished",
    date: dt.date | None = None,
    date_to: dt.date | None = None,
    league: Annotated[str | None, Query(min_length=1, max_length=8, pattern=r"^[A-Z0-9]+$")] = None,
    team: Annotated[str | None, Query(min_length=1, max_length=64, pattern=TEAM_PATTERN)] = None,
    outcome: Literal["all", "correct", "wrong"] = "all",
    confidence: Literal["all", "high", "low"] = "all",
) -> rr.ResultFilters:
    if date is not None and date_to is not None:
        if date_to < date:
            raise APIError(422, "INVALID_DATE_RANGE", "date_to must not be before date")
        if source != "backtest" and (date_to - date).days > MAX_RANGE_DAYS:
            raise APIError(422, "DATE_RANGE_TOO_WIDE",
                           f"date_to must be at most {MAX_RANGE_DAYS} days after date")  # fmt: skip
    if league is not None and league != rr.UEFA and league not in LEAGUES:
        raise APIError(404, "LEAGUE_NOT_FOUND", f"unknown league '{league}'")
    return rr.ResultFilters(source=source, status=status, date_from=date, date_to=date_to, league=league,
                            team=(team.strip() or None) if team else None, outcome=outcome,
                            confidence=confidence)  # fmt: skip


FiltersDep = Annotated[rr.ResultFilters, Depends(_filters)]


# --- mapping ---------------------------------------------------------------------


def _kickoff(day: dt.date, time: str | None) -> dt.datetime | None:
    if not time:
        return None
    hh, mm = (int(p) for p in time.split(":"))
    return dt.datetime.combine(day, dt.time(hh, mm), tzinfo=UK_TZ)


def _live_model(payload: str | None) -> str | None:
    try:
        model = json.loads(payload or "{}").get("model")
    except (ValueError, AttributeError):
        return None
    return str(model) if model else None


def to_item(row: Any) -> ResultItem:
    probs = (float(row.prob_home), float(row.prob_draw), float(row.prob_away))
    predicted = max(range(3), key=lambda k: (probs[k], -k))  # ties -> earlier outcome (numpy argmax)
    finished = row.match_status == "finished" and row.home_goals is not None and row.away_goals is not None
    actual = None
    if finished:
        actual = 0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2
    frozen_at = row.frozen_at.replace(tzinfo=dt.UTC) if row.frozen_at is not None else None
    return ResultItem(
        id=f"{row.source}-{row.match_id}",
        match_id=row.match_id,
        source=row.source,
        status="finished" if finished else "pending",
        date=row.date,
        kickoff=_kickoff(row.date, row.time),
        league=row.league,
        home=row.home,
        away=row.away,
        score=ResultScore(home=row.home_goals, away=row.away_goals) if finished else None,
        actual=results_stats.OUTCOMES[actual] if actual is not None else None,
        predicted=results_stats.OUTCOMES[predicted],
        probs=ResultProbs(home=probs[0], draw=probs[1], away=probs[2]),
        prob_predicted=probs[predicted],
        prob_of_actual=probs[actual] if actual is not None else None,
        correct=predicted == actual if actual is not None else None,
        rps=results_stats.row_rps(probs, actual) if actual is not None else None,
        model=row.model if row.source == "backtest" else _live_model(row.payload),
        model_version=row.model_version,
        frozen_at=frozen_at,
    )


# --- endpoints -------------------------------------------------------------------


@router.get("/results", response_model=ResultsPage)
def results(
    session: SessionDep,
    filters: FiltersDep,
    page: Annotated[int, Query(ge=1, le=100_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 25,
) -> ResultsPage:
    rows, total = rr.list_results(session, filters, page, page_size)
    return ResultsPage(items=[to_item(r) for r in rows], total=total, page=page, page_size=page_size)


@router.get("/results/summary", response_model=ResultsSummary)
def results_summary(session: SessionDep, filters: FiltersDep) -> ResultsSummary:
    summary = results_stats.summarise(rr.metric_rows(session, filters))
    return ResultsSummary(
        source=filters.source,
        pending=rr.count_pending(session, filters),
        available=rr.available(session),
        **summary,
    )
