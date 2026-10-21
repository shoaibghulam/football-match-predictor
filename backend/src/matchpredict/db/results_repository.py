"""Queries for the Results page: our predictions joined to the actual results.

Two sources are never mixed silently — every row carries ``source``:

* ``live``     — ``prediction_snapshots`` (pre-kickoff predictions frozen by the app);
* ``backtest`` — ``backtest_predictions`` (walk-forward, out-of-sample; see backtest_export).

Filters are pushed into SQL (predicted outcome / correctness / confidence are CASE expressions),
so a page is one ``UNION ALL ... LIMIT/OFFSET`` query plus one count, with no N+1.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Any, Literal

from sqlalchemy import (
    ColumnElement,
    Select,
    and_,
    case,
    false,
    func,
    literal,
    null,
    or_,
    select,
    union_all,
)
from sqlalchemy.orm import Session

from matchpredict.db.models import STATUS_FINISHED, Match, PredictionSnapshot
from matchpredict.db.results_models import BacktestPrediction
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES

Source = Literal["live", "backtest", "all"]
Status = Literal["finished", "pending"]
Outcome = Literal["all", "correct", "wrong"]
Confidence = Literal["all", "high", "low"]

HIGH_CONFIDENCE = 0.6  # "high confidence": our favourite outcome got at least 60%
UEFA = "UEFA"  # league filter meaning every European competition


@dataclass(frozen=True)
class ResultFilters:
    source: Source = "live"
    status: Status = "finished"
    date_from: dt.date | None = None
    date_to: dt.date | None = None
    league: str | None = None  # league code, or "UEFA"
    team: str | None = None  # case-insensitive substring of home or away
    outcome: Outcome = "all"
    confidence: Confidence = "all"


@dataclass(frozen=True)
class _Cols:
    """The prediction columns of one source table."""

    table: Any
    prob_home: Any
    prob_draw: Any
    prob_away: Any


def _cols(source: str) -> _Cols:
    t = PredictionSnapshot if source == "live" else BacktestPrediction
    return _Cols(t, t.prob_home, t.prob_draw, t.prob_away)


def predicted_expr(c: _Cols) -> ColumnElement[str]:
    """argmax of (home, draw, away); ties go to the earlier outcome like ``numpy.argmax``."""
    return case(
        (and_(c.prob_home >= c.prob_draw, c.prob_home >= c.prob_away), "H"),
        (c.prob_draw >= c.prob_away, "D"),
        else_="A",
    )


def top_prob_expr(c: _Cols) -> ColumnElement[float]:
    return case(
        (and_(c.prob_home >= c.prob_draw, c.prob_home >= c.prob_away), c.prob_home),
        (c.prob_draw >= c.prob_away, c.prob_draw),
        else_=c.prob_away,
    )


def actual_expr() -> ColumnElement[str]:
    return case(
        (Match.home_goals > Match.away_goals, "H"),
        (Match.home_goals == Match.away_goals, "D"),
        else_="A",
    )


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _conditions(c: _Cols, source: str, f: ResultFilters) -> list[ColumnElement[bool]]:
    finished = and_(Match.status == STATUS_FINISHED, Match.home_goals.is_not(None), Match.away_goals.is_not(None))
    conds: list[ColumnElement[bool]] = []
    if f.status == "pending":
        # only the live record has predictions for matches not yet played
        conds.append(Match.status != STATUS_FINISHED if source == "live" else false())
    else:
        conds.append(finished)
    if f.date_from is not None:
        conds.append(Match.date >= f.date_from)
    if f.date_to is not None:
        conds.append(Match.date <= f.date_to)
    if f.league == UEFA:
        conds.append(Match.league.in_(EUROPEAN_LEAGUES))
    elif f.league:
        conds.append(Match.league == f.league)
    if f.team:
        needle = f"%{_escape_like(f.team.lower())}%"
        conds.append(or_(func.lower(Match.home).like(needle, escape="\\"),
                         func.lower(Match.away).like(needle, escape="\\")))  # fmt: skip
    if f.outcome != "all" and f.status == "finished":
        hit = predicted_expr(c) == actual_expr()
        conds.append(hit if f.outcome == "correct" else ~hit)
    if f.confidence != "all":
        top = top_prob_expr(c)
        conds.append(top >= HIGH_CONFIDENCE if f.confidence == "high" else top < HIGH_CONFIDENCE)
    return conds


def _sources(f: ResultFilters) -> tuple[str, ...]:
    return ("live", "backtest") if f.source == "all" else (f.source,)


def _row_select(source: str, f: ResultFilters) -> Select:
    c = _cols(source)
    live = source == "live"
    t = c.table
    return (
        select(
            literal(source).label("source"),
            Match.id.label("match_id"),
            Match.date.label("date"),
            Match.time.label("time"),
            Match.league.label("league"),
            Match.home.label("home"),
            Match.away.label("away"),
            Match.home_goals.label("home_goals"),
            Match.away_goals.label("away_goals"),
            Match.status.label("match_status"),
            c.prob_home.label("prob_home"),
            c.prob_draw.label("prob_draw"),
            c.prob_away.label("prob_away"),
            (null() if live else t.model).label("model"),
            t.model_version.label("model_version"),
            (t.payload if live else null()).label("payload"),
            (t.updated_at if live else null()).label("frozen_at"),
        )
        .join(Match, Match.id == t.match_id)
        .where(*_conditions(c, source, f))
    )


def _union(f: ResultFilters) -> Any:
    selects = [_row_select(s, f) for s in _sources(f)]
    return (selects[0] if len(selects) == 1 else union_all(*selects)).subquery("results")


def list_results(session: Session, f: ResultFilters, page: int, page_size: int) -> tuple[list[Any], int]:
    """One page of rows (newest first; pending: soonest first) and the total row count."""
    sub = _union(f)
    total = session.scalar(select(func.count()).select_from(sub)) or 0
    if f.status == "pending":
        order = (sub.c.date.asc(), sub.c.time.asc(), sub.c.match_id.asc())
    else:
        order = (sub.c.date.desc(), sub.c.time.desc(), sub.c.source.asc(), sub.c.match_id.desc())
    rows = session.execute(select(sub).order_by(*order).limit(page_size).offset((page - 1) * page_size)).all()
    return list(rows), int(total)


def _metric_select(source: str, f: ResultFilters) -> Select:
    c = _cols(source)
    return (
        select(
            literal(source).label("source"),
            Match.date, Match.league, Match.home_goals, Match.away_goals,
            Match.odds_home, Match.odds_draw, Match.odds_away,
            c.prob_home.label("prob_home"), c.prob_draw.label("prob_draw"), c.prob_away.label("prob_away"),
        )
        .join(Match, Match.id == c.table.match_id)
        .where(*_conditions(c, source, f))
    )


def metric_rows(session: Session, f: ResultFilters) -> list[Any]:
    """Every finished row matching ``f`` with just the columns the summary needs."""
    finished = replace(f, status="finished")
    selects = [_metric_select(s, finished) for s in _sources(finished)]
    stmt = selects[0] if len(selects) == 1 else union_all(*selects)
    return list(session.execute(stmt).all())


def count_pending(session: Session, f: ResultFilters) -> int:
    if f.source == "backtest":
        return 0
    sub = _union(replace(f, status="pending", outcome="all"))
    return int(session.scalar(select(func.count()).select_from(sub)) or 0)


def available(session: Session) -> dict[str, int]:
    """Unfiltered number of finished, scored matches per source (drives the default tab)."""
    out = {}
    for source in ("live", "backtest"):
        sub = _union(ResultFilters(source=source))  # type: ignore[arg-type]
        out[source] = int(session.scalar(select(func.count()).select_from(sub)) or 0)
    return out

