"""SQLAlchemy 2.0 ORM models."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

STATUS_FINISHED = "finished"
STATUS_SCHEDULED = "scheduled"
STATUS_LIVE = "live"  # in play / half time; home_goals/away_goals hold the current score
STATUSES = (STATUS_SCHEDULED, STATUS_LIVE, STATUS_FINISHED)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("league", "date", "home", "away", name="uq_match"),
        Index("ix_matches_date_league", "date", "league"),
        Index("ix_matches_home_date", "home", "date"),
        Index("ix_matches_away_date", "away", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league: Mapped[str] = mapped_column(String(8))
    date: Mapped[dt.date] = mapped_column(Date)
    time: Mapped[str | None] = mapped_column(String(5))
    home: Mapped[str] = mapped_column(String(64))
    away: Mapped[str] = mapped_column(String(64))
    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    home_shots: Mapped[int | None] = mapped_column(Integer)
    away_shots: Mapped[int | None] = mapped_column(Integer)
    home_target: Mapped[int | None] = mapped_column(Integer)
    away_target: Mapped[int | None] = mapped_column(Integer)
    home_corners: Mapped[int | None] = mapped_column(Integer)
    away_corners: Mapped[int | None] = mapped_column(Integer)
    home_xg: Mapped[float | None] = mapped_column(Float)
    away_xg: Mapped[float | None] = mapped_column(Float)
    odds_home: Mapped[float | None] = mapped_column(Float)
    odds_draw: Mapped[float | None] = mapped_column(Float)
    odds_away: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(10), default=STATUS_FINISHED, index=True)
    source: Mapped[str] = mapped_column(String(32), default="history")
    # Added for UEFA competitions / live scores (nullable; see db.session.migrate).
    minute: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    unmapped: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class PredictionSnapshot(Base):
    """Pre-kickoff prediction of a fixture, frozen at kickoff (refreshed only before it)."""

    __tablename__ = "prediction_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    model_version: Mapped[str] = mapped_column(String(40))
    prob_home: Mapped[float] = mapped_column(Float)
    prob_draw: Mapped[float] = mapped_column(Float)
    prob_away: Mapped[float] = mapped_column(Float)
    xg_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    xg_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[str] = mapped_column(Text)  # full PredictionOut JSON
    kickoff: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)  # UTC, naive
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(10), default="running")  # running|success|partial|failed
    results_rows: Mapped[int] = mapped_column(Integer, default=0)
    fixtures_rows: Mapped[int] = mapped_column(Integer, default=0)
    inserted: Mapped[int] = mapped_column(Integer, default=0)
    updated: Mapped[int] = mapped_column(Integer, default=0)
    applied_results: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


INT_COLUMNS = (
    "home_goals", "away_goals", "home_shots", "away_shots",
    "home_target", "away_target", "home_corners", "away_corners",
)  # fmt: skip
FLOAT_COLUMNS = ("home_xg", "away_xg", "odds_home", "odds_draw", "odds_away")
VALUE_COLUMNS = ("time", *INT_COLUMNS, *FLOAT_COLUMNS)
# Optional frame columns (outside contracts.MATCH_COLUMNS) that ingest may provide.
EXTRA_COLUMNS = ("minute", "stage", "unmapped")
