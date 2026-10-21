"""Pydantic response models (the OpenAPI contract the frontend is generated from)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel


class Probs(BaseModel):
    home: float
    draw: float
    away: float


class ExpectedGoals(BaseModel):
    home: float
    away: float


class ScoreProb(BaseModel):
    score: str
    p: float


class PredictionOut(BaseModel):
    probs: Probs
    expected_goals: ExpectedGoals
    over_2_5: float
    btts: float
    top_scores: list[ScoreProb]
    model: str
    model_version: str
    # European model only: "low" when a club is rated from a prior / has few rated games
    confidence: Literal["high", "low"] | None = None


class PredictResponse(PredictionOut):
    home: str
    away: str
    league: str


class SetupOut(BaseModel):
    """First-start progress (``ready`` once the history seed and the first ingest are done)."""

    stage: Literal["starting", "seeding_history", "seeding_uefa", "ingesting", "ready"] = "ready"
    detail: str = ""
    progress: float | None = None  # 0..1 within the stage, when known
    started_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
    first_start: bool = False


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    last_result_date: dt.date | None
    last_ingest: dt.datetime | None
    matches: int
    fixtures: int
    europe_model_loaded: bool = False
    europe_model_version: str | None = None
    football_data_org_key_configured: bool = False
    live_matches: int = 0
    model_v2_loaded: bool = False
    model_v2_version: str | None = None
    model_v2_scopes: list[str] = []  # scopes served by v2 ("domestic", "europe")
    setup: SetupOut = SetupOut()


class LeagueOut(BaseModel):
    code: str
    name: str
    country: str


class TeamOut(BaseModel):
    name: str
    league: str


class Odds(BaseModel):
    home: float
    draw: float
    away: float


class Score(BaseModel):
    home: int
    away: int


class LiveInfo(BaseModel):
    minute: int | None  # match minute (from the feed, else estimated from kick-off)
    minute_estimated: bool = False
    score: Score  # current score


class FixtureOut(BaseModel):
    id: int
    league: str
    date: dt.date
    time: str | None  # "HH:MM", UK local time as published by football-data.co.uk
    kickoff: dt.datetime | None  # timezone-aware ISO 8601 (Europe/London); null if time unknown
    home: str
    away: str
    status: Literal["scheduled", "live", "finished"]
    score: Score | None  # final score, or the current score while live
    odds: Odds | None
    prediction: PredictionOut | None
    competition_type: Literal["domestic", "europe"] = "domestic"
    stage: str | None = None  # UEFA: qual | league | ko | final
    featured: bool = False  # UEFA match, or both clubs ClubElo >= 1780
    live: LiveInfo | None = None
    # When set, ``prediction`` is the frozen pre-kickoff snapshot (UTC time of its last refresh).
    prediction_frozen_at: dt.datetime | None = None


class FormResult(BaseModel):
    date: dt.date
    league: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    venue: Literal["H", "A"]
    opponent: str
    goals_for: int
    goals_against: int
    result: Literal["W", "D", "L"]


class TeamForm(BaseModel):
    team: str
    league: str | None
    results: list[FormResult]
    ratings: dict[str, float] | None
