"""Backtest (walk-forward, out-of-sample) predictions for the Results page.

Rows are written by ``matchpredict.predictor.backtest_export``: each finished match of a test
season N is priced by a model trained only on matches before 1 Jul of season N-1 (early stopping /
calibration on N-1). They are kept apart from ``prediction_snapshots`` (the live record) and are
always labelled "backtest" by the API.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from matchpredict.db.models import Base, utcnow


class BacktestPrediction(Base):
    __tablename__ = "backtest_predictions"
    __table_args__ = (Index("ix_backtest_predictions_season", "season"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)  # matches.id
    season: Mapped[int] = mapped_column(Integer)  # test season start year (2024 = 2024/25)
    model: Mapped[str] = mapped_column(String(16))  # "market" (odds used) or "base"
    model_version: Mapped[str] = mapped_column(String(40))  # e.g. "v2-backtest-2024/25"
    prob_home: Mapped[float] = mapped_column(Float)
    prob_draw: Mapped[float] = mapped_column(Float)
    prob_away: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
