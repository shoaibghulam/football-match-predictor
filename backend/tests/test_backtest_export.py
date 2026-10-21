"""Walk-forward backtest export into ``backtest_predictions`` (synthetic v2 world)."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import func, select
from v2_kit import TINY, synthetic_frame

from matchpredict.config import Settings
from matchpredict.db import repository as repo
from matchpredict.db.models import Match
from matchpredict.db.results_models import BacktestPrediction
from matchpredict.db.session import create_session_factory
from matchpredict.predictor import backtest_export as bx
from matchpredict.predictor.v2 import train as v2_train


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'bt.db').as_posix()}"
    monkeypatch.setattr(bx, "get_settings", lambda: Settings(openfootball_dir=tmp_path / "none"))
    monkeypatch.setattr(v2_train, "ELO_CSV", tmp_path / "no-elo.csv")
    factory = create_session_factory(url)
    frame = synthetic_frame(seasons=range(2016, 2022))
    with factory() as s:
        repo.upsert_matches(s, frame[frame["league"] != "UCL"], source="history")
        repo.upsert_matches(s, frame[frame["league"] == "UCL"], source="openfootball")
    return url


def test_export_stores_one_out_of_sample_row_per_test_match(db_url):
    written = bx.export(db_url, [2021, 2030], TINY)  # 2030 has no matches -> skipped
    assert list(written) == [2021]
    factory = create_session_factory(db_url)
    with factory() as s:
        rows = s.execute(
            select(BacktestPrediction, Match).join(Match, Match.id == BacktestPrediction.match_id)
        ).all()
        n_season = s.scalar(select(func.count()).select_from(Match).where(Match.date >= "2021-07-01"))
    assert len(rows) == written[2021] == n_season
    probs = np.array([[b.prob_home, b.prob_draw, b.prob_away] for b, _ in rows])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert all(m.date.year >= 2021 for _, m in rows)  # only the test season is stored
    by_league = {(m.league, b.model) for b, m in rows}
    assert ("E0", "market") in by_league and ("UCL", "base") in by_league
    assert {b.model_version for b, _ in rows} == {"v2-backtest-2021/22"}


def test_export_replaces_a_season_instead_of_duplicating(db_url):
    first = bx.export(db_url, [2021], TINY)
    second = bx.export(db_url, [2021], TINY)
    factory = create_session_factory(db_url)
    with factory() as s:
        total = s.scalar(select(func.count()).select_from(BacktestPrediction))
    assert total == first[2021] == second[2021]


def test_labels():
    assert bx.season_label(2025) == "2025/26"
    assert bx.model_version(2009) == "v2-backtest-2009/10"
