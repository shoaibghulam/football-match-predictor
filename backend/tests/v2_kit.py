"""Synthetic two-country world for v2 tests (not collected: no ``test_`` prefix).

Country ENG (league E0, clubs A1..A10) is stronger than ESP (league SP1, clubs B1..B10).
Each season: a double round robin per league plus a few UCL matches between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.predictor.v2.global_rating import GlobalConfig
from matchpredict.predictor.v2.train import TrainConfig

TEAMS = {"E0": [f"A{i}" for i in range(1, 11)], "SP1": [f"B{i}" for i in range(1, 11)]}
STRENGTH = {**{t: 0.3 - 0.06 * i for i, t in enumerate(TEAMS["E0"])},
            **{t: 0.0 - 0.06 * i for i, t in enumerate(TEAMS["SP1"])}}  # fmt: skip

TINY = TrainConfig(iterations=40, learning_rate=0.2, market_learning_rate=0.2, early_stopping_rounds=10,
                   goals_iterations=15, euro_weight=2.0, rating=GlobalConfig())  # fmt: skip


def _row(day, league, home, away, hg, ag, stage="dom", odds=None, source="history") -> dict:
    row = dict.fromkeys(MATCH_COLUMNS, np.nan)
    row.update(date=pd.Timestamp(day), time=None, league=league, home=home, away=away,
               home_goals=float(hg), away_goals=float(ag), stage=stage, source=source)  # fmt: skip
    row["home_shots"], row["away_shots"] = 10.0 + hg, 9.0 + ag
    if odds is not None:
        row["odds_home"], row["odds_draw"], row["odds_away"] = odds
    return row


def _goals(rng: np.random.Generator, home: str, away: str, hfa: float = 0.25) -> tuple[int, int]:
    diff = STRENGTH[home] - STRENGTH[away]
    return int(rng.poisson(np.exp(0.2 + diff + hfa))), int(rng.poisson(np.exp(0.05 - diff)))


def _odds(home: str, away: str) -> tuple[float, float, float]:
    diff = STRENGTH[home] - STRENGTH[away] + 0.25
    ph = 1 / (1 + np.exp(-2.2 * diff)) * 0.74
    pd_ = 0.26
    pa = 1 - ph - pd_
    return tuple(round(1.05 / p, 2) for p in (ph, pd_, pa))  # type: ignore[return-value]


def synthetic_frame(seasons=range(2016, 2022), seed: int = 0, euro_per_season: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for season in seasons:
        start = pd.Timestamp(f"{season}-08-08")
        for league, teams in TEAMS.items():
            fixtures = [(h, a) for h in teams for a in teams if h != a]
            rng.shuffle(fixtures)
            for i, (h, a) in enumerate(fixtures):
                day = start + pd.Timedelta(days=7 * (i // 5) + (i % 2))
                hg, ag = _goals(rng, h, a)
                rows.append(_row(day, league, h, a, hg, ag, odds=_odds(h, a)))
        for j in range(euro_per_season):
            h = TEAMS["E0"][j % 4] if j % 2 == 0 else TEAMS["SP1"][j % 4]
            a = TEAMS["SP1"][(j + 1) % 4] if j % 2 == 0 else TEAMS["E0"][(j + 1) % 4]
            day = start + pd.Timedelta(days=30 + 14 * j + 3)
            hg, ag = _goals(rng, h, a, hfa=0.2)
            rows.append(_row(day, "UCL", h, a, hg, ag, stage="league", source="openfootball"))
    frame = pd.DataFrame(rows)
    frame["date"] = frame["date"].astype("datetime64[ns]")
    return frame
