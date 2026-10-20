"""Synthetic European history + a small trained EuropeService (not collected by pytest)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchpredict.predictor.europe import EuropeService
from matchpredict.predictor.europe_train import VARIANTS, _fit, replay

COVERED = {"Barcelona": ("ESP", 1970.0), "Paris SG": ("FRA", 1920.0), "Arsenal": ("ENG", 2020.0),
           "Bayern Munich": ("GER", 1970.0), "Benfica": ("POR", 1800.0), "Celtic": ("SCO", 1700.0),
           "Lens": ("FRA", 1760.0), "Club Brugge": ("BEL", 1720.0)}  # fmt: skip
UNCOVERED = {"Slavia Praha": "CZE", "FK Kairat": "KAZ", "Dinamo Zagreb": "CRO", "Shakhtar Donetsk": "UKR"}
TRUE = {**{k: v[1] for k, v in COVERED.items()}, "Slavia Praha": 1680.0, "FK Kairat": 1450.0,
        "Dinamo Zagreb": 1640.0, "Shakhtar Donetsk": 1660.0}  # fmt: skip


def snapshots() -> list[tuple[pd.Timestamp, list[tuple[str, str, float]]]]:
    clubs = [(name, c, elo) for name, (c, elo) in COVERED.items()]
    return [(pd.Timestamp("2019-07-01"), clubs), (pd.Timestamp("2026-09-01"), clubs)]


def history(n_per_season: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(TRUE)
    country = {**{k: v[0] for k, v in COVERED.items()}, **UNCOVERED}
    rows = []
    for year in range(2019, 2026):
        season = f"{year}-{(year + 1) % 100:02d}"
        for i in range(n_per_season):
            h, a = rng.choice(len(teams), size=2, replace=False)
            home, away = teams[h], teams[a]
            day = pd.Timestamp(f"{year}-09-15") + pd.Timedelta(days=int(i * 2))
            stage = "qual" if i < 10 else "league" if i < 70 else "ko"
            diff = (TRUE[home] - TRUE[away] + 60) / 400
            lam_h, lam_a = 1.4 * 10 ** (diff / 2), 1.1 * 10 ** (-diff / 2)
            rows.append({
                "date": day, "time": "20:00", "league": "UCL", "season": season, "stage": stage,
                "home_raw": home, "home_country": country[home], "away_raw": away,
                "away_country": country[away], "home_goals": float(rng.poisson(lam_h)),
                "away_goals": float(rng.poisson(lam_a)), "neutral": False, "home": home, "away": away,
            })  # fmt: skip
    return pd.DataFrame(rows)


def table_and_book():
    hist = history()
    table, book, euro_log = replay(hist, snapshots())
    table = pd.concat([table, hist[["season", "home_goals", "away_goals", "league", "stage"]]], axis=1)
    return hist, table, book, euro_log


def make_service() -> EuropeService:
    hist, table, book, euro_log = table_and_book()
    model = _fit(VARIANTS[0], table, None)
    return EuropeService(
        model=model,
        book=book,
        model_version="europe-test",
        snapshot_date=pd.Timestamp("2026-09-01"),
        last_result_date=pd.Timestamp(hist["date"].max()),
        recent_euro=euro_log[-200:],
        aliases={"FC Barcelona": "Barcelona"},
    )
