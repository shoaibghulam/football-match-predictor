"""Rating hyperparameter selection uses only pre-test seasons."""

from __future__ import annotations

from v2_kit import synthetic_frame

from matchpredict.predictor.v2 import tune
from matchpredict.predictor.v2.data import (
    build_canonicaliser,
    prepare_frame,
    standardise,
)
from matchpredict.predictor.v2.global_rating import GlobalConfig


def test_score_worker_grid_and_choose(tmp_path):
    raw = synthetic_frame(seasons=range(2008, 2023), euro_per_season=12)
    names = build_canonicaliser(standardise(raw))
    frame = prepare_frame(raw, names)
    res = tune.score(frame, names.countries, None, GlobalConfig(seed_clubelo=False))
    assert 0.5 < res["domestic_ll"] < 1.3 and res["europe_n"] == 12 * 5
    path = tmp_path / "frame.pkl"
    frame.to_pickle(path)
    overrides, out = tune._worker((str(path), names.countries, None, {"k": 10.0, "seed_clubelo": False}))
    assert overrides == {"k": 10.0, "seed_clubelo": False} and out["domestic_n"] == res["domestic_n"]
    grid = tune.grid()
    assert len(grid) == len({str(sorted(g.items())) for g in grid}) > 10
    results = [({"k": 15.0, "mov": "log"}, {"domestic_ll": 1.0, "europe_ll": 0.99}),
               ({"k": 20.0, "mov": "clubelo"}, {"domestic_ll": 1.1, "europe_ll": 0.90}),
               ({"k_euro": 30.0, "k_country": 6.0}, {"domestic_ll": 1.2, "europe_ll": 0.95}),
               ({"k_euro": 45.0, "k_country": 0.0}, {"domestic_ll": 1.2, "europe_ll": 0.97})]  # fmt: skip
    chosen = tune.choose(results)
    assert (chosen.k, chosen.mov, chosen.k_euro, chosen.k_country) == (15.0, "log", 30.0, 6.0)
