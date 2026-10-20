"""Hyperparameter selection for the global rating — on pre-test seasons only.

Score of a config: run the rating pass, then fit a multinomial logistic regression on the
rating expectation over 2008/09–2015/16 and measure log loss over 2016/17–2020/21,
separately for domestic and UEFA matches. The test seasons (2022/23 onwards) and the
2021/22 model-selection season are never used here.

CLI: ``uv run python -m matchpredict.predictor.v2.tune`` (parallel grid, ~3 min).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matchpredict.predictor.data import outcomes
from matchpredict.predictor.metrics import log_loss, rps
from matchpredict.predictor.v2.global_rating import (
    ClubEloBook,
    GlobalConfig,
    GlobalRating,
)

log = logging.getLogger(__name__)

FIT_FROM, FIT_TO = pd.Timestamp("2008-07-01"), pd.Timestamp("2016-07-01")
EVAL_TO = pd.Timestamp("2021-07-01")


def _design(feats: pd.DataFrame) -> np.ndarray:
    exp = feats["g_exp_home"].clip(1e-4, 1 - 1e-4).to_numpy()
    logit = np.log(exp / (1 - exp))
    return np.column_stack([logit, np.abs(logit)])


def score(frame: pd.DataFrame, countries: dict, book: ClubEloBook | None, config: GlobalConfig) -> dict:
    rating = GlobalRating(config=config, countries=dict(countries), clubelo=book)
    feats = rating.build_features(frame[frame["date"] < EVAL_TO])
    sub = frame.loc[feats.index]
    y = outcomes(sub)
    x = _design(feats)
    out: dict[str, float] = {}
    for scope, mask in (("domestic", sub["comp"] == "DOM"), ("europe", sub["comp"] != "DOM")):
        fit = (mask & (sub["date"] >= FIT_FROM) & (sub["date"] < FIT_TO)).to_numpy()
        ev = (mask & (sub["date"] >= FIT_TO)).to_numpy()
        model = LogisticRegression(max_iter=1000).fit(x[fit], y[fit])
        p = model.predict_proba(x[ev])
        out[f"{scope}_ll"] = log_loss(p, y[ev])
        out[f"{scope}_rps"] = rps(p, y[ev])
        out[f"{scope}_n"] = int(ev.sum())
    return out


def _worker(args: tuple) -> tuple[dict, dict]:
    frame_path, countries, elo_csv, overrides = args
    frame = pd.read_pickle(frame_path)
    book = ClubEloBook.from_csv(elo_csv) if elo_csv else None
    config = replace(GlobalConfig(), **overrides)
    if not config.seed_clubelo:
        book = None
    return overrides, score(frame, countries, book, config)


def grid() -> list[dict]:
    """Stage A (domestic dynamics) x Stage B (European linkage) candidates."""
    domestic = [
        {"k": 20.0, "mov": "clubelo"}, {"k": 15.0, "mov": "clubelo"}, {"k": 25.0, "mov": "clubelo"},
        {"k": 20.0, "mov": "log"}, {"k": 20.0, "mov": "none"}, {"k": 20.0, "season_regress": 0.1},
        {"k": 20.0, "seed_clubelo": False},
    ]  # fmt: skip
    europe = [
        {"k_euro": 30.0, "k_country": 3.0}, {"k_euro": 20.0, "k_country": 3.0},
        {"k_euro": 45.0, "k_country": 3.0}, {"k_euro": 30.0, "k_country": 0.0},
        {"k_euro": 30.0, "k_country": 1.5}, {"k_euro": 30.0, "k_country": 6.0},
        {"k_euro": 30.0, "k_country": 3.0, "newcomer_gap": 0.0},
        {"k_euro": 30.0, "k_country": 3.0, "newcomer_gap": 120.0},
        {"k_euro": 30.0, "k_country": 3.0, "provisional_factor": 1.0},
        {"k_euro": 30.0, "k_country": 3.0, "provisional_factor": 2.5},
    ]  # fmt: skip
    seen, out = set(), []
    for cand in [*domestic, *europe]:
        key = json.dumps(cand, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def run_grid(frame_path: str, countries: dict, elo_csv: str | None, candidates: list[dict], workers: int = 6) -> list:
    jobs = [(frame_path, countries, elo_csv, c) for c in candidates]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, jobs))


def choose(results: list[tuple[dict, dict]], base: GlobalConfig | None = None) -> GlobalConfig:
    """Best domestic overrides (domestic log loss) + best European overrides (UEFA log loss)."""
    base = base or GlobalConfig()
    dom_keys = {"k", "mov", "season_regress", "seed_clubelo"}
    dom = [(r["domestic_ll"], o) for o, r in results if set(o) <= dom_keys]
    euro = [(r["europe_ll"], o) for o, r in results if not set(o) & (dom_keys - {"k"}) and "k_euro" in o]
    best_dom = min(dom, key=lambda t: t[0])[1] if dom else {}
    best_euro = min(euro, key=lambda t: t[0])[1] if euro else {}
    return replace(base, **{**best_dom, **best_euro})


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    import tempfile
    from pathlib import Path

    from matchpredict.config import REPO_DIR, get_settings
    from matchpredict.predictor.v2.data import load_matches

    parser = argparse.ArgumentParser(description="Tune the v2 global rating on pre-test seasons")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[4] / "artifacts" / "v2_tuning.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = get_settings()
    elo_csv = REPO_DIR / "data" / "raw" / "EloRatings.csv"
    frame, names = load_matches(settings.database_url, settings.openfootball_dir, elo_csv)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "frame.pkl")
        frame.to_pickle(path)
        results = run_grid(path, names.countries, str(elo_csv), grid(), args.workers)
    for overrides, res in sorted(results, key=lambda t: t[1]["domestic_ll"]):
        log.info("%s -> %s", overrides, {k: round(v, 5) if isinstance(v, float) else v for k, v in res.items()})
    chosen = choose(results)
    payload = {"results": [{"overrides": o, **r} for o, r in results], "chosen": asdict(chosen)}
    args.out.write_text(json.dumps(payload, indent=2, default=lambda v: None if isinstance(v, float) and math.isnan(v) else v))
    log.info("chosen config: %s", asdict(chosen))


if __name__ == "__main__":  # pragma: no cover
    main()
