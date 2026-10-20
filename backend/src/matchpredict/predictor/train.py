"""Walk-forward evaluation and final artifact training.

CLI: ``uv run python -m matchpredict.predictor.train [--quick] [--data PATH] [--out PATH]``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matchpredict.contracts import DEFAULT_ARTIFACT
from matchpredict.predictor.data import load_history, outcomes, season_of, season_start
from matchpredict.predictor.features import (
    BASE_FEATURES,
    CAT_FEATURES,
    MARKET_FEATURES,
    MARKET_ONLY,
    RATING_FEATURES,
    build_feature_table,
)
from matchpredict.predictor.goals import GoalsModel
from matchpredict.predictor.metrics import ece, evaluate
from matchpredict.predictor.model import OutcomeModel
from matchpredict.predictor.ratings import EngineConfig, RatingsEngine

log = logging.getLogger("matchpredict.train")

BACKEND_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA = BACKEND_DIR.parent / "data" / "raw" / "Matches.csv"
TEST_SEASONS: tuple[int, ...] = (2022, 2023, 2024, 2025)
QUICK_TEST_SEASONS: tuple[int, ...] = (2024, 2025)
QUICK_FROM = pd.Timestamp("2012-07-01")
BURN_IN_DAYS = 365  # first year of ratings is too cold to train on
CALIBRATION_DAYS = 365  # final model: calibrate on the most recent 12 months
MARKET_BASELINE = ["mkt_home", "mkt_draw", "mkt_away"]
MODEL_NAMES = ("base_rate", "elo_logit", "bookmaker", "goals", "base", "market")


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 2000
    learning_rate: float = 0.05
    market_learning_rate: float = 0.02  # residual on top of the odds needs only small steps
    depth: int = 6
    l2_leaf_reg: float = 5.0
    early_stopping_rounds: int = 100
    goals_iterations: int = 300
    engine: EngineConfig = field(default_factory=EngineConfig)


QUICK_CONFIG = TrainConfig(iterations=400, learning_rate=0.1, market_learning_rate=0.05, goals_iterations=100)


@dataclass
class Dataset:
    matches: pd.DataFrame
    table: pd.DataFrame
    y: np.ndarray
    season: np.ndarray
    engine: RatingsEngine


def prepare(matches: pd.DataFrame, config: TrainConfig) -> Dataset:
    matches = matches.reset_index(drop=True)
    table, engine = build_feature_table(matches, config=config.engine)
    return Dataset(matches, table, outcomes(matches), season_of(matches["date"]), engine)


def _outcome_model(name: str, config: TrainConfig) -> OutcomeModel:
    """"base": ratings only. "market": residual boosting on top of de-vigged odds."""
    is_market = name == "market"
    return OutcomeModel(
        feature_cols=MARKET_FEATURES if is_market else BASE_FEATURES,
        cat_cols=CAT_FEATURES,
        baseline_cols=MARKET_BASELINE if is_market else [],
        iterations=config.iterations,
        learning_rate=config.market_learning_rate if is_market else config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        early_stopping_rounds=config.early_stopping_rounds,
    )


def _goals_model(config: TrainConfig) -> GoalsModel:
    return GoalsModel(feature_cols=RATING_FEATURES, cat_cols=CAT_FEATURES, max_iter=config.goals_iterations)


def _train_mask(ds: Dataset, before: pd.Timestamp) -> np.ndarray:
    dates = ds.matches["date"]
    return ((dates >= dates.min() + pd.Timedelta(days=BURN_IN_DAYS)) & (dates < before)).to_numpy()


# ------------------------------------------------------------------ baselines
def base_rate_probs(y_train: np.ndarray, n: int) -> np.ndarray:
    freq = np.bincount(y_train, minlength=3) / len(y_train)
    return np.tile(freq, (n, 1))


def elo_logit_probs(train: pd.DataFrame, y_train: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(max_iter=500).fit(train[["elo_diff"]].to_numpy() / 100.0, y_train)
    return model.predict_proba(test[["elo_diff"]].to_numpy() / 100.0)


def bookmaker_probs(test: pd.DataFrame) -> np.ndarray:
    return test[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(dtype=float)


# --------------------------------------------------------------- walk-forward
def run_fold(ds: Dataset, test_season: int, config: TrainConfig) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Train < start(N-1), early-stop + calibrate on N-1, predict season N.

    Returns model -> probabilities on the evaluated test rows and those rows' indices.
    Only test rows with valid odds are evaluated so every model is scored on the same matches.
    """
    train = _train_mask(ds, season_start(test_season - 1))
    calib = ds.season == test_season - 1
    test_idx = np.flatnonzero((ds.season == test_season) & ds.table["mkt_home"].notna().to_numpy())
    X, y = ds.table, ds.y
    X_test = X.iloc[test_idx]
    probs: dict[str, np.ndarray] = {
        "base_rate": base_rate_probs(y[train], len(test_idx)),
        "elo_logit": elo_logit_probs(X[train], y[train], X_test),
        "bookmaker": bookmaker_probs(X_test),
    }
    started = time.perf_counter()
    goals = _goals_model(config).fit(
        X[train], ds.matches["home_goals"].to_numpy()[train], ds.matches["away_goals"].to_numpy()[train]
    )
    probs["goals"] = goals.predict_1x2(X_test)
    for name in ("base", "market"):
        model = _outcome_model(name, config).fit(X[train], y[train], X[calib], y[calib])
        probs[name] = model.predict_proba(X_test)
        log.info("  %s: %d trees, T=%.3f", name, model.best_iteration, model.temperature)
    log.info("fold %d: train=%d calib=%d test=%d (%.0fs)", test_season, train.sum(), calib.sum(),
             len(test_idx), time.perf_counter() - started)
    return probs, test_idx


def _metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float | list[float]]:
    out: dict[str, float | list[float]] = dict(evaluate(probs, y))
    out["ece"] = ece(probs, y)
    return out


def walk_forward(ds: Dataset, seasons: tuple[int, ...], config: TrainConfig) -> dict:
    per_fold: dict[str, dict] = {}
    pooled_probs: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
    pooled_idx: list[np.ndarray] = []
    last_fold: tuple[dict[str, np.ndarray], np.ndarray] | None = None
    for season in seasons:
        probs, idx = run_fold(ds, season, config)
        y = ds.y[idx]
        per_fold[f"{season}/{(season + 1) % 100:02d}"] = {name: _metrics(p, y) for name, p in probs.items()}
        for name in MODEL_NAMES:
            pooled_probs[name].append(probs[name])
        pooled_idx.append(idx)
        last_fold = (probs, idx)
    idx_all = np.concatenate(pooled_idx)
    pooled = {name: _metrics(np.vstack(p), ds.y[idx_all]) for name, p in pooled_probs.items()}
    return {"pooled": pooled, "per_fold": per_fold, "per_league_last_fold": _per_league(ds, *last_fold)}


def _per_league(ds: Dataset, probs: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, dict]:
    leagues = ds.matches["league"].to_numpy()[idx]
    y = ds.y[idx]
    out = {}
    for league in sorted(set(leagues)):
        mask = leagues == league
        out[league] = {name: evaluate(p[mask], y[mask]) for name, p in probs.items()}
    return out


# ------------------------------------------------------------- final artifact
def fit_artifact(
    matches: pd.DataFrame,
    config: TrainConfig | None = None,
    metrics: dict | None = None,
    model_version: str | None = None,
    dataset: Dataset | None = None,
) -> dict:
    """Fit the serving models on all history and return the artifact dict ModelService loads."""
    config = config or TrainConfig()
    ds = dataset or prepare(matches, config)
    dates = ds.matches["date"]
    calib_start = dates.max() - pd.Timedelta(days=CALIBRATION_DAYS)
    train = _train_mask(ds, calib_start)
    calib = (dates >= calib_start).to_numpy()
    usable = _train_mask(ds, dates.max() + pd.Timedelta(days=1))
    X, y = ds.table, ds.y
    models = {}
    for name in ("base", "market"):
        tuned = _outcome_model(name, config).fit(X[train], y[train], X[calib], y[calib])
        models[name] = tuned.refit_full(X[usable], y[usable])
        log.info("final %s: %d trees, T=%.3f", name, models[name].iterations, tuned.temperature)
    goals = _goals_model(config).fit(
        X[usable], ds.matches["home_goals"].to_numpy()[usable], ds.matches["away_goals"].to_numpy()[usable]
    )
    return {
        "model_version": model_version or pd.Timestamp.today().strftime("%Y-%m-%d"),
        "base": models["base"],
        "market": models["market"],
        "goals": goals,
        "engine": ds.engine,
        "base_features": list(BASE_FEATURES),
        "market_features": list(MARKET_FEATURES),
        "market_only": list(MARKET_ONLY),
        "last_result_date": pd.Timestamp(dates.max()),
        "trained_on": int(usable.sum()),
        "config": asdict(config),
        "metrics": metrics or {},
    }


# ----------------------------------------------------------------- reporting
def format_tables(report: dict) -> str:
    lines = ["## Pooled walk-forward results", "", _table(report["pooled"], with_ece=True), ""]
    for fold, res in report["per_fold"].items():
        lines += [f"### Test season {fold}", "", _table(res), ""]
    lines += ["## Per league, last fold (RPS)", "",
              "| League | n | " + " | ".join(MODEL_NAMES) + " |",
              "|---|---|" + "---|" * len(MODEL_NAMES)]
    for league, res in report["per_league_last_fold"].items():
        n = res["base"]["n"]
        lines.append(f"| {league} | {n} | " + " | ".join(f"{res[m]['rps']:.4f}" for m in MODEL_NAMES) + " |")
    return "\n".join(lines) + "\n"


def _table(results: dict, with_ece: bool = False) -> str:
    head = "| Model | n | RPS | Log loss | Brier | Accuracy |" + (" ECE H / D / A |" if with_ece else "")
    rows = [head, "|---|---|---|---|---|---|" + ("---|" if with_ece else "")]
    for name in MODEL_NAMES:
        m = results[name]
        row = (f"| {name} | {m['n']} | {m['rps']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} "
               f"| {m['accuracy']:.1%} |")
        if with_ece:
            row += " " + " / ".join(f"{e:.4f}" for e in m["ece"]) + " |"
        rows.append(row)
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation + final artifact")
    parser.add_argument("--quick", action="store_true", help="subset of data, fewer trees, 2 folds")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / DEFAULT_ARTIFACT)
    parser.add_argument("--skip-eval", action="store_true", help="only fit the final artifact")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    started = time.perf_counter()
    matches = load_history(args.data)
    config, seasons = (QUICK_CONFIG, QUICK_TEST_SEASONS) if args.quick else (TrainConfig(), TEST_SEASONS)
    if args.quick:
        matches = matches[matches["date"] >= QUICK_FROM]
    ds = prepare(matches, config)
    log.info("features: %d matches (%.0fs)", len(ds.matches), time.perf_counter() - started)

    report = {} if args.skip_eval else walk_forward(ds, seasons, config)
    artifact = fit_artifact(matches, config, metrics=report, dataset=ds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)
    (args.out.parent / "metrics.json").write_text(json.dumps(report, indent=2))
    if report:
        tables = format_tables(report)
        (args.out.parent / "report_tables.md").write_text(tables)
        print(tables)
    log.info("saved %s (%.1f MB) in %.0fs", args.out, args.out.stat().st_size / 1e6, time.perf_counter() - started)


if __name__ == "__main__":
    main()
