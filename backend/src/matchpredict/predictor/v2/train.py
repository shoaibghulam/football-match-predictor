"""v2 walk-forward evaluation, ablations, promotion gate and final artifact.

CLI (from ``backend/``)::

    uv run python -m matchpredict.predictor.v2.train            # walk-forward + gate + artifact
    uv run python -m matchpredict.predictor.v2.train --select   # ablations on the selection split
    uv run python -m matchpredict.predictor.v2.train --skip-eval --out PATH  # retrain only

Protocol (as v1): test season N in 2022/23..2025/26 — train on matches before 1 Jul of N-1,
early-stop + fit the ensemble/calibration on season N-1, score season N. Choices were made on
the selection split only (train < 2020-07-01, early-stop/pool 2020/21, score 2021/22).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from matchpredict.config import BACKEND_DIR, REPO_DIR, get_settings
from matchpredict.predictor.data import outcomes, season_of, season_start
from matchpredict.predictor.goals import (
    GoalsModel,
    btts,
    outcome_probs,
    over_line,
    rescale_to_1x2,
)
from matchpredict.predictor.metrics import ece, evaluate, log_loss
from matchpredict.predictor.v2.data import load_matches
from matchpredict.predictor.v2.ensemble import LogPool, fit_pool, reliability
from matchpredict.predictor.v2.features import (
    MARKET_BASELINE,
    DomesticEngine,
    FeatureSet,
    FeatureState,
    build_table,
)
from matchpredict.predictor.v2.global_rating import (
    ClubEloBook,
    GlobalConfig,
    GlobalRating,
)
from matchpredict.predictor.v2.models import CatModel, EloLR
from matchpredict.predictor.v2.names import Canonicaliser

log = logging.getLogger("matchpredict.v2.train")

ARTIFACT_VERSION = 2
DEFAULT_OUT = BACKEND_DIR / "artifacts" / "model_v2.joblib"
ELO_CSV = REPO_DIR / "data" / "raw" / "EloRatings.csv"
TEST_SEASONS: tuple[int, ...] = (2022, 2023, 2024, 2025)
SELECT_SEASON = 2021
BURN_IN_DAYS = 365
CALIBRATION_DAYS = 365
TOP_LEAGUES = 10
V1_METRICS = BACKEND_DIR / "artifacts" / "metrics.json"
V1_EUROPE_METRICS = BACKEND_DIR / "artifacts" / "europe_metrics.json"
MEMBERS = ("base", "goals", "elo")


@dataclass(frozen=True)
class TrainConfig:
    rating: GlobalConfig = field(default_factory=GlobalConfig)
    # chosen on the selection split: the ClubElo as-of block did not help (and is stale at serving)
    features: FeatureSet = field(default_factory=lambda: FeatureSet(clubelo=False))
    iterations: int = 3000
    learning_rate: float = 0.06
    market_learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 5.0
    early_stopping_rounds: int = 100
    goals_iterations: int = 300
    euro_weight: float = 1.0  # selection: 1 beat 3 and 8 on both scopes
    members: tuple[str, ...] = MEMBERS


@dataclass
class Dataset:
    matches: pd.DataFrame
    table: pd.DataFrame
    y: np.ndarray
    season: np.ndarray
    state: FeatureState
    names: Canonicaliser

    @property
    def euro(self) -> np.ndarray:
        return self.table["is_euro"].to_numpy() > 0

    @property
    def has_odds(self) -> np.ndarray:
        return self.table["mkt_home"].notna().to_numpy()


def new_state(names: Canonicaliser, config: TrainConfig, book: ClubEloBook | None) -> FeatureState:
    rating_book = book if config.rating.seed_clubelo else None
    rating = GlobalRating(config=config.rating, countries=dict(names.countries), clubelo=rating_book)
    return FeatureState(rating=rating, domestic=DomesticEngine(), clubelo=book)


def prepare(frame: pd.DataFrame, names: Canonicaliser, config: TrainConfig, book: ClubEloBook | None) -> Dataset:
    frame = frame.reset_index(drop=True)
    started = time.perf_counter()
    state = new_state(names, config, book)
    table = build_table(frame, state)
    log.info("feature table: %d rows x %d cols (%.0fs)", *table.shape, time.perf_counter() - started)
    return Dataset(frame, table, outcomes(frame), season_of(frame["date"]), state, names)


# ------------------------------------------------------------------ members
def _cat(config: TrainConfig, market: bool, features: FeatureSet | None = None) -> CatModel:
    fs = features or config.features
    return CatModel(
        feature_cols=fs.columns(market=market),
        cat_cols=fs.cat_columns(),
        baseline_cols=list(MARKET_BASELINE) if market else [],
        iterations=config.iterations,
        learning_rate=config.market_learning_rate if market else config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        early_stopping_rounds=config.early_stopping_rounds,
    )


def goals_columns(config: TrainConfig) -> list[str]:
    return [c for c in config.features.columns() if c not in ("league", "stage")]


def _goals(config: TrainConfig) -> GoalsModel:
    return GoalsModel(feature_cols=goals_columns(config), cat_cols=["league"], max_iter=config.goals_iterations)


def _weights(ds: Dataset, mask: np.ndarray, config: TrainConfig) -> np.ndarray:
    return np.where(ds.euro[mask], config.euro_weight, 1.0)


@dataclass
class Members:
    base: CatModel
    market: CatModel
    goals: GoalsModel
    elo: EloLR


def fit_members(ds: Dataset, train: np.ndarray, calib: np.ndarray, config: TrainConfig) -> Members:
    X, y = ds.table, ds.y
    t0 = time.perf_counter()
    base = _cat(config, market=False).fit(
        X[train], y[train], X[calib], y[calib], _weights(ds, train, config), _weights(ds, calib, config)
    )
    t1 = time.perf_counter()
    dom_train, dom_calib = train & ~ds.euro, calib & ~ds.euro
    market = _cat(config, market=True).fit(X[dom_train], y[dom_train], X[dom_calib], y[dom_calib])
    t2 = time.perf_counter()
    hg, ag = ds.matches["home_goals"].to_numpy(), ds.matches["away_goals"].to_numpy()
    goals = _goals(config).fit(X[train], hg[train], ag[train])
    elo = EloLR().fit(X[train], y[train], _weights(ds, train, config))
    log.info("  members: base %d trees (%.0fs), market %d trees (%.0fs), goals+elo %.0fs",
             base.best_iteration, t1 - t0, market.best_iteration, t2 - t1, time.perf_counter() - t2)
    return Members(base, market, goals, elo)


def member_probs(members: Members, X: pd.DataFrame, names: tuple[str, ...] = MEMBERS) -> dict[str, np.ndarray]:
    out = {"base": members.base.predict_proba(X), "goals": members.goals.predict_1x2(X), "elo": members.elo.predict_proba(X)}
    return {k: out[k] for k in names}


def _with_base(names: tuple[str, ...]) -> tuple[str, ...]:
    return names if "base" in names else (*names, "base")  # the market pool always uses base


@dataclass
class Pools:
    domestic: LogPool
    europe: LogPool
    market: LogPool

    def describe(self) -> dict:
        return {"domestic": self.domestic.describe(), "europe": self.europe.describe(), "market": self.market.describe()}


def fit_pools(ds: Dataset, members: Members, calib: np.ndarray, config: TrainConfig) -> Pools:
    X = ds.table[calib]
    y = ds.y[calib]
    euro = ds.euro[calib]
    odds = ds.has_odds[calib]
    probs = member_probs(members, X, _with_base(config.members))
    names = list(config.members)
    dom = fit_pool(names, [probs[n][~euro] for n in names], y[~euro])
    eur = fit_pool(names, [probs[n][euro] for n in names], y[euro]) if euro.sum() >= 50 else dom
    mk = members.market.predict_proba(X[odds & ~euro])
    market = fit_pool(["market", "base"], [mk, probs["base"][odds & ~euro]], y[odds & ~euro])
    return Pools(dom, eur, market)


def ensemble_probs(members: Members, pools: Pools, X: pd.DataFrame, config: TrainConfig) -> dict[str, np.ndarray]:
    """Served probabilities: v2_base (no odds), v2_market (odds rows; NaN elsewhere)."""
    probs = member_probs(members, X, _with_base(config.members))
    names = list(config.members)
    euro = X["is_euro"].to_numpy() > 0
    stacked = [probs[n] for n in names]
    base = np.where(euro[:, None], pools.europe.predict(stacked), pools.domestic.predict(stacked))
    odds = X["mkt_home"].notna().to_numpy() & ~euro
    market = np.full_like(base, np.nan)
    if odds.any():
        mk = members.market.predict_proba(X[odds])
        market[odds] = pools.market.predict([mk, probs["base"][odds]])
    return {**probs, "v2_base": base, "v2_market": market}


# --------------------------------------------------------------- baselines
def base_rate_probs(y_train: np.ndarray, n: int) -> np.ndarray:
    return np.tile(np.bincount(y_train, minlength=3) / len(y_train), (n, 1))


# ------------------------------------------------------------- walk-forward
def _train_mask(ds: Dataset, before: pd.Timestamp) -> np.ndarray:
    dates = ds.matches["date"]
    return ((dates >= dates.min() + pd.Timedelta(days=BURN_IN_DAYS)) & (dates < before)).to_numpy()


@dataclass
class FoldResult:
    season: int
    idx: np.ndarray
    probs: dict[str, np.ndarray]
    goal_markets: dict[str, np.ndarray]
    pools: dict
    trees: dict


def _goal_markets(members: Members, X: pd.DataFrame, final: np.ndarray) -> dict[str, np.ndarray]:
    matrices = members.goals.score_matrices(X)
    over = np.empty(len(X))
    both = np.empty(len(X))
    for i, m in enumerate(matrices):
        scaled = rescale_to_1x2(m, final[i])
        over[i], both[i] = over_line(scaled, 2.5), btts(scaled)
    raw = outcome_probs(matrices)
    return {"over_2_5": over, "btts": both, "goals_raw_home": raw[:, 0]}


def run_fold(ds: Dataset, test_season: int, config: TrainConfig) -> FoldResult:
    started = time.perf_counter()
    train = _train_mask(ds, season_start(test_season - 1))
    calib = ds.season == test_season - 1
    test = ds.season == test_season
    idx = np.flatnonzero(test)
    members = fit_members(ds, train, calib, config)
    pools = fit_pools(ds, members, calib, config)
    X = ds.table.iloc[idx]
    probs = ensemble_probs(members, pools, X, config)
    probs["base_rate"] = base_rate_probs(ds.y[train], len(idx))
    probs["bookmaker"] = X[MARKET_BASELINE].to_numpy(dtype=float)
    served = np.where(np.isfinite(probs["v2_market"]).all(axis=1, keepdims=True), probs["v2_market"], probs["v2_base"])
    probs["v2_served"] = served
    markets = _goal_markets(members, X, served)
    log.info("fold %d: train=%d calib=%d test=%d (%.0fs) pools=%s", test_season, train.sum(), calib.sum(),
             len(idx), time.perf_counter() - started, json.dumps(pools.describe()))
    trees = {"base": members.base.best_iteration, "market": members.market.best_iteration}
    return FoldResult(test_season, idx, probs, markets, pools.describe(), trees)


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    out: dict = dict(evaluate(p, y))
    out["ece"] = ece(p, y)
    return out


SCOPE_MODELS = {
    "domestic_odds": ("base_rate", "elo", "goals", "base", "bookmaker", "v2_base", "v2_market"),
    "domestic_all": ("base_rate", "elo", "goals", "base", "v2_base"),
    "europe": ("base_rate", "elo", "goals", "base", "v2_base"),
    "all": ("base_rate", "elo", "v2_base", "v2_served"),
}


def scope_masks(ds: Dataset, idx: np.ndarray) -> dict[str, np.ndarray]:
    euro = ds.euro[idx]
    odds = ds.has_odds[idx]
    source = ds.matches["source"].to_numpy()[idx]
    return {
        "domestic_odds": ~euro & odds,
        "domestic_all": ~euro,
        "europe": euro,
        "all": np.ones(len(idx), bool),
        # rows v1 was evaluated on (Matches.csv / openfootball) for like-for-like comparisons
        "domestic_odds_v1rows": ~euro & odds & (source == "history"),
        "europe_v1rows": euro & (source == "openfootball"),
    }


def summarise(ds: Dataset, folds: list[FoldResult]) -> dict:
    idx = np.concatenate([f.idx for f in folds])
    probs = {k: np.vstack([f.probs[k] for f in folds]) for k in folds[0].probs}
    markets = {k: np.concatenate([f.goal_markets[k] for f in folds]) for k in folds[0].goal_markets}
    y = ds.y[idx]
    masks = scope_masks(ds, idx)
    report: dict = {"pooled": {}, "per_fold": {}, "pools": {}, "trees": {}}
    for scope, mask in masks.items():
        models = SCOPE_MODELS.get(scope.replace("_v1rows", ""), ())
        report["pooled"][scope] = {m: _metrics(probs[m][mask], y[mask]) for m in models}
    for f in folds:
        fm = scope_masks(ds, f.idx)
        fy = ds.y[f.idx]
        label = f"{f.season}/{(f.season + 1) % 100:02d}"
        report["per_fold"][label] = {
            scope: {m: evaluate(f.probs[m][fm[scope]], fy[fm[scope]]) for m in SCOPE_MODELS[scope]}
            for scope in ("domestic_odds", "europe")
        }
        report["pools"][label] = f.pools
        report["trees"][label] = f.trees
    report["per_league"] = _per_league(ds, idx, probs, masks["domestic_odds"])
    report["per_competition"] = _per_competition(ds, idx, probs, masks["europe"])
    report["reliability"] = {
        "domestic_v2_served": reliability(probs["v2_served"][~ds.euro[idx]], y[~ds.euro[idx]]),
        "europe_v2_base": reliability(probs["v2_base"][masks["europe"]], y[masks["europe"]]),
    }
    report["goal_markets"] = _goal_market_metrics(ds, idx, markets)
    return report


def _per_league(ds: Dataset, idx: np.ndarray, probs: dict, mask: np.ndarray) -> dict:
    leagues = ds.matches["league"].to_numpy()[idx]
    y = ds.y[idx]
    counts = pd.Series(leagues[mask]).value_counts()
    out = {}
    for league in counts.index[:TOP_LEAGUES]:
        m = mask & (leagues == league)
        out[league] = {k: evaluate(probs[k][m], y[m]) for k in ("elo", "bookmaker", "v2_base", "v2_market")}
    return out


def _per_competition(ds: Dataset, idx: np.ndarray, probs: dict, mask: np.ndarray) -> dict:
    comp = ds.matches["comp"].to_numpy()[idx]
    stage = ds.matches["stage"].to_numpy()[idx]
    y = ds.y[idx]
    out = {}
    groups = {c: comp == c for c in ("UCL", "UEL", "UECL")}
    groups["qualifiers"] = stage == "qual"
    groups["main draw"] = np.isin(stage, ("league", "ko", "final"))
    for name, g in groups.items():
        m = mask & g
        if m.sum():
            out[name] = {k: evaluate(probs[k][m], y[m]) for k in ("base_rate", "elo", "goals", "base", "v2_base")}
    return out


def _goal_market_metrics(ds: Dataset, idx: np.ndarray, markets: dict) -> dict:
    total = (ds.matches["home_goals"] + ds.matches["away_goals"]).to_numpy()[idx]
    both = ((ds.matches["home_goals"] > 0) & (ds.matches["away_goals"] > 0)).to_numpy()[idx]
    out = {}
    for name, p, hit in (("over_2_5", markets["over_2_5"], total > 2.5), ("btts", markets["btts"], both)):
        hit = hit.astype(float)
        const = np.full_like(p, hit.mean())
        out[name] = {
            "n": len(p),
            "brier": float(((p - hit) ** 2).mean()),
            "brier_constant": float(((const - hit) ** 2).mean()),
            "log_loss": float(-(hit * np.log(np.clip(p, 1e-9, 1)) + (1 - hit) * np.log(np.clip(1 - p, 1e-9, 1))).mean()),
            "mean_pred": float(p.mean()),
            "observed": float(hit.mean()),
        }
    return out


def walk_forward(ds: Dataset, config: TrainConfig, seasons: tuple[int, ...] = TEST_SEASONS) -> dict:
    folds = [run_fold(ds, s, config) for s in seasons]
    return summarise(ds, folds)


# -------------------------------------------------------------- selection
def selection_experiments(ds: Dataset, config: TrainConfig) -> dict:
    """Ablations on the selection split (train < 2020-07-01, calib 2020/21, score 2021/22)."""
    train = _train_mask(ds, season_start(SELECT_SEASON - 1))
    calib = ds.season == SELECT_SEASON - 1
    test = ds.season == SELECT_SEASON
    X, y = ds.table, ds.y
    dom_test, eur_test = test & ~ds.euro & ds.has_odds, test & ds.euro
    out: dict = {"n_domestic": int(dom_test.sum()), "n_europe": int(eur_test.sum())}

    def score(p_all: np.ndarray) -> dict:
        p = pd.DataFrame(p_all, index=np.flatnonzero(test))
        d = p.loc[np.flatnonzero(dom_test)].to_numpy()
        e = p.loc[np.flatnonzero(eur_test)].to_numpy()
        return {"dom_ll": log_loss(d, y[dom_test]), "dom_rps": evaluate(d, y[dom_test])["rps"],
                "eur_ll": log_loss(e, y[eur_test]), "eur_rps": evaluate(e, y[eur_test])["rps"]}  # fmt: skip

    full = FeatureSet()
    variants = {
        "v1 features only": replace(full, global_elo=False, global_goals=False, clubelo=False, context=False),
        "+ global elo": replace(full, global_goals=False, clubelo=False, context=False),
        "+ global goals": replace(full, clubelo=False, context=False),
        "+ context (no ClubElo)": replace(full, clubelo=False),
        "+ ClubElo (full)": full,
        "global only (no v1 block)": replace(full, v1=False, clubelo=False),
    }
    for name, fs in variants.items():
        started = time.perf_counter()
        cfg = replace(config, features=fs)
        model = _cat(cfg, market=False).fit(X[train], y[train], X[calib], y[calib],
                                            _weights(ds, train, cfg), _weights(ds, calib, cfg))  # fmt: skip
        pool = fit_pool(["m"], [model.predict_proba(X[calib])], y[calib])
        out[name] = {**score(pool.predict([model.predict_proba(X[test])])), "trees": model.best_iteration}
        log.info("select %-28s %s (%.0fs)", name, out[name], time.perf_counter() - started)
    for w in (3.0, 8.0):
        cfg = replace(config, euro_weight=w)
        model = _cat(cfg, market=False).fit(X[train], y[train], X[calib], y[calib],
                                            _weights(ds, train, cfg), _weights(ds, calib, cfg))  # fmt: skip
        pool = fit_pool(["m"], [model.predict_proba(X[calib])], y[calib])
        out[f"full, euro weight {w:g}"] = score(pool.predict([model.predict_proba(X[test])]))
        log.info("select euro weight %g %s", w, out[f"full, euro weight {w:g}"])
    members = fit_members(ds, train, calib, config)
    pools = fit_pools(ds, members, calib, config)
    ens = ensemble_probs(members, pools, X[test], config)
    for name in ("base", "goals", "elo", "v2_base"):
        out[f"member {name}"] = score(ens[name])
    single = replace(config, members=("base",))
    pools_single = fit_pools(ds, members, calib, single)
    out["pool base only"] = score(ensemble_probs(members, pools_single, X[test], single)["v2_base"])
    for names in (("base", "goals"), ("goals", "elo"), ("base", "elo")):
        sub = replace(config, members=names)
        out[f"pool {'+'.join(names)}"] = score(ensemble_probs(members, fit_pools(ds, members, calib, sub), X[test], sub)["v2_base"])
    mk = ens["v2_market"]
    dom_rows = np.flatnonzero(dom_test)
    pos = np.searchsorted(np.flatnonzero(test), dom_rows)
    out["market ensemble (dom)"] = {"dom_ll": log_loss(mk[pos], y[dom_test]), "dom_rps": evaluate(mk[pos], y[dom_test])["rps"]}
    book = X.loc[dom_test, MARKET_BASELINE].to_numpy(dtype=float)
    out["bookmaker (dom)"] = {"dom_ll": log_loss(book, y[dom_test]), "dom_rps": evaluate(book, y[dom_test])["rps"]}
    out["pools"] = pools.describe()
    return out


# ---------------------------------------------------------- final artifact
def league_last_dates(matches: pd.DataFrame) -> dict[str, pd.Timestamp]:
    return {lg: pd.Timestamp(d) for lg, d in matches.groupby("league")["date"].max().items()}


APPLIED_KEY_DAYS = 30


def applied_keys(matches: pd.DataFrame, days: int = APPLIED_KEY_DAYS) -> list[tuple[str, str, str, str]]:
    """(league, date, home, away) of the matches in each league's last ``days`` days (already in the
    ratings), so the service never re-applies them when the daily ingest sends them again."""
    last = matches.groupby("league")["date"].transform("max")
    recent = matches[matches["date"] > last - pd.Timedelta(days=days)]
    return [(str(lg), str(pd.Timestamp(d).date()), str(h), str(a))
            for lg, d, h, a in zip(recent["league"], recent["date"], recent["home"], recent["away"], strict=True)]  # fmt: skip


def fit_artifact(ds: Dataset, config: TrainConfig, metrics: dict | None = None, gate: dict | None = None,
                 model_version: str | None = None) -> dict:  # fmt: skip
    dates = ds.matches["date"]
    calib_start = dates.max() - pd.Timedelta(days=CALIBRATION_DAYS)
    train = _train_mask(ds, calib_start)
    calib = (dates >= calib_start).to_numpy()
    usable = _train_mask(ds, dates.max() + pd.Timedelta(days=1))
    tuned = fit_members(ds, train, calib, config)
    pools = fit_pools(ds, tuned, calib, config)
    validation = _validation(ds, tuned, pools, calib, config)
    X, y = ds.table, ds.y
    hg, ag = ds.matches["home_goals"].to_numpy(), ds.matches["away_goals"].to_numpy()
    dom_usable = usable & ~ds.euro
    final = Members(
        base=tuned.base.refit_full(X[usable], y[usable], _weights(ds, usable, config)),
        market=tuned.market.refit_full(X[dom_usable], y[dom_usable]),
        goals=_goals(config).fit(X[usable], hg[usable], ag[usable]),
        elo=EloLR().fit(X[usable], y[usable], _weights(ds, usable, config)),
    )
    log.info("final: base %d trees, market %d trees; validation %s", final.base.iterations,
             final.market.iterations, validation)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": model_version or f"v2-{pd.Timestamp.today():%Y-%m-%d}",
        "state": ds.state,
        "names": ds.names,
        "members": final,
        "pools": pools,
        "config": config,
        "league_last_date": league_last_dates(ds.matches),
        "applied_keys": applied_keys(ds.matches),
        "last_result_date": pd.Timestamp(dates.max()),
        "trained_on": int(usable.sum()),
        "validation": validation,
        "gate": gate or {"domestic": False, "europe": False},
        "metrics": metrics or {},
    }


def _validation(ds: Dataset, members: Members, pools: Pools, calib: np.ndarray, config: TrainConfig) -> dict:
    X = ds.table[calib]
    y = ds.y[calib]
    probs = ensemble_probs(members, pools, X, config)
    euro = ds.euro[calib]
    odds = np.isfinite(probs["v2_market"]).all(axis=1)
    served = np.where(odds[:, None], probs["v2_market"], probs["v2_base"])
    return {
        "window_start": str(pd.Timestamp(ds.matches["date"][calib].min()).date()),
        "domestic_log_loss": log_loss(served[~euro], y[~euro]),
        "europe_log_loss": log_loss(probs["v2_base"][euro], y[euro]) if euro.any() else math.nan,
        "n_domestic": int((~euro).sum()),
        "n_europe": int(euro.sum()),
    }


# ------------------------------------------------------------------- gate
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def v1_reference() -> dict:
    """v1 pooled test-fold numbers (same seasons) from the v1 metrics files."""
    dom = _load_json(V1_METRICS).get("pooled", {})
    eur = _load_json(V1_EUROPE_METRICS).get("walk_forward", {}).get("overall", {})
    return {"domestic_base": dom.get("base"), "domestic_market": dom.get("market"),
            "bookmaker": dom.get("bookmaker"), "europe": eur.get("europe_lr_dom"),
            "europe_elo_baseline": eur.get("elo_baseline")}  # fmt: skip


def gate_decision(report: dict, v1: dict) -> dict:
    """Promote v2 per scope only when it beats v1 on the test folds (RPS and log loss)."""
    pooled = report["pooled"]
    dom = pooled["domestic_odds_v1rows"]
    eur = pooled["europe_v1rows"]
    reasons = {}

    def better(new: dict, old: dict | None) -> bool:
        return old is not None and new["rps"] < old["rps"] and new["log_loss"] < old["log_loss"]

    dom_ok = better(dom["v2_market"], v1.get("domestic_market")) and better(dom["v2_base"], v1.get("domestic_base"))
    eur_ok = better(eur["v2_base"], v1.get("europe"))
    reasons["domestic"] = {
        "v2_market": _short(dom["v2_market"]), "v1_market": _short(v1.get("domestic_market")),
        "v2_base": _short(dom["v2_base"]), "v1_base": _short(v1.get("domestic_base")),
    }
    reasons["europe"] = {"v2": _short(eur["v2_base"]), "v1": _short(v1.get("europe"))}
    return {"domestic": bool(dom_ok), "europe": bool(eur_ok), "evidence": reasons}


def _short(m: dict | None) -> dict | None:
    return None if m is None else {k: m[k] for k in ("n", "rps", "log_loss") if k in m}


# ----------------------------------------------------------------- report
def format_report(report: dict, v1: dict, gate: dict) -> str:
    lines: list[str] = []
    labels = {"domestic_odds": "Domestic (rows with odds)", "domestic_odds_v1rows": "Domestic, v1 test rows",
              "domestic_all": "Domestic (all rows)", "europe": "European (UCL/UEL/UECL)",
              "europe_v1rows": "European, v1 test rows (openfootball)", "all": "All matches"}  # fmt: skip
    for scope, label in labels.items():
        lines += [f"### {label}", "", "| Model | n | RPS | Log loss | Brier | Accuracy | ECE H / D / A |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for name, m in report["pooled"][scope].items():
            lines.append(f"| {name} | {m['n']} | {m['rps']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
                         f"{m['accuracy']:.1%} | {' / '.join(f'{e:.4f}' for e in m['ece'])} |")  # fmt: skip
        lines.append("")
    lines += ["### v1 reference (from v1 metrics files)", ""]
    for key, m in v1.items():
        if m:
            lines.append(f"- {key}: n={m['n']} RPS {m['rps']:.4f} log loss {m['log_loss']:.4f}")
    lines += ["", "### Per test season", "", "| Season | Scope | Model | n | RPS | Log loss | Accuracy |", "|---|---|---|---|---|---|---|"]
    for season, scopes in report["per_fold"].items():
        for scope, models in scopes.items():
            for name, m in models.items():
                lines.append(f"| {season} | {scope} | {name} | {m['n']} | {m['rps']:.4f} | {m['log_loss']:.4f} | {m['accuracy']:.1%} |")
    lines += ["", "### Top-10 leagues (pooled test folds, odds rows, RPS)", "",
              "| League | n | Elo-logit | Bookmaker | v2 base | v2 market |", "|---|---|---|---|---|---|"]  # fmt: skip
    for league, res in report["per_league"].items():
        lines.append(f"| {league} | {res['v2_base']['n']} | {res['elo']['rps']:.4f} | {res['bookmaker']['rps']:.4f} | "
                     f"{res['v2_base']['rps']:.4f} | {res['v2_market']['rps']:.4f} |")  # fmt: skip
    lines += ["", "### European competitions (pooled test folds, RPS / log loss)", "",
              "| Group | n | base rates | global-Elo logit | goals | CatBoost | v2 ensemble |", "|---|---|---|---|---|---|---|"]
    for name, res in report["per_competition"].items():
        cells = " | ".join(f"{res[k]['rps']:.4f} / {res[k]['log_loss']:.4f}" for k in ("base_rate", "elo", "goals", "base", "v2_base"))
        lines.append(f"| {name} | {res['v2_base']['n']} | {cells} |")
    lines += ["", f"Gate: domestic={gate['domestic']} europe={gate['europe']}", ""]
    return "\n".join(lines)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def save_artifact(artifact: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    joblib.dump(artifact, tmp, compress=3)
    tmp.replace(out)


def load_inputs(database_url: str | None = None) -> tuple[pd.DataFrame, Canonicaliser, ClubEloBook | None]:
    settings = get_settings()
    frame, names = load_matches(database_url or settings.database_url, settings.openfootball_dir, ELO_CSV)
    book = ClubEloBook.from_csv(ELO_CSV) if ELO_CSV.exists() else None
    return frame, names, book


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI (exercised by the real run)
    parser = argparse.ArgumentParser(description="MatchPredict v2: walk-forward evaluation + artifact")
    parser.add_argument("--select", action="store_true", help="run the ablations on the selection split only")
    parser.add_argument("--skip-eval", action="store_true", help="only fit the final artifact (weekly retrain)")
    parser.add_argument("--gate-from", type=Path, default=None, help="artifact whose gate decision is kept (--skip-eval)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.perf_counter()
    config = TrainConfig()
    frame, names, book = load_inputs(args.database_url)
    ds = prepare(frame, names, config, book)
    metrics_dir = args.out.parent
    if args.select:
        result = selection_experiments(ds, config)
        (metrics_dir / "v2_selection.json").write_text(json.dumps(result, indent=2, default=_json_default))
        log.info("selection done in %.0fs", time.perf_counter() - started)
        return
    report: dict = {}
    v1 = v1_reference()
    if args.skip_eval:
        previous = args.gate_from or args.out
        gate = joblib.load(previous).get("gate") if previous.exists() else {"domestic": False, "europe": False}
    else:
        report = walk_forward(ds, config)
        gate = gate_decision(report, v1)
        tables = format_report(report, v1, gate)
        (metrics_dir / "v2_report_tables.md").write_text(tables, encoding="utf-8")
        print(tables)  # CLI output
    artifact = fit_artifact(ds, config, metrics=report, gate=gate)
    save_artifact(artifact, args.out)
    payload = {"model_version": artifact["model_version"], "gate": gate, "validation": artifact["validation"],
               "config": asdict(config), "report": report, "v1_reference": v1,
               "pools_final": artifact["pools"].describe(), "seconds": round(time.perf_counter() - started)}  # fmt: skip
    metrics_name = "metrics_v2.json" if args.out.name == DEFAULT_OUT.name else f"{args.out.stem}_metrics.json"
    (metrics_dir / metrics_name).write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    log.info("saved %s (%.1f MB) in %.0fs; gate=%s", args.out, args.out.stat().st_size / 1e6,
             time.perf_counter() - started, gate)


if __name__ == "__main__":  # pragma: no cover
    # Run via the importable module so pickled classes are saved as
    # matchpredict.predictor.v2.train.<Class>, not __main__.<Class>.
    from matchpredict.predictor.v2.train import main as _module_main

    _module_main()
