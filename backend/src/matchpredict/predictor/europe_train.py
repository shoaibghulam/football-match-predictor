"""Training and walk-forward evaluation for the European model (``predictor/europe.py``).

Usage: ``uv run python -m matchpredict.predictor.europe [--no-download] [--no-db]``

1. Download / parse openfootball European matches (2011-12 …) and map team names
   to canonical (football-data / ClubElo) names.
2. Replay history chronologically: ClubElo snapshots + European results -> features.
3. Walk-forward by season (test 2022-23 … 2025-26) vs simple Elo baselines.
4. Fit on everything, bring ratings up to date with DB results, save the artifact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from matchpredict.config import BACKEND_DIR, REPO_DIR, get_settings
from matchpredict.ingest import openfootball
from matchpredict.ingest.teamnames import COUNTRY_CODES, TeamNameMapper, log_misses
from matchpredict.predictor import metrics
from matchpredict.predictor.europe import (
    FEATURES,
    EloBook,
    EloConfig,
    EuropeModel,
    EuropeService,
    MatchContext,
    elo_baseline_probs,
    feature_row,
)

log = logging.getLogger(__name__)

DEFAULT_ARTIFACT = BACKEND_DIR / "artifacts" / "europe.joblib"
DEFAULT_METRICS = BACKEND_DIR / "artifacts" / "europe_metrics.json"
OPENFOOTBALL_DIR = BACKEND_DIR / "data" / "openfootball"
ELO_CSV = REPO_DIR / "data" / "raw" / "EloRatings.csv"
MATCHES_CSV = REPO_DIR / "data" / "raw" / "Matches.csv"
TEST_SEASONS: tuple[str, ...] = ("2022-23", "2023-24", "2024-25", "2025-26")
DB_REPLAY_FROM = pd.Timestamp("2025-07-01")
LEG_WINDOW_DAYS = 35


COMPACT_FEATURES: tuple[str, ...] = (
    "elo_diff", "abs_diff", "neutral", "leg2", "agg_gd", "unc_home", "unc_away", "diff_x_unc",
)  # fmt: skip


@dataclass(frozen=True)
class Variant:
    name: str
    c: float = 0.05
    domestic_weight: float = 0.0  # 0 = European rows only
    features: tuple[str, ...] = COMPACT_FEATURES


VARIANTS: tuple[Variant, ...] = (
    Variant("europe_lr"),
    Variant("europe_lr_full", features=FEATURES),
    Variant("europe_lr_dom", domestic_weight=0.2, features=(*COMPACT_FEATURES, "domestic", "diff_x_domestic")),
)


# --- data --------------------------------------------------------------------------


def load_snapshots(path: Path) -> list[tuple[pd.Timestamp, list[tuple[str, str, float]]]]:
    elo = pd.read_csv(path, parse_dates=["date"])
    elo = elo.dropna(subset=["elo"])
    out = []
    for day, grp in elo.groupby("date", sort=True):
        out.append((pd.Timestamp(day), list(zip(grp["club"], grp["country"], grp["elo"].astype(float)))))
    return out


def candidates_from(elo_csv: Path, db_teams: Sequence[tuple[str, str]] = ()) -> dict[str, str | None]:
    """ClubElo names (with country) plus DB team names (country from their league)."""
    from matchpredict.ingest.leagues import EUROPEAN_LEAGUES, LEAGUES

    elo = pd.read_csv(elo_csv, usecols=["club", "country"])
    cands: dict[str, str | None] = dict(zip(elo["club"], elo["country"]))
    for team, league in db_teams:
        if league in EUROPEAN_LEAGUES:
            continue  # raw UEFA-only names are not canonical candidates
        country = COUNTRY_CODES.get(LEAGUES[league].country) if league in LEAGUES else None
        cands.setdefault(team, country)
    return cands


def map_history(raw: pd.DataFrame, mapper: TeamNameMapper) -> pd.DataFrame:
    """Add canonical ``home`` / ``away`` and ``home_mapped`` / ``away_mapped`` columns."""
    frame = raw.copy()
    for side in ("home", "away"):
        resolved = [mapper.resolve(n, c) for n, c in zip(frame[f"{side}_raw"], frame[f"{side}_country"])]
        frame[side] = [r[0] for r in resolved]
        frame[f"{side}_mapped"] = [r[1] for r in resolved]
    return frame


def mapping_coverage(frame: pd.DataFrame, covered_countries: set[str]) -> dict:
    from matchpredict.ingest.teamnames import source_country

    sides = []
    for side in ("home", "away"):
        part = frame[[f"{side}_raw", f"{side}_country", f"{side}_mapped"]]
        sides.append(part.set_axis(["raw", "country", "mapped"], axis=1))
    apps = pd.concat(sides)
    in_cov = apps["country"].map(source_country).isin(covered_countries)
    unique = apps.drop_duplicates(["raw", "country"])
    unique_cov = unique["country"].map(source_country).isin(covered_countries)
    misses = unique[unique_cov & ~unique["mapped"]]
    return {
        "appearances": len(apps),
        "appearances_mapped_pct": round(100 * float(apps["mapped"].mean()), 1),
        "covered_country_appearances_mapped_pct": round(100 * float(apps.loc[in_cov, "mapped"].mean()), 1),
        "unique_names": len(unique),
        "unique_mapped": int(unique["mapped"].sum()),
        "unmapped_in_covered_countries": sorted(f"{r} ({c})" for r, c in zip(misses["raw"], misses["country"])),
        "unmapped_countries": unique.loc[~unique["mapped"], "country"].value_counts().to_dict(),
    }


def outcome_codes(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))


# --- replay ------------------------------------------------------------------------


def replay(
    matches: pd.DataFrame,
    snapshots: Sequence[tuple[pd.Timestamp, list[tuple[str, str, float]]]],
    config: EloConfig | None = None,
    until: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, EloBook, list[dict]]:
    """Chronological replay -> (pre-match feature table, final book, European result log)."""
    from matchpredict.ingest.teamnames import source_country

    book = EloBook(config)
    frame = matches.sort_values(["date", "time"], na_position="first", kind="mergesort")
    snaps = list(snapshots)
    s_idx = 0
    legs: dict[tuple, dict] = {}
    rows: list[dict] = []
    log_rows: list[dict] = []
    for m in frame.itertuples(index=False):
        day = pd.Timestamp(m.date)
        while s_idx < len(snaps) and snaps[s_idx][0] < day:
            book.apply_snapshot(*snaps[s_idx])
            s_idx += 1
        rh = book.ensure(m.home, source_country(m.home_country), m.stage)
        ra = book.ensure(m.away, source_country(m.away_country), m.stage)
        pair_key = (m.league, m.season, m.stage, frozenset((m.home, m.away)))
        first = legs.get(pair_key)
        leg2 = (
            m.stage in ("qual", "ko")
            and first is not None
            and first["home"] == m.away
            and 0 < (day - first["date"]).days <= LEG_WINDOW_DAYS
        )
        agg = float(first["away_goals"] - first["home_goals"]) if leg2 else 0.0
        ctx = MatchContext(m.league, m.stage, bool(m.neutral), bool(leg2), agg)
        feats = feature_row(rh, ra, m.home not in book.covered, m.away not in book.covered, ctx)
        rows.append({**feats, "elo_home": rh, "elo_away": ra})
        hg, ag = int(m.home_goals), int(m.away_goals)
        legs[pair_key] = {"home": m.home, "away": m.away, "home_goals": hg, "away_goals": ag, "date": day}
        for team in (m.home, m.away):
            book.league.setdefault(team, m.league)
        book.update(day, m.home, m.away, hg, ag, neutral=bool(m.neutral), european=True)
        log_rows.append(
            {"date": day.isoformat(), "league": m.league, "home": m.home, "away": m.away,
             "home_goals": hg, "away_goals": ag}
        )  # fmt: skip
    for day, clubs in snaps[s_idx:]:
        if until is None or day <= until:
            book.apply_snapshot(day, clubs)
    table = pd.DataFrame(rows, index=frame.index)
    return table.reindex(matches.index), book, log_rows


def domestic_rows(path: Path) -> pd.DataFrame:
    """Matches.csv rows with ClubElo HomeElo/AwayElo as domestic training rows."""
    usecols = ["Division", "MatchDate", "HomeElo", "AwayElo", "FTHome", "FTAway"]
    raw = pd.read_csv(path, usecols=usecols, parse_dates=["MatchDate"], low_memory=False)
    raw = raw.dropna()
    ctx = MatchContext("DOM", "league", domestic=True)
    feats = pd.DataFrame(
        [feature_row(h, a, False, False, ctx) for h, a in zip(raw["HomeElo"], raw["AwayElo"])]
    )
    feats["date"] = raw["MatchDate"].to_numpy()
    feats["home_goals"] = raw["FTHome"].to_numpy(dtype=float)
    feats["away_goals"] = raw["FTAway"].to_numpy(dtype=float)
    return feats.reset_index(drop=True)


# --- evaluation --------------------------------------------------------------------


def _season_start(season: str) -> pd.Timestamp:
    return pd.Timestamp(f"{season[:4]}-07-01")


def _fit(variant: Variant, euro: pd.DataFrame, dom: pd.DataFrame | None) -> EuropeModel:
    train = euro
    weights = np.ones(len(euro))
    if variant.domestic_weight > 0 and dom is not None and len(dom):
        train = pd.concat([euro[list(FEATURES) + ["home_goals", "away_goals"]], dom], ignore_index=True)
        weights = np.concatenate([weights, np.full(len(dom), variant.domestic_weight)])
    y = outcome_codes(train["home_goals"].to_numpy(), train["away_goals"].to_numpy())
    model = EuropeModel(c=variant.c, features=variant.features)
    return model.fit(train, y, train["home_goals"].to_numpy(), train["away_goals"].to_numpy(), weights)


def walk_forward(
    table: pd.DataFrame,
    dom: pd.DataFrame | None,
    seasons: Sequence[str] = TEST_SEASONS,
    variants: Sequence[Variant] = VARIANTS,
    hfa: float = 60.0,
) -> dict:
    """Season-by-season out-of-sample metrics for each variant and the Elo baselines."""
    per_season: dict[str, dict] = {}
    pooled: dict[str, list[np.ndarray]] = {}
    pooled_y: list[np.ndarray] = []
    for season in seasons:
        test = table[table["season"] == season]
        train = table[table["season"] < season]
        if test.empty or train.empty:
            continue
        y_test = outcome_codes(test["home_goals"].to_numpy(), test["away_goals"].to_numpy())
        y_train = outcome_codes(train["home_goals"].to_numpy(), train["away_goals"].to_numpy())
        dom_train = None if dom is None else dom[dom["date"] < _season_start(season)]
        probs: dict[str, np.ndarray] = {
            "elo_baseline": elo_baseline_probs(
                (test["elo_home"] - test["elo_away"]).to_numpy(),
                test["neutral"].to_numpy() > 0, float((y_train == 1).mean()), hfa,
            ),
            "base_rates": np.tile(np.bincount(y_train, minlength=3) / len(y_train), (len(test), 1)),
        }
        for variant in variants:
            probs[variant.name] = _fit(variant, train, dom_train).predict_proba(test)
        per_season[season] = {k: metrics.evaluate(p, y_test) for k, p in probs.items()}
        for k, p in probs.items():
            pooled.setdefault(k, []).append(p)
        pooled_y.append(y_test)
    y_all = np.concatenate(pooled_y) if pooled_y else np.array([], dtype=int)
    overall = {k: metrics.evaluate(np.vstack(v), y_all) for k, v in pooled.items()}
    return {"per_season": per_season, "overall": overall}


def best_variant(report: dict, variants: Sequence[Variant] = VARIANTS) -> Variant:
    overall = report.get("overall", {})
    scored = [(overall[v.name]["rps"], i, v) for i, v in enumerate(variants) if v.name in overall]
    return min(scored)[2] if scored else variants[0]


def format_report(report: dict) -> str:
    lines = ["| Season | Model | n | RPS | Log loss | Brier | Accuracy |", "|---|---|---|---|---|---|---|"]
    for season, res in [*report["per_season"].items(), ("all", report["overall"])]:
        for name, m in res.items():
            lines.append(
                f"| {season} | {name} | {m['n']} | {m['rps']:.4f} | {m['log_loss']:.4f} "
                f"| {m['brier']:.4f} | {m['accuracy']:.3f} |"
            )
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------------


def _db_inputs(database_url: str) -> tuple[list[tuple[str, str]], pd.DataFrame | None]:
    """DB team names and recent finished matches (best effort; empty when no DB)."""
    try:
        from matchpredict.db import repository
        from matchpredict.db.session import create_session_factory

        factory = create_session_factory(database_url)
        with factory() as session:
            teams = repository.teams(session, window_days=4000)
            recent = repository.finished_frame(
                session, after=DB_REPLAY_FROM.date(), include_europe=True
            )
        return teams, recent
    except Exception:
        log.exception("could not read the DB; continuing without DB teams/results")
        return [], None


def train(
    *,
    download: bool = True,
    use_db: bool = True,
    artifact: Path = DEFAULT_ARTIFACT,
    metrics_path: Path = DEFAULT_METRICS,
    today: dt.date | None = None,
) -> tuple[EuropeService, dict]:
    started = time.perf_counter()
    today = today or dt.date.today()
    if download:
        openfootball.download(OPENFOOTBALL_DIR, today)
    raw = openfootball.load_cached(OPENFOOTBALL_DIR)
    settings = get_settings()
    db_teams, recent = _db_inputs(settings.database_url) if use_db else ([], None)
    mapper = TeamNameMapper(candidates_from(ELO_CSV, db_teams))
    history = map_history(raw, mapper)
    log_misses(mapper, "openfootball")
    snapshots = load_snapshots(ELO_CSV)
    covered = {c for _, clubs in snapshots[-1:] for _, c, _ in clubs}
    coverage = mapping_coverage(history, covered)
    table, book, euro_log = replay(history, snapshots)
    table = pd.concat([table, history[["season", "home_goals", "away_goals", "league", "stage"]]], axis=1)
    dom = domestic_rows(MATCHES_CSV) if MATCHES_CSV.exists() else None
    report = walk_forward(table, dom)
    chosen = best_variant(report)
    log.info("walk-forward:\n%s\nchosen variant: %s", format_report(report), chosen.name)
    model = _fit(chosen, table, dom)
    version = f"europe-{today.isoformat()}"
    service = EuropeService(
        model=model,
        book=book,
        model_version=version,
        snapshot_date=snapshots[-1][0] if snapshots else None,
        last_result_date=pd.Timestamp(history["date"].max()) if len(history) else None,
        recent_euro=euro_log[-400:],
        aliases={
            **dict(zip(history["home_raw"], history["home"])),
            **dict(zip(history["away_raw"], history["away"])),
        },
    )
    applied = service.update_results(recent) if recent is not None and len(recent) else 0
    service.metrics = {
        "walk_forward": report,
        "chosen": chosen.name,
        "coverage": coverage,
        "history_matches": len(history),
        "seasons": sorted(history["season"].unique().tolist()),
        "db_results_applied": int(applied),
        "trained_on": len(table) + (len(dom) if dom is not None and chosen.domestic_weight else 0),
    }
    service.save(artifact)
    metrics_path.write_text(json.dumps(service.metrics, indent=2, default=str), encoding="utf-8")
    log.info("saved %s (%.1fs, %d DB results applied)", artifact, time.perf_counter() - started, applied)
    return service, report


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Train the MatchPredict European model")
    parser.add_argument("--no-download", action="store_true", help="use the cached openfootball files")
    parser.add_argument("--no-db", action="store_true", help="do not read team names / results from the DB")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service, report = train(download=not args.no_download, use_db=not args.no_db, artifact=args.artifact)
    print(format_report(report))  # CLI output
    print(json.dumps(service.metrics["coverage"], indent=2))
