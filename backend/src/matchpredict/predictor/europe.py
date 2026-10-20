"""Cross-league "European" model for UEFA club competitions (UCL / UEL / UECL).

Strength: one ClubElo-scale rating per club. Ratings come from the most recent
ClubElo snapshot (``data/raw/EloRatings.csv``) at or before the match day; clubs
outside ClubElo's coverage get a dynamic Elo seeded from a country / stage prior
and updated with their European results. After the last snapshot the ratings are
kept current in-process with domestic and European results from the DB.

Outcome: multinomial logistic regression on the rating difference plus stage,
competition, coverage and second-leg features. Goals: two Poisson regressions
-> score matrix, rescaled so its 1X2 margins equal the outcome model.

Train / evaluate: ``uv run python -m matchpredict.predictor.europe``
(see ``predictor/europe_train.py``) -> ``backend/artifacts/europe.joblib``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from matchpredict.contracts import Prediction
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES, LEAGUES
from matchpredict.ingest.teamnames import COUNTRY_CODES, TeamNameMapper

log = logging.getLogger(__name__)

MODEL_NAME = "europe"
ARTIFACT_VERSION = 1
MAX_GOALS = 10
TOP_SCORES = 5
LEG_WINDOW_DAYS = 35
RECENT_EURO_DAYS = 60
STAGES = ("qual", "league", "ko", "final")
UNKNOWN_MARGIN = 50.0  # an unseen club is assumed a bit weaker than the competition median

FEATURES: tuple[str, ...] = (
    "elo_diff", "abs_diff", "neutral", "st_qual", "st_ko", "comp_uel", "comp_uecl",
    "unc_home", "unc_away", "diff_x_unc", "leg2", "agg_gd", "domestic", "diff_x_domestic",
)  # fmt: skip


@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    k_provisional: float = 40.0
    provisional_games: int = 10
    home_adv: float = 65.0  # domestic, ClubElo-like
    euro_home_adv: float = 60.0
    prior_qual: float = 1300.0
    prior_main: float = 1550.0
    prior_ko: float = 1600.0


def elo_expectation(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def margin_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


def infer_stage(competition: str, day: pd.Timestamp) -> str:
    """Best guess of the stage when the source does not say (UEFA calendar)."""
    if competition not in EUROPEAN_LEAGUES:
        return "league"
    if day.month in (6, 7, 8):
        return "qual"
    if day.month in (9, 10, 11, 12, 1):
        return "league"
    return "ko"


class EloBook:
    """Ratings with per-team timestamps: a match only moves ratings older than itself."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self.ratings: dict[str, float] = {}
        self.stamp: dict[str, pd.Timestamp] = {}
        self.games: dict[str, int] = {}
        self.country: dict[str, str | None] = {}
        self.league: dict[str, str] = {}
        self.covered: set[str] = set()

    # --- snapshots ----------------------------------------------------------------
    def apply_snapshot(self, day: pd.Timestamp, clubs: Iterable[tuple[str, str, float]]) -> None:
        seen: set[str] = set()
        for club, country, elo in clubs:
            self.ratings[club] = float(elo)
            self.stamp[club] = day
            self.country.setdefault(club, country)
            seen.add(club)
        self.covered = seen

    # --- priors -------------------------------------------------------------------
    def prior(self, country: str | None, stage: str) -> float:
        cfg = self.config
        peers = [
            r for t, r in self.ratings.items()
            if country is not None and self.country.get(t) == country
        ]  # fmt: skip
        if peers:
            return float(np.mean(peers))
        if stage == "qual":
            return cfg.prior_qual
        return cfg.prior_ko if stage in ("ko", "final") else cfg.prior_main

    def ensure(self, team: str, country: str | None, stage: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.prior(country, stage)
            self.country[team] = country
            self.games[team] = 0
        elif country is not None and self.country.get(team) is None:
            self.country[team] = country
        return self.ratings[team]

    # --- updates ------------------------------------------------------------------
    def _k(self, team: str) -> float:
        cfg = self.config
        if team in self.covered or self.games.get(team, 0) >= cfg.provisional_games:
            return cfg.k
        return cfg.k_provisional

    def update(
        self,
        day: pd.Timestamp,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        *,
        neutral: bool = False,
        european: bool = True,
    ) -> bool:
        """Apply one result; each side only moves when the match is newer than its rating."""
        cfg = self.config
        stale_h = self.stamp.get(home, pd.Timestamp.min) < day
        stale_a = self.stamp.get(away, pd.Timestamp.min) < day
        if not (stale_h or stale_a):
            return False
        hfa = 0.0 if neutral else (cfg.euro_home_adv if european else cfg.home_adv)
        rh, ra = self.ratings[home], self.ratings[away]
        expected = elo_expectation(rh - ra + hfa)
        actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        delta = (actual - expected) * margin_multiplier(home_goals - away_goals)
        if stale_h:
            self.ratings[home] = rh + self._k(home) * delta
            self.stamp[home] = day
        if stale_a:
            self.ratings[away] = ra - self._k(away) * delta
            self.stamp[away] = day
        for team in (home, away):
            self.games[team] = self.games.get(team, 0) + 1
        return True

    def state(self) -> dict:
        return {
            "ratings": dict(self.ratings),
            "stamp": {k: v.isoformat() for k, v in self.stamp.items()},
            "games": dict(self.games),
            "country": dict(self.country),
            "league": dict(self.league),
            "covered": sorted(self.covered),
            "config": asdict(self.config),
        }

    @classmethod
    def from_state(cls, state: Mapping) -> EloBook:
        book = cls(EloConfig(**state["config"]))
        book.ratings = dict(state["ratings"])
        book.stamp = {k: pd.Timestamp(v) for k, v in state["stamp"].items()}
        book.games = dict(state["games"])
        book.country = dict(state["country"])
        book.league = dict(state["league"])
        book.covered = set(state["covered"])
        return book


# --- features ----------------------------------------------------------------------


@dataclass(frozen=True)
class MatchContext:
    competition: str
    stage: str
    neutral: bool = False
    leg2: bool = False
    agg_gd: float = 0.0  # first-leg goal difference from the current home side's view
    domestic: bool = False


def feature_row(
    r_home: float, r_away: float, unc_home: bool, unc_away: bool, ctx: MatchContext
) -> dict[str, float]:
    diff = (r_home - r_away) / 100.0
    any_unc = float(unc_home or unc_away)
    return {
        "elo_diff": diff,
        "abs_diff": abs(diff),
        "neutral": float(ctx.neutral),
        "st_qual": float(ctx.stage == "qual"),
        "st_ko": float(ctx.stage in ("ko", "final")),
        "comp_uel": float(ctx.competition == "UEL"),
        "comp_uecl": float(ctx.competition == "UECL"),
        "unc_home": float(unc_home),
        "unc_away": float(unc_away),
        "diff_x_unc": diff * any_unc,
        "leg2": float(ctx.leg2),
        "agg_gd": float(np.clip(ctx.agg_gd, -3, 3)),
        "domestic": float(ctx.domestic),
        "diff_x_domestic": diff * float(ctx.domestic),
    }


# --- goals -------------------------------------------------------------------------


def _poisson_pmf(lam: float) -> np.ndarray:
    k = np.arange(MAX_GOALS + 1)
    logp = -lam + k * math.log(max(lam, 1e-9)) - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(logp)


def score_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    matrix = np.outer(_poisson_pmf(lam_home), _poisson_pmf(lam_away))
    return matrix / matrix.sum()


def rescale_to_1x2(matrix: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Scale the home-win / draw / away-win regions so their sums equal ``probs``."""
    idx = np.arange(matrix.shape[0])
    home = idx[:, None] > idx[None, :]
    draw = idx[:, None] == idx[None, :]
    away = ~(home | draw)
    out = matrix.copy()
    for mask, p in ((home, probs[0]), (draw, probs[1]), (away, probs[2])):
        total = out[mask].sum()
        if total > 0:
            out[mask] *= p / total
    return out / out.sum()


def summarise_matrix(matrix: np.ndarray) -> dict:
    idx = np.arange(matrix.shape[0])
    total_goals = idx[:, None] + idx[None, :]
    flat = [
        (f"{h}-{a}", float(matrix[h, a]))
        for h in range(matrix.shape[0])
        for a in range(matrix.shape[1])
    ]
    flat.sort(key=lambda s: s[1], reverse=True)
    return {
        "expected_goals": {
            "home": float((matrix.sum(axis=1) * idx).sum()),
            "away": float((matrix.sum(axis=0) * idx).sum()),
        },
        "over_2_5": float(matrix[total_goals > 2].sum()),
        "btts": float(matrix[1:, 1:].sum()),
        "top_scores": flat[:TOP_SCORES],
    }


# --- model -------------------------------------------------------------------------


@dataclass
class EuropeModel:
    """Outcome (multinomial LR) + goals (two Poisson GLMs) on FEATURES."""

    c: float = 1.0
    features: tuple[str, ...] = FEATURES
    scaler: StandardScaler | None = None
    outcome: LogisticRegression | None = None
    goals_home: PoissonRegressor | None = None
    goals_away: PoissonRegressor | None = None

    def _x(self, frame: pd.DataFrame, fit: bool = False) -> np.ndarray:
        x = frame[list(self.features)].to_numpy(dtype=float)
        if fit:
            self.scaler = StandardScaler().fit(x)
        assert self.scaler is not None
        scale = np.where(self.scaler.scale_ > 0, self.scaler.scale_, 1.0)
        return (x - self.scaler.mean_) / scale

    def fit(
        self,
        frame: pd.DataFrame,
        y: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> EuropeModel:
        x = self._x(frame, fit=True)
        self.outcome = LogisticRegression(C=self.c, max_iter=2000)
        self.outcome.fit(x, y, sample_weight=weights)
        self.goals_home = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(
            x, home_goals, sample_weight=weights
        )
        self.goals_away = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(
            x, away_goals, sample_weight=weights
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.outcome is not None
        probs = self.outcome.predict_proba(self._x(frame))
        order = np.argsort(self.outcome.classes_)
        return probs[:, order]

    def predict_goals(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self.goals_home is not None and self.goals_away is not None
        x = self._x(frame)
        return self.goals_home.predict(x), self.goals_away.predict(x)


def elo_baseline_probs(diff_elo: np.ndarray, neutral: np.ndarray, draw_rate: float, hfa: float) -> np.ndarray:
    """Simple Elo baseline: Elo expectation with a constant draw share."""
    expected = 1.0 / (1.0 + 10.0 ** (-(diff_elo + np.where(neutral, 0.0, hfa)) / 400.0))
    p_draw = np.full_like(expected, draw_rate)
    p_home = np.clip(expected - draw_rate / 2, 0.02, None)
    p_away = np.clip(1 - expected - draw_rate / 2, 0.02, None)
    probs = np.column_stack([p_home, p_draw, p_away])
    return probs / probs.sum(axis=1, keepdims=True)


# --- service -----------------------------------------------------------------------


def _country_of_league(league: str | None) -> str | None:
    if league is None or league not in LEAGUES:
        return None
    return COUNTRY_CODES.get(LEAGUES[league].country)


@dataclass
class EuropeService:
    """Serves contracts.Prediction for UEFA (and cross-league) fixtures."""

    model: EuropeModel
    book: EloBook
    model_version: str
    snapshot_date: pd.Timestamp | None = None
    last_result_date: pd.Timestamp | None = None
    metrics: dict = field(default_factory=dict)
    recent_euro: list[dict] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._mapper: TeamNameMapper | None = None

    # --- names --------------------------------------------------------------------
    def _names(self) -> TeamNameMapper:
        if self._mapper is None or len(self._mapper.candidates) != len(self.book.ratings):
            self._mapper = TeamNameMapper(dict(self.book.country))
        return self._mapper

    def resolve(self, name: str, country: str | None = None) -> str | None:
        if name in self.book.ratings:
            return name
        if name in self.aliases and self.aliases[name] in self.book.ratings:
            return self.aliases[name]
        return self._names().map(name, country)

    def known_teams(self) -> dict[str, str]:
        out = {}
        for team in self.book.ratings:
            out[team] = self.book.league.get(team) or "UCL"
        return out

    def strength(self, team: str) -> float | None:
        resolved = self.resolve(team)
        return None if resolved is None else float(self.book.ratings[resolved])

    def team_ratings(self, team: str) -> dict[str, float]:
        resolved = self.resolve(team)
        if resolved is None:
            raise KeyError(team)
        return {"elo": float(self.book.ratings[resolved])}

    # --- prediction ---------------------------------------------------------------
    def _first_leg(self, home: str, away: str, competition: str, day: pd.Timestamp) -> dict | None:
        for rec in reversed(self.recent_euro):
            gap = (day - pd.Timestamp(rec["date"])).days
            if (
                rec["home"] == away
                and rec["away"] == home
                and rec["league"] == competition
                and 0 < gap <= LEG_WINDOW_DAYS
            ):
                return rec
        return None

    def context(
        self,
        home: str,
        away: str,
        competition: str,
        day: pd.Timestamp,
        stage: str | None = None,
        neutral: bool = False,
    ) -> MatchContext:
        stage = stage or infer_stage(competition, day)
        european = competition in EUROPEAN_LEAGUES
        leg = self._first_leg(home, away, competition, day) if european and stage != "league" else None
        agg = 0.0 if leg is None else float(leg["away_goals"] - leg["home_goals"])
        return MatchContext(
            competition=competition,
            stage="final" if neutral and stage == "ko" else stage,
            neutral=neutral,
            leg2=leg is not None,
            agg_gd=agg,
            domestic=not european,
        )

    def competition_prior(self, competition: str) -> float:
        """Rating for a club we have never seen: median of recent entrants minus a margin."""
        teams = {t for r in self.recent_euro if r["league"] == competition for t in (r["home"], r["away"])}
        rated = [self.book.ratings[t] for t in teams if t in self.book.ratings]
        if not rated:
            return self.book.config.prior_main - UNKNOWN_MARGIN
        return float(np.median(rated)) - UNKNOWN_MARGIN

    def _side(self, name: str, competition: str, allow_unknown: bool) -> tuple[str, float, bool, bool]:
        """(name, rating, uncovered?, low confidence?) for one side."""
        resolved = self.resolve(name)
        if resolved is None:
            if not allow_unknown:
                raise KeyError(name)
            return name, self.competition_prior(competition), True, True
        provisional = (
            resolved not in self.book.covered
            and self.book.games.get(resolved, 0) < self.book.config.provisional_games
        )
        return resolved, self.book.ratings[resolved], resolved not in self.book.covered, provisional

    def predict(
        self,
        home: str,
        away: str,
        competition: str,
        date: pd.Timestamp,
        odds: tuple[float, float, float] | None = None,  # accepted for Protocol parity; unused
        *,
        stage: str | None = None,
        neutral: bool = False,
        allow_unknown: bool | None = None,
    ) -> Prediction:
        """Prediction for a fixture. Unknown clubs in UEFA competitions get a competition
        prior (``confidence="low"``); elsewhere they raise KeyError."""
        if allow_unknown is None:
            allow_unknown = competition in EUROPEAN_LEAGUES
        h, rh, unc_h, low_h = self._side(home, competition, allow_unknown)
        a, ra, unc_a, low_a = self._side(away, competition, allow_unknown)
        day = pd.Timestamp(date)
        ctx = self.context(h, a, competition, day, stage, neutral)
        row = feature_row(rh, ra, unc_h, unc_a, ctx)
        frame = pd.DataFrame([row])
        probs = self.model.predict_proba(frame)[0]
        lam_h, lam_a = (float(v[0]) for v in self.model.predict_goals(frame))
        matrix = rescale_to_1x2(score_matrix(lam_h, lam_a), probs)
        summary = summarise_matrix(matrix)
        rounded = np.round(probs, 4)
        return {
            "probs": {"home": float(rounded[0]), "draw": float(rounded[1]), "away": float(rounded[2])},
            "expected_goals": {k: round(v, 2) for k, v in summary["expected_goals"].items()},
            "over_2_5": round(summary["over_2_5"], 4),
            "btts": round(summary["btts"], 4),
            "top_scores": [{"score": s, "p": round(p, 4)} for s, p in summary["top_scores"]],
            "model": MODEL_NAME,
            "model_version": self.model_version,
            "confidence": "low" if (low_h or low_a) else "high",
        }

    # --- updates ------------------------------------------------------------------
    def update_results(self, matches: pd.DataFrame) -> int:
        """Apply finished matches (MATCH_COLUMNS; optional ``stage``) to the ratings."""
        if matches.empty:
            return 0
        frame = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("date", kind="mergesort")
        applied = 0
        for row in frame.itertuples(index=False):
            applied += self._apply_row(row)
        cutoff = (self.last_result_date or pd.Timestamp.min) - pd.Timedelta(days=RECENT_EURO_DAYS)
        self.recent_euro = [r for r in self.recent_euro if pd.Timestamp(r["date"]) >= cutoff]
        return applied

    def _apply_row(self, row) -> int:
        day = pd.Timestamp(row.date)
        league = str(row.league)
        european = league in EUROPEAN_LEAGUES
        stage = getattr(row, "stage", None) or infer_stage(league, day)
        if not european:
            home = self.resolve(row.home) or row.home
            away = self.resolve(row.away) or row.away
        else:
            home, away = row.home, row.away
        country = None if european else _country_of_league(league)
        self.book.ensure(home, country, stage)
        self.book.ensure(away, country, stage)
        if not european:
            self.book.league[home] = self.book.league[away] = league
        else:
            for team in (home, away):
                self.book.league.setdefault(team, league)
        hg, ag = int(row.home_goals), int(row.away_goals)
        moved = self.book.update(day, home, away, hg, ag, european=european)
        if not moved:
            return 0
        if european:
            self.recent_euro.append(
                {"date": day.isoformat(), "league": league, "home": home, "away": away,
                 "home_goals": hg, "away_goals": ag}
            )  # fmt: skip
        if self.last_result_date is None or day > self.last_result_date:
            self.last_result_date = day
        return 1

    # --- persistence --------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "model": self.model,
            "book": self.book.state(),
            "model_version": self.model_version,
            "snapshot_date": self.snapshot_date,
            "last_result_date": self.last_result_date,
            "metrics": self.metrics,
            "recent_euro": self.recent_euro,
            "aliases": self.aliases,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> EuropeService:
        payload = joblib.load(Path(path))
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(f"unsupported europe artifact version {payload.get('artifact_version')}")
        return cls(
            model=payload["model"],
            book=EloBook.from_state(payload["book"]),
            model_version=payload["model_version"],
            snapshot_date=payload["snapshot_date"],
            last_result_date=payload["last_result_date"],
            metrics=payload.get("metrics", {}),
            recent_euro=list(payload.get("recent_euro", [])),
            aliases=dict(payload.get("aliases", {})),
        )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI shim
    from matchpredict.predictor.europe_train import main as train_main

    train_main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
