"""Sequential, leakage-free team ratings and pre-match features.

Matches are processed one calendar day at a time: features for every match on a day
are read from the state *before* any of that day's results are applied, so a match
never sees its own result or a same-day result.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FORM_WINDOW = 5
STATS = ("shots", "target", "corners", "xg")  # optional columns home_<stat> / away_<stat>
EWMA_KEYS = ("goals",) + STATS
LEAGUE_PRIOR = (0.45, 0.26)  # home-win rate, draw rate
SEED_MIN_TEAMS = 10
SEED_OFFSET = 100.0


@dataclass(frozen=True)
class EloConfig:
    initial: float = 1500.0
    k: float = 20.0
    home_advantage: float = 60.0


@dataclass(frozen=True)
class PiConfig:
    learning_rate: float = 0.035
    cross_rate: float = 0.7
    base: float = 3.0


@dataclass(frozen=True)
class GoalRatingConfig:
    """Online Poisson attack/defence ratings: log(lambda) = league mean + attack - opponent defence."""

    learning_rate: float = 0.03
    league_rate: float = 0.005
    seed: float = -0.1  # newcomer to an established league starts below average
    prior_home: float = math.log(1.45)
    prior_away: float = math.log(1.1)


@dataclass(frozen=True)
class EngineConfig:
    elo: EloConfig = field(default_factory=EloConfig)
    pi: PiConfig = field(default_factory=PiConfig)
    goal: GoalRatingConfig = field(default_factory=GoalRatingConfig)
    elo_fast_k: float = 40.0
    ewma_alpha: float = 0.15
    league_alpha: float = 0.005


@dataclass(frozen=True)
class MatchResult:
    home: str
    away: str
    league: str
    date: np.datetime64
    home_goals: float
    away_goals: float
    stats: tuple[tuple[str, float, float], ...] = ()


def _elo_expectation(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _margin_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0


def _pi_to_goals(rating: float, base: float) -> float:
    return math.copysign(10.0 ** (abs(rating) / base) - 1.0, rating)


def _points(goals_for: float, goals_against: float) -> int:
    if goals_for > goals_against:
        return 3
    return 1 if goals_for == goals_against else 0


class RatingsEngine:
    """Holds rating state for every team; picklable so it can be served."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.elo: dict[str, float] = {}
        self.pi: dict[str, list[float]] = {}  # team -> [home rating, away rating]
        self.played: dict[str, int] = {}
        self.last_date: dict[str, np.datetime64] = {}
        self.form: dict[str, deque[tuple[int, float]]] = {}
        self.ewma: dict[tuple[str, str], float] = {}
        self.league_rates: dict[str, list[float]] = {}
        self.league_teams: dict[str, set[str]] = {}
        self.team_league: dict[str, str] = {}  # team -> league of its most recent match
        self.elo_fast: dict[str, float] = {}
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.league_mu: dict[str, list[float]] = {}  # league -> [log home rate, log away rate]

    # ----------------------------------------------------------------- reading
    def features_for(self, home: str, away: str, league: str, date) -> dict[str, float]:
        """Pre-match features for a fixture given the current state (no update)."""
        cfg = self.config
        day = np.datetime64(date, "D")
        elo_home, elo_away = self._elo_of(home, league), self._elo_of(away, league)
        pi_home = self.pi.get(home, (0.0, 0.0))[0]
        pi_away = self.pi.get(away, (0.0, 0.0))[1]
        home_rate, draw_rate = self.league_rates.get(league, LEAGUE_PRIOR)
        lam_home, lam_away = self._goal_lambdas(home, away, league)
        fast_diff = self._elo_of(home, league, self.elo_fast) - self._elo_of(away, league, self.elo_fast)
        feats = {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away,
            "elo_exp_home": _elo_expectation(elo_home - elo_away + cfg.elo.home_advantage),
            "pi_home": pi_home,
            "pi_away": pi_away,
            "pi_diff": pi_home - pi_away,
            "pi_pred_gd": _pi_to_goals(pi_home, cfg.pi.base) - _pi_to_goals(pi_away, cfg.pi.base),
            "league_home_rate": home_rate,
            "league_draw_rate": draw_rate,
            "elo_fast_diff": fast_diff,
            "gr_lambda_home": lam_home,
            "gr_lambda_away": lam_away,
            "gr_att_home": self._rating_of(self.attack, home, league),
            "gr_def_home": self._rating_of(self.defence, home, league),
            "gr_att_away": self._rating_of(self.attack, away, league),
            "gr_def_away": self._rating_of(self.defence, away, league),
        }
        for side, team in (("home", home), ("away", away)):
            feats.update(self._team_features(team, side, day))
        return feats

    def _team_features(self, team: str, side: str, day: np.datetime64) -> dict[str, float]:
        form = self.form.get(team, ())
        last = self.last_date.get(team)
        feats = {
            f"form5_{side}": float(sum(p for p, _ in form)),
            f"gd5_{side}": float(sum(g for _, g in form)),
            f"rest_{side}": float((day - last) / np.timedelta64(1, "D")) if last is not None else np.nan,
            f"played_{side}": float(self.played.get(team, 0)),
        }
        for key in EWMA_KEYS:
            for direction in ("for", "against"):
                feats[f"{key}_{direction}_{side}"] = self.ewma.get((team, f"{key}_{direction}"), np.nan)
        return feats

    def _elo_of(self, team: str, league: str, table: dict[str, float] | None = None) -> float:
        table = self.elo if table is None else table
        if team in table:
            return table[team]
        members = [table[t] for t in self.league_teams.get(league, ()) if t in table]
        if len(members) >= SEED_MIN_TEAMS:
            return float(np.mean(members)) - SEED_OFFSET
        return self.config.elo.initial

    def _rating_of(self, table: dict[str, float], team: str, league: str) -> float:
        if team in table:
            return table[team]
        established = len(self.league_teams.get(league, ())) >= SEED_MIN_TEAMS
        return self.config.goal.seed if established else 0.0

    def _goal_lambdas(self, home: str, away: str, league: str) -> tuple[float, float]:
        cfg = self.config.goal
        mu_home, mu_away = self.league_mu.get(league, (cfg.prior_home, cfg.prior_away))
        att_h, def_h = self._rating_of(self.attack, home, league), self._rating_of(self.defence, home, league)
        att_a, def_a = self._rating_of(self.attack, away, league), self._rating_of(self.defence, away, league)
        return math.exp(mu_home + att_h - def_a), math.exp(mu_away + att_a - def_h)

    # ---------------------------------------------------------------- updating
    def update(self, result: MatchResult) -> None:
        """Apply one finished match to the state."""
        self._update_elo(result)
        self._update_elo(result, self.elo_fast, self.config.elo_fast_k)
        self._update_goal_ratings(result)
        self._update_pi(result)
        self._update_team_history(result)
        self._update_league(result)

    def _update_elo(self, r: MatchResult, table: dict[str, float] | None = None, k: float | None = None) -> None:
        cfg = self.config.elo
        table = self.elo if table is None else table
        k = cfg.k if k is None else k
        elo_home, elo_away = self._elo_of(r.home, r.league, table), self._elo_of(r.away, r.league, table)
        expected = _elo_expectation(elo_home - elo_away + cfg.home_advantage)
        score = (_points(r.home_goals, r.away_goals) / 3.0) if r.home_goals != r.away_goals else 0.5
        margin = _margin_multiplier(int(abs(r.home_goals - r.away_goals)))
        delta = k * margin * (score - expected)
        table[r.home] = elo_home + delta
        table[r.away] = elo_away - delta

    def _update_goal_ratings(self, r: MatchResult) -> None:
        cfg = self.config.goal
        lam_home, lam_away = self._goal_lambdas(r.home, r.away, r.league)
        err_home, err_away = r.home_goals - lam_home, r.away_goals - lam_away
        att_h, def_h = self._rating_of(self.attack, r.home, r.league), self._rating_of(self.defence, r.home, r.league)
        att_a, def_a = self._rating_of(self.attack, r.away, r.league), self._rating_of(self.defence, r.away, r.league)
        lr = cfg.learning_rate
        self.attack[r.home] = att_h + lr * err_home
        self.defence[r.away] = def_a - lr * err_home
        self.attack[r.away] = att_a + lr * err_away
        self.defence[r.home] = def_h - lr * err_away
        mu = self.league_mu.setdefault(r.league, [cfg.prior_home, cfg.prior_away])
        mu[0] += cfg.league_rate * err_home
        mu[1] += cfg.league_rate * err_away

    def _update_pi(self, r: MatchResult) -> None:
        cfg = self.config.pi
        home = self.pi.setdefault(r.home, [0.0, 0.0])
        away = self.pi.setdefault(r.away, [0.0, 0.0])
        predicted = _pi_to_goals(home[0], cfg.base) - _pi_to_goals(away[1], cfg.base)
        error = (r.home_goals - r.away_goals) - predicted
        weighted = math.copysign(cfg.base * math.log10(1.0 + abs(error)), error)
        step = cfg.learning_rate * weighted
        home[0] += step
        home[1] += step * cfg.cross_rate
        away[1] -= step
        away[0] -= step * cfg.cross_rate

    def _update_team_history(self, r: MatchResult) -> None:
        sides = (
            (r.home, r.home_goals, r.away_goals, 0),
            (r.away, r.away_goals, r.home_goals, 1),
        )
        for team, scored, conceded, idx in sides:
            self.played[team] = self.played.get(team, 0) + 1
            self.last_date[team] = np.datetime64(r.date, "D")
            self.form.setdefault(team, deque(maxlen=FORM_WINDOW)).append(
                (_points(scored, conceded), scored - conceded)
            )
            self._ewma(team, "goals_for", scored)
            self._ewma(team, "goals_against", conceded)
            for stat, home_val, away_val in r.stats:
                mine, theirs = (home_val, away_val) if idx == 0 else (away_val, home_val)
                self._ewma(team, f"{stat}_for", mine)
                self._ewma(team, f"{stat}_against", theirs)

    def _ewma(self, team: str, key: str, value: float) -> None:
        if value is None or math.isnan(value):
            return
        prev = self.ewma.get((team, key))
        self.ewma[(team, key)] = value if prev is None else prev + self.config.ewma_alpha * (value - prev)

    def _update_league(self, r: MatchResult) -> None:
        alpha = self.config.league_alpha
        rates = self.league_rates.setdefault(r.league, list(LEAGUE_PRIOR))
        rates[0] += alpha * (float(r.home_goals > r.away_goals) - rates[0])
        rates[1] += alpha * (float(r.home_goals == r.away_goals) - rates[1])
        teams = self.league_teams.setdefault(r.league, set())
        teams.update((r.home, r.away))
        self.team_league[r.home] = r.league
        self.team_league[r.away] = r.league

    # ------------------------------------------------------------------ batch
    def build_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Feature rows for every match (same index/order), updating state as it goes.

        Rows with missing goals (upcoming fixtures) get features but do not update state.
        """
        dates = matches["date"].to_numpy(dtype="datetime64[D]")
        home, away, league = (matches[c].to_numpy() for c in ("home", "away", "league"))
        home_goals = matches["home_goals"].to_numpy(dtype=float)
        away_goals = matches["away_goals"].to_numpy(dtype=float)
        stat_arrays = [
            (s, matches[f"home_{s}"].to_numpy(dtype=float), matches[f"away_{s}"].to_numpy(dtype=float))
            for s in STATS
            if f"home_{s}" in matches.columns and f"away_{s}" in matches.columns
        ]

        rows: list[dict[str, float] | None] = [None] * len(matches)
        pending: list[MatchResult] = []
        current_day = None
        for i in np.argsort(dates, kind="stable"):
            if dates[i] != current_day:
                self._apply(pending)
                pending, current_day = [], dates[i]
            rows[i] = self.features_for(home[i], away[i], league[i], dates[i])
            if not (math.isnan(home_goals[i]) or math.isnan(away_goals[i])):
                pending.append(
                    MatchResult(
                        home=home[i], away=away[i], league=league[i], date=dates[i],
                        home_goals=home_goals[i], away_goals=away_goals[i],
                        stats=tuple((s, h[i], a[i]) for s, h, a in stat_arrays),
                    )
                )
        self._apply(pending)
        return pd.DataFrame(rows, index=matches.index)

    def _apply(self, results: list[MatchResult]) -> None:
        for result in results:
            self.update(result)
