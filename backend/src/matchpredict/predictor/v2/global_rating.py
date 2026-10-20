"""Own global (cross-country) club ratings: Elo + Poisson attack/defence in one chronological pass.

Effective strength of a club = its own rating + its **country offset**:

    elo(team) = r[team] + off[country(team)]

* Domestic results are zero-sum inside a league and move only ``r`` (divisions of one
  country are linked through promotion/relegation, because a club keeps its rating).
* European (UCL/UEL/UECL) results move the clubs *and* their countries' offsets, so a
  league whose clubs over-perform in Europe is lifted as a whole (cross-league linkage).
  Countries without domestic data (CRO, UKR, KAZ, …) are rated from European results
  only; a new club from such a country starts from the country's prior (its clubs' mean).
* Home advantage is learned online per domestic league and separately for UEFA matches;
  finals are neutral. Results use a margin-of-victory multiplier.
* New clubs: ClubElo's last snapshot strictly before the match day (optional), else the
  bottom of their league, else their country's prior, else a stage prior.

The same structure is used for goal ratings: log λ = μ[competition] + att(team) − def(opp),
with attack/defence country offsets updated only by European matches.

Day-batch semantics (as ``predictor/ratings.py``): every match of a day reads the state
before any result of that day is applied, so no match sees its own or a same-day result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matchpredict.ingest.leagues import EUROPEAN_LEAGUES

EURO_KEY = "EURO"
ACTIVE_DAYS = 400  # a club counts as a current league member if it played within this window
SEED_MIN_TEAMS = 8
BOTTOM_N = 4
SEED_CODES = {"rated": 0.0, "clubelo": 1.0, "league": 2.0, "country": 3.0, "stage": 4.0}


@dataclass(frozen=True)
class GlobalConfig:
    # defaults chosen by predictor/v2/tune.py on 2016/17-2020/21 (see docs/08-model-v2.md)
    k: float = 15.0
    k_euro: float = 30.0
    provisional_games: int = 12
    provisional_factor: float = 1.75
    k_country: float = 6.0  # country offset step per unit (S - E) of a European match
    hfa_init: float = 65.0
    hfa_euro_init: float = 55.0
    k_hfa: float = 0.3
    mov: str = "log"  # "clubelo" | "log" | "none"
    seed_clubelo: bool = True
    promoted_gap: float = 0.0  # below the mean of the league's bottom four
    newcomer_gap: float = 120.0  # below the country's mean for a new UEFA-only club
    prior_qual: float = 1300.0
    prior_league: float = 1500.0
    prior_ko: float = 1580.0
    season_regress: float = 0.0  # shrink toward the league mean at each 1 July
    goal_lr: float = 0.03
    goal_country_lr: float = 0.01
    goal_league_lr: float = 0.005
    goal_seed_scale: float = 0.0012  # elo points -> log-goal seed for new clubs
    prior_home: float = math.log(1.45)
    prior_away: float = math.log(1.1)
    prior_euro_home: float = math.log(1.55)
    prior_euro_away: float = math.log(1.15)


def elo_expectation(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def margin_multiplier(goal_diff: int, kind: str) -> float:
    gd = abs(int(goal_diff))
    if kind == "none" or gd <= 1:
        return 1.0
    if kind == "log":
        return 1.0 + math.log(gd)  # 2 -> 1.69, 3 -> 2.10
    return 1.5 if gd == 2 else (11.0 + gd) / 8.0  # ClubElo


class ClubEloBook:
    """ClubElo snapshots: rating of a club at the latest snapshot strictly before a day."""

    def __init__(self, snapshots: pd.DataFrame | None = None) -> None:
        self._dates: dict[str, np.ndarray] = {}  # int64 days since epoch, ascending
        self._values: dict[str, np.ndarray] = {}
        self.last_snapshot: pd.Timestamp | None = None
        if snapshots is not None and len(snapshots):
            frame = snapshots.dropna(subset=["elo"]).sort_values(["club", "date"], kind="mergesort")
            days = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
            values = frame["elo"].to_numpy(dtype=float)
            clubs = frame["club"].to_numpy()
            bounds = np.flatnonzero(clubs[1:] != clubs[:-1]) + 1
            for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(clubs)], strict=True):
                self._dates[str(clubs[lo])] = days[lo:hi]
                self._values[str(clubs[lo])] = values[lo:hi]
            self.last_snapshot = pd.Timestamp(pd.to_datetime(frame["date"]).max())

    @classmethod
    def from_csv(cls, path) -> ClubEloBook:
        return cls(pd.read_csv(path, usecols=["date", "club", "elo"]))

    def __contains__(self, club: str) -> bool:
        return club in self._dates

    def get(self, club: str, day: np.datetime64) -> float:
        dates = self._dates.get(club)
        if dates is None:
            return math.nan
        i = int(np.searchsorted(dates, np.datetime64(day, "D").astype(np.int64), side="left"))
        return float(self._values[club][i - 1]) if i > 0 else math.nan


@dataclass(frozen=True)
class GlobalMatch:
    home: str
    away: str
    league: str
    date: np.datetime64
    home_goals: float
    away_goals: float
    stage: str = "dom"
    neutral: bool = False


@dataclass
class GlobalRating:
    """Rating state (picklable). ``countries``: canonical club -> country code."""

    config: GlobalConfig = field(default_factory=GlobalConfig)
    countries: dict[str, str] = field(default_factory=dict)
    clubelo: ClubEloBook | None = None
    r: dict[str, float] = field(default_factory=dict)
    off: dict[str, float] = field(default_factory=dict)
    hfa: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    seed_kind: dict[str, str] = field(default_factory=dict)
    last_date: dict[str, np.datetime64] = field(default_factory=dict)
    league_of: dict[str, str] = field(default_factory=dict)  # latest domestic league
    members: dict[str, set[str]] = field(default_factory=dict)  # league -> teams ever seen
    att: dict[str, float] = field(default_factory=dict)
    dfn: dict[str, float] = field(default_factory=dict)
    catt: dict[str, float] = field(default_factory=dict)
    cdef: dict[str, float] = field(default_factory=dict)
    mu: dict[str, list[float]] = field(default_factory=dict)
    season: int | None = None

    def __post_init__(self) -> None:
        self._league_cache: dict[tuple[str, np.datetime64], tuple[float, float]] = {}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._league_cache = {}

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_league_cache"}

    # ------------------------------------------------------------------ helpers
    def country(self, team: str) -> str | None:
        return self.countries.get(team)

    def offset(self, team: str) -> float:
        c = self.country(team)
        return self.off.get(c, 0.0) if c else 0.0

    def elo(self, team: str) -> float:
        return self.r[team] + self.offset(team)

    def _active(self, league: str, day: np.datetime64) -> list[str]:
        horizon = np.datetime64(day, "D") - np.timedelta64(ACTIVE_DAYS, "D")
        return [t for t in self.members.get(league, ()) if t in self.r and self.last_date.get(t, horizon) > horizon]

    def league_strength(self, league: str, day: np.datetime64) -> tuple[float, float]:
        """(mean elo, mean of the bottom four) of a league's current members (NaN if too few)."""
        key = (league, np.datetime64(day, "D"))
        hit = self._league_cache.get(key)
        if hit is None:
            ratings = sorted(self.elo(t) for t in self._active(league, day))
            if len(ratings) < SEED_MIN_TEAMS:
                hit = (math.nan, math.nan)
            else:
                hit = (float(np.mean(ratings)), float(np.mean(ratings[:BOTTOM_N])))
            self._league_cache[key] = hit
        return hit

    def _country_clubs(self, country: str, day: np.datetime64) -> list[float]:
        horizon = np.datetime64(day, "D") - np.timedelta64(3 * ACTIVE_DAYS, "D")
        return [
            self.elo(t) for t, c in self.countries.items()
            if c == country and t in self.r and self.last_date.get(t, horizon) > horizon
        ]  # fmt: skip

    def _domestic_leagues_of(self, country: str) -> list[str]:
        from matchpredict.predictor.v2.names import league_country

        return [lg for lg in self.members if league_country(lg) == country]

    # ------------------------------------------------------------------ seeding
    def seed(self, team: str, league: str, stage: str, day: np.datetime64) -> tuple[float, str]:
        """(effective elo, seed kind) for a club never rated before (no state change)."""
        cfg = self.config
        if cfg.seed_clubelo and self.clubelo is not None:
            value = self.clubelo.get(team, day)
            if not math.isnan(value):
                return value, "clubelo"
        if league not in EUROPEAN_LEAGUES:
            _, bottom = self.league_strength(league, day)
            if not math.isnan(bottom):
                return bottom - cfg.promoted_gap, "league"
        country = self.country(team)
        if country:
            leagues = self._domestic_leagues_of(country)
            bottoms = [b for b in (self.league_strength(lg, day)[1] for lg in leagues) if not math.isnan(b)]
            if bottoms:  # a club from a country we rate domestically, outside our divisions
                return min(bottoms) - cfg.promoted_gap, "country"
            clubs = self._country_clubs(country, day)
            if clubs:
                return float(np.mean(clubs)) - cfg.newcomer_gap, "country"
        if stage == "qual":
            return cfg.prior_qual, "stage"
        return (cfg.prior_ko if stage in ("ko", "final") else cfg.prior_league), "stage"

    def _goal_reference(self, league: str, day: np.datetime64) -> float:
        mean, _ = self.league_strength(league, day) if league not in EUROPEAN_LEAGUES else (math.nan, 0)
        return 1500.0 if math.isnan(mean) else mean

    def ensure(self, team: str, league: str, stage: str, day: np.datetime64) -> None:
        if team in self.r:
            return
        elo, kind = self.seed(team, league, stage, day)
        self.r[team] = elo - self.offset(team)
        self.seed_kind[team] = kind
        self.games[team] = 0
        rel = self.config.goal_seed_scale * (elo - self._goal_reference(league, day))
        country = self.country(team)
        self.att[team] = rel - (self.catt.get(country, 0.0) if country else 0.0)
        self.dfn[team] = rel - (self.cdef.get(country, 0.0) if country else 0.0)

    def _peek(self, team: str, league: str, stage: str, day: np.datetime64) -> tuple[float, str, float, float]:
        """(elo, seed kind, attack, defence) without changing state."""
        if team in self.r:
            return self.elo(team), "rated", self._att(team), self._def(team)
        elo, kind = self.seed(team, league, stage, day)
        rel = self.config.goal_seed_scale * (elo - self._goal_reference(league, day))
        return elo, kind, rel, rel

    def _att(self, team: str) -> float:
        c = self.country(team)
        return self.att.get(team, 0.0) + (self.catt.get(c, 0.0) if c else 0.0)

    def _def(self, team: str) -> float:
        c = self.country(team)
        return self.dfn.get(team, 0.0) + (self.cdef.get(c, 0.0) if c else 0.0)

    def _mu(self, league: str) -> list[float]:
        cfg = self.config
        if league in EUROPEAN_LEAGUES:
            return self.mu.get(EURO_KEY, [cfg.prior_euro_home, cfg.prior_euro_away])
        return self.mu.get(league, [cfg.prior_home, cfg.prior_away])

    def _hfa(self, league: str, neutral: bool) -> float:
        if neutral:
            return 0.0
        if league in EUROPEAN_LEAGUES:
            return self.hfa.get(EURO_KEY, self.config.hfa_euro_init)
        return self.hfa.get(league, self.config.hfa_init)

    # ------------------------------------------------------------------ reading
    def features_for(
        self, home: str, away: str, league: str, day, stage: str = "dom", neutral: bool = False
    ) -> dict[str, float]:
        day = np.datetime64(day, "D")
        eh, kh, ah, dh = self._peek(home, league, stage, day)
        ea, ka, aa, da = self._peek(away, league, stage, day)
        hfa = self._hfa(league, neutral)
        mu_h, mu_a = self._mu(league)
        lam_h = math.exp(mu_h + ah - da)
        lam_a = math.exp(mu_a + aa - dh)
        if neutral:
            lam_h = lam_a = math.sqrt(lam_h * lam_a)
        lh = self.league_of.get(home, league)
        la = self.league_of.get(away, league)
        sh = self.league_strength(lh, day)[0] if lh not in EUROPEAN_LEAGUES else math.nan
        sa = self.league_strength(la, day)[0] if la not in EUROPEAN_LEAGUES else math.nan
        return {
            "g_elo_home": eh,
            "g_elo_away": ea,
            "g_elo_diff": eh - ea,
            "g_hfa": hfa,
            "g_exp_home": elo_expectation(eh - ea + hfa),
            "g_off_home": self.offset(home) if home in self.r else math.nan,
            "g_off_away": self.offset(away) if away in self.r else math.nan,
            "g_lstr_home": sh,
            "g_lstr_away": sa,
            "g_lstr_diff": sh - sa,
            "g_games_home": float(self.games.get(home, 0)),
            "g_games_away": float(self.games.get(away, 0)),
            "g_seed_home": SEED_CODES[kh] if self.games.get(home, 0) < self.config.provisional_games else 0.0,
            "g_seed_away": SEED_CODES[ka] if self.games.get(away, 0) < self.config.provisional_games else 0.0,
            "g_lam_home": lam_h,
            "g_lam_away": lam_a,
            "g_att_home": ah,
            "g_def_home": dh,
            "g_att_away": aa,
            "g_def_away": da,
        }

    def confidence(self, team: str) -> str:
        """"low" for an unseen or provisional club (seeded from a prior, few games)."""
        if team not in self.r:
            return "low"
        provisional = self.games.get(team, 0) < self.config.provisional_games
        return "low" if provisional and self.seed_kind.get(team) != "clubelo" else "high"

    # ----------------------------------------------------------------- updating
    def _k(self, team: str, european: bool) -> float:
        cfg = self.config
        k = cfg.k_euro if european else cfg.k
        if self.games.get(team, 0) < cfg.provisional_games and self.seed_kind.get(team) != "clubelo":
            k *= cfg.provisional_factor
        return k

    def update(self, m: GlobalMatch) -> None:
        european = m.league in EUROPEAN_LEAGUES
        self.ensure(m.home, m.league, m.stage, m.date)
        self.ensure(m.away, m.league, m.stage, m.date)
        self._update_elo(m, european)
        self._update_goals(m, european)
        for team in (m.home, m.away):
            self.games[team] = self.games.get(team, 0) + 1
            self.last_date[team] = np.datetime64(m.date, "D")
            if not european:
                self.league_of[team] = m.league
                self.members.setdefault(m.league, set()).add(team)

    def _update_elo(self, m: GlobalMatch, european: bool) -> None:
        cfg = self.config
        eh, ea = self.elo(m.home), self.elo(m.away)
        expected = elo_expectation(eh - ea + self._hfa(m.league, m.neutral))
        hg, ag = m.home_goals, m.away_goals
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        err = (score - expected) * margin_multiplier(int(hg - ag), cfg.mov)
        self.r[m.home] += self._k(m.home, european) * err
        self.r[m.away] -= self._k(m.away, european) * err
        if european:
            ch, ca = self.country(m.home), self.country(m.away)
            if ch != ca:
                if ch:
                    self.off[ch] = self.off.get(ch, 0.0) + cfg.k_country * err
                if ca:
                    self.off[ca] = self.off.get(ca, 0.0) - cfg.k_country * err
        if not m.neutral:
            key = EURO_KEY if european else m.league
            self.hfa[key] = self._hfa(m.league, False) + cfg.k_hfa * (score - expected)

    def _update_goals(self, m: GlobalMatch, european: bool) -> None:
        cfg = self.config
        mu = self._mu(m.league)
        ah, dh, aa, da = self._att(m.home), self._def(m.home), self._att(m.away), self._def(m.away)
        lam_h, lam_a = math.exp(mu[0] + ah - da), math.exp(mu[1] + aa - dh)
        if m.neutral:
            lam_h = lam_a = math.sqrt(lam_h * lam_a)
        err_h, err_a = m.home_goals - lam_h, m.away_goals - lam_a
        lr = cfg.goal_lr
        self.att[m.home] += lr * err_h
        self.dfn[m.away] -= lr * err_h
        self.att[m.away] += lr * err_a
        self.dfn[m.home] -= lr * err_a
        if european:
            ch, ca = self.country(m.home), self.country(m.away)
            if ch != ca:
                lc = cfg.goal_country_lr
                if ch:
                    self.catt[ch] = self.catt.get(ch, 0.0) + lc * err_h
                    self.cdef[ch] = self.cdef.get(ch, 0.0) - lc * err_a
                if ca:
                    self.catt[ca] = self.catt.get(ca, 0.0) + lc * err_a
                    self.cdef[ca] = self.cdef.get(ca, 0.0) - lc * err_h
        if not m.neutral:
            key = EURO_KEY if european else m.league
            new = [mu[0] + cfg.goal_league_lr * err_h, mu[1] + cfg.goal_league_lr * err_a]
            self.mu[key] = new

    def _new_season(self, day: np.datetime64) -> None:
        stamp = pd.Timestamp(day)
        season = stamp.year if stamp.month >= 7 else stamp.year - 1
        if self.season is None:
            self.season = season
            return
        if season == self.season:
            return
        self.season = season
        shrink = self.config.season_regress
        if shrink <= 0:
            return
        for league in list(self.members):
            active = self._active(league, day)
            if len(active) < SEED_MIN_TEAMS:
                continue
            mean = float(np.mean([self.elo(t) for t in active]))
            for team in active:
                self.r[team] -= shrink * (self.elo(team) - mean)
        self._league_cache.clear()

    # -------------------------------------------------------------------- batch
    def build_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Pre-match features for every row (same index), applying results day by day.

        Rows without goals get features but do not update. Needs ``stage`` / ``neutral``
        columns (defaults: domestic, not neutral)."""
        n = len(matches)
        dates = matches["date"].to_numpy(dtype="datetime64[D]")
        home, away, league = (matches[c].to_numpy() for c in ("home", "away", "league"))
        hg = matches["home_goals"].to_numpy(dtype=float)
        ag = matches["away_goals"].to_numpy(dtype=float)
        stage = matches["stage"].to_numpy() if "stage" in matches else np.array(["dom"] * n, dtype=object)
        neutral = matches["neutral"].to_numpy(dtype=float) > 0 if "neutral" in matches else np.zeros(n, bool)
        rows: list[dict[str, float] | None] = [None] * n
        pending: list[GlobalMatch] = []
        current = None
        for i in np.argsort(dates, kind="stable"):
            if dates[i] != current:
                self._apply(pending)
                pending, current = [], dates[i]
                self._new_season(current)
                self._league_cache.clear()
            st = str(stage[i]) if stage[i] is not None else "dom"
            rows[i] = self.features_for(home[i], away[i], league[i], dates[i], st, bool(neutral[i]))
            for team in (home[i], away[i]):
                self.ensure(team, league[i], st, dates[i])  # seed from pre-day state
            if not (math.isnan(hg[i]) or math.isnan(ag[i])):
                pending.append(GlobalMatch(home[i], away[i], league[i], dates[i], hg[i], ag[i], st, bool(neutral[i])))
        self._apply(pending)
        self._league_cache.clear()
        return pd.DataFrame(rows, index=matches.index)

    def _apply(self, matches: list[GlobalMatch]) -> None:
        for m in matches:
            self.update(m)

    # ------------------------------------------------------------------ queries
    def table(self, teams: Mapping[str, str] | None = None) -> pd.DataFrame:
        """Current ratings as a frame (team, country, elo, games, seed)."""
        names = list(teams) if teams is not None else list(self.r)
        rows = [
            {"team": t, "country": self.country(t), "elo": self.elo(t), "games": self.games.get(t, 0),
             "seed": self.seed_kind.get(t)}
            for t in names if t in self.r
        ]  # fmt: skip
        return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
