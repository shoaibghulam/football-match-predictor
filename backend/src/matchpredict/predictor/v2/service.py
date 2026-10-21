"""``V2Service``: serves domestic and UEFA predictions from one v2 artifact.

Implements ``contracts.PredictionService`` plus the extras the API uses for UEFA fixtures
(``resolve``, ``strength``, keyword ``stage`` / ``neutral``) and a ``confidence`` field.

The artifact (``model_v2.joblib``) is immutable; daily results only change the rating
state, which is saved to a small sidecar (``model_v2.joblib.state``) and re-applied on load.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from matchpredict.contracts import ExpectedGoals, Prediction, Probs, ScoreProb
from matchpredict.ingest.leagues import EUROPEAN_LEAGUES
from matchpredict.ingest.teamnames import TeamNameMapper
from matchpredict.predictor.data import sort_matches
from matchpredict.predictor.europe import infer_stage
from matchpredict.predictor.features import valid_odds
from matchpredict.predictor.goals import (
    btts,
    expected_goals,
    over_line,
    rescale_to_1x2,
    top_scores,
)
from matchpredict.predictor.v2.data import FRAME_COLUMNS, canonicalise, standardise
from matchpredict.predictor.v2.features import FeatureState, build_table, fixture_row
from matchpredict.predictor.v2.train import ARTIFACT_VERSION, ensemble_probs

log = logging.getLogger(__name__)

DECIMALS = 4
TOP_SCORES = 5
RECENT_DAYS = 10  # late / partial-day results within this window are still applied once
STATE_SUFFIX = ".state"


def _round_probs(probs: np.ndarray) -> Probs:
    home, draw = round(float(probs[0]), DECIMALS), round(float(probs[1]), DECIMALS)
    return Probs(home=home, draw=draw, away=round(1.0 - home - draw, DECIMALS))


def _load_pickle(path: Path):
    """joblib.load that also reads artifacts pickled by ``python -m ...v2.train``.

    Those store the training dataclasses as ``__main__.<Class>``; map them back to the module.
    """
    from matchpredict.predictor.v2 import train as train_module

    main = sys.modules.get("__main__")
    classes = {
        name: obj
        for name, obj in vars(train_module).items()
        if isinstance(obj, type) and obj.__module__ == train_module.__name__
    }
    added = [name for name in classes if main is not None and not hasattr(main, name)]
    for name in added:
        setattr(main, name, classes[name])
    try:
        return joblib.load(path)
    finally:
        for name in added:
            delattr(main, name)


def _key(league: str, day, home: str, away: str) -> tuple[str, str, str, str]:
    return (str(league), str(pd.Timestamp(day).date()), str(home), str(away))


class V2Service:
    """Thread-safe (one internal lock) v2 predictor."""

    update_lookback_days = 2 * RECENT_DAYS  # ingest fetches this far back; dedupe skips what is applied

    def __init__(self, artifact: dict, artifact_path: Path | None = None) -> None:
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("not a MatchPredict v2 artifact")
        self.artifact = artifact
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self.model_version: str = artifact["model_version"]
        self.state: FeatureState = artifact["state"]
        self.names = artifact["names"]
        self.members = artifact["members"]
        self.pools = artifact["pools"]
        self.config = artifact["config"]
        self.gate: dict = dict(artifact.get("gate") or {})
        self.validation: dict = dict(artifact.get("validation") or {})
        self.last_result_date: pd.Timestamp | None = artifact.get("last_result_date")
        self.league_last_date: dict[str, pd.Timestamp] = dict(artifact.get("league_last_date") or {})
        self.applied: set[tuple[str, str, str, str]] = set(artifact.get("applied_keys") or ())
        # results on or before a league's last training day are already in the ratings
        self.train_cutoff: dict[str, pd.Timestamp] = dict(artifact.get("league_last_date") or {})
        self._lock = threading.RLock()
        self._mapper: TeamNameMapper | None = None

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: str | Path) -> V2Service:
        path = Path(path)
        service = cls(_load_pickle(path), artifact_path=path)
        sidecar = path.with_name(path.name + STATE_SUFFIX)
        if sidecar.exists():
            try:
                service._restore(_load_pickle(sidecar))
            except Exception:
                log.exception("ignoring unreadable v2 state sidecar %s", sidecar)
        return service

    def _restore(self, saved: dict) -> None:
        if saved.get("model_version") != self.model_version:
            log.info("v2 state sidecar is for %s, artifact is %s; ignoring", saved.get("model_version"), self.model_version)
            return
        clubelo = self.state.clubelo
        self.state = saved["state"]
        self.state.clubelo = clubelo
        self.last_result_date = saved.get("last_result_date")
        self.league_last_date = dict(saved.get("league_last_date") or {})
        self.applied = set(saved.get("applied") or ())

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the live rating state next to the artifact (the artifact itself is not rewritten)."""
        target = Path(path) if path is not None else self.artifact_path
        if target is None:
            raise ValueError("no artifact path to save the v2 state next to")
        sidecar = target.with_name(target.name + STATE_SUFFIX)
        with self._lock:
            state = FeatureState(rating=self.state.rating, domestic=self.state.domestic, legs=self.state.legs)
            payload = {"model_version": self.model_version, "state": state, "applied": sorted(self.applied),
                       "last_result_date": self.last_result_date, "league_last_date": dict(self.league_last_date)}  # fmt: skip
            tmp = sidecar.with_suffix(".tmp")
            joblib.dump(payload, tmp, compress=3)
            tmp.replace(sidecar)
        return sidecar

    # ------------------------------------------------------------------ names
    def _known(self, team: str) -> bool:
        return team in self.state.rating.r or team in self.state.domestic.played

    def resolve(self, name: str, country: str | None = None) -> str | None:
        canonical = self.names.canonical(name)
        if self._known(canonical):
            return canonical
        folded = canonical.casefold()
        for team in self.state.rating.r:
            if team.casefold() == folded:
                return team
        if self._mapper is None or len(self._mapper.candidates) < len(self.state.rating.r):
            self._mapper = TeamNameMapper({t: self.state.rating.country(t) for t in self.state.rating.r})
        hit = self._mapper.map(name, country)
        return self.names.canonical(hit) if hit else None

    def known_teams(self) -> dict[str, str]:
        out = dict(self.state.domestic.team_league)
        for team in self.state.rating.r:
            out.setdefault(team, "UCL")
        return out

    def strength(self, team: str) -> float | None:
        resolved = self.resolve(team)
        return None if resolved is None else float(self.state.rating.elo(resolved))

    def team_ratings(self, team: str) -> dict[str, float]:
        resolved = self.resolve(team)
        if resolved is None:
            raise KeyError(f"unknown team: {team!r}")
        eng, rating = self.state.domestic, self.state.rating
        pi_home, pi_away = eng.pi.get(resolved, (0.0, 0.0))
        return {
            "elo": float(rating.elo(resolved)),
            "country_offset": float(rating.offset(resolved)),
            "pi_home": float(pi_home),
            "pi_away": float(pi_away),
            "form5": float(sum(points for points, _ in eng.form.get(resolved, ()))),
            "attack": float(rating._att(resolved)),
            "defence": float(rating._def(resolved)),
            "played": float(rating.games.get(resolved, 0)),
        }

    # ------------------------------------------------------------- predicting
    def _side(self, name: str, league: str, allow_unknown: bool) -> str:
        resolved = self.resolve(name)
        if resolved is not None:
            return resolved
        known_league = league in self.state.domestic.league_teams or league in EUROPEAN_LEAGUES
        if allow_unknown or known_league:
            return self.names.canonical(name)  # seeded from league / country / stage priors
        raise KeyError(f"unknown team {name!r} in unknown league {league!r}")

    def predict(
        self,
        home: str,
        away: str,
        league: str,
        date: pd.Timestamp,
        odds: tuple[float, float, float] | None = None,
        *,
        stage: str | None = None,
        neutral: bool = False,
        allow_unknown: bool | None = None,
    ) -> Prediction:
        european = league in EUROPEAN_LEAGUES
        allow = european if allow_unknown is None else allow_unknown
        with self._lock:
            h = self._side(home, league, allow)
            a = self._side(away, league, allow)
            day = pd.Timestamp(date)
            st = stage or (infer_stage(league, day) if european else "dom")
            is_neutral = bool(neutral or st == "final")
            use_market = (not european) and valid_odds(odds)
            row = fixture_row(self.state, h, a, league, day, st, is_neutral, odds if use_market else None)
            probs_all = ensemble_probs(self.members, self.pools, row, self.config)
            probs = probs_all["v2_market"][0] if use_market else probs_all["v2_base"][0]
            matrix = rescale_to_1x2(self.members.goals.score_matrices(row)[0], probs)
            low = self.state.rating.confidence(h) == "low" or self.state.rating.confidence(a) == "low"
        xg_home, xg_away = expected_goals(matrix)
        return Prediction(
            probs=_round_probs(probs),
            expected_goals=ExpectedGoals(home=round(xg_home, 2), away=round(xg_away, 2)),
            over_2_5=round(over_line(matrix, 2.5), DECIMALS),
            btts=round(btts(matrix), DECIMALS),
            top_scores=[ScoreProb(score=s, p=round(p, DECIMALS)) for s, p in top_scores(matrix, TOP_SCORES)],
            model="market" if use_market else ("europe" if european else "base"),
            model_version=self.model_version,
            confidence="low" if low else "high",  # type: ignore[typeddict-unknown-key]
        )

    # --------------------------------------------------------------- updating
    def _new_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        trained = frame["league"].map(self.train_cutoff)
        recent = frame["league"].map(self.league_last_date) - pd.Timedelta(days=RECENT_DAYS)
        keys = [_key(lg, d, h, a) for lg, d, h, a in zip(frame["league"], frame["date"], frame["home"], frame["away"], strict=True)]
        fresh = np.array([k not in self.applied for k in keys], dtype=bool)
        after_training = (trained.isna() | (frame["date"] > trained)).to_numpy()
        not_stale = (recent.isna() | (frame["date"] > recent)).to_numpy()
        return frame[fresh & after_training & not_stale]

    def update_results(self, matches: pd.DataFrame) -> int:
        """Apply finished domestic + UEFA matches not applied yet; return the count applied.

        Rules: nothing on or before the league's last training day (already in the ratings); after
        that an exact (league, date, home, away) key for the last ``RECENT_DAYS`` days, so partial
        match days (live polls) and late results are applied exactly once."""
        finished = matches.dropna(subset=["home_goals", "away_goals"])
        if finished.empty:
            return 0
        with self._lock:
            frame = canonicalise(standardise(finished), self.names)
            frame = frame[frame["home"] != frame["away"]]
            frame = frame.drop_duplicates(["league", "date", "home", "away"], keep="last")
            frame = self._new_rows(frame)
            if frame.empty:
                return 0
            ordered = sort_matches(frame.reindex(columns=list(FRAME_COLUMNS)))
            build_table(ordered, self.state)
            for lg, d, h, a in zip(ordered["league"], ordered["date"], ordered["home"], ordered["away"], strict=True):
                self.applied.add(_key(lg, d, h, a))
            for league, last in ordered.groupby("league")["date"].max().items():
                prev = self.league_last_date.get(league)
                self.league_last_date[league] = pd.Timestamp(last) if prev is None else max(prev, pd.Timestamp(last))
            newest = pd.Timestamp(ordered["date"].max())
            self.last_result_date = newest if self.last_result_date is None else max(self.last_result_date, newest)
            horizon = str((newest - pd.Timedelta(days=2 * RECENT_DAYS)).date())
            self.applied = {k for k in self.applied if k[1] >= horizon}
            self._mapper = None
            return len(ordered)
