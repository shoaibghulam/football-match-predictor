"""Global rating: leakage, cross-league linkage, seeding, home advantage, persistence."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
import pytest

from matchpredict.predictor.v2.global_rating import (
    ClubEloBook,
    GlobalConfig,
    GlobalMatch,
    GlobalRating,
    elo_expectation,
    margin_multiplier,
)

COUNTRIES = {**{f"A{i}": "ENG" for i in range(12)}, **{f"B{i}": "ESP" for i in range(12)}, "Z1": "CRO", "Z2": "CRO"}


def matches(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, league, home, away, home_goals, away_goals[, stage])"""
    out = []
    for r in rows:
        stage = r[6] if len(r) > 6 else ("dom" if r[1] not in ("UCL", "UEL", "UECL") else "league")
        out.append({"date": pd.Timestamp(r[0]), "league": r[1], "home": r[2], "away": r[3],
                    "home_goals": float(r[4]), "away_goals": float(r[5]), "stage": stage,
                    "neutral": float(stage == "final")})  # fmt: skip
    return pd.DataFrame(out)


def season(league: str, prefix: str, start: str, n: int = 10, rng=None, bias: float = 0.0) -> list[tuple]:
    """Round robin of n clubs; results drawn from a fair coin (+bias for the home side)."""
    rng = rng or np.random.default_rng(0)
    teams = [f"{prefix}{i}" for i in range(n)]
    rows, day = [], pd.Timestamp(start)
    for i, h in enumerate(teams):
        for a in teams[i + 1 :]:
            draw = rng.random()
            hg, ag = (1, 0) if draw < 0.4 + bias else (0, 0) if draw < 0.7 else (0, 1)
            rows.append((day, league, h, a, hg, ag))
            day += pd.Timedelta(days=1)
    return rows


def test_expectation_and_margin_multiplier():
    assert elo_expectation(0) == pytest.approx(0.5)
    assert elo_expectation(400) == pytest.approx(10 / 11)
    assert margin_multiplier(1, "clubelo") == 1.0
    assert margin_multiplier(2, "clubelo") == 1.5
    assert margin_multiplier(4, "clubelo") == pytest.approx(15 / 8)
    assert margin_multiplier(3, "log") == pytest.approx(1 + math.log(3))
    assert margin_multiplier(5, "none") == 1.0


def test_own_result_and_same_day_results_are_never_visible():
    rows = [("2020-01-01", "E0", "A0", "A1", 1, 1), ("2020-01-08", "E0", "A0", "A2", 5, 0),
            ("2020-01-08", "E0", "A0", "A3", 4, 0), ("2020-01-15", "E0", "A0", "A1", 0, 0)]  # fmt: skip
    base = GlobalRating(countries=COUNTRIES).build_features(matches(rows))
    flipped = [*rows[:1], ("2020-01-08", "E0", "A0", "A2", 0, 5), ("2020-01-08", "E0", "A0", "A3", 0, 4), rows[3]]
    other = GlobalRating(countries=COUNTRIES).build_features(matches(flipped))
    # the two same-day rows read the same pre-day state, whatever their results
    assert base.loc[1, "g_elo_home"] == base.loc[2, "g_elo_home"]
    pd.testing.assert_frame_equal(base.iloc[:3], other.iloc[:3])
    # the next day sees both results
    assert base.loc[3, "g_elo_home"] > other.loc[3, "g_elo_home"]


def test_upcoming_rows_get_features_but_do_not_update():
    frame = matches([("2020-01-01", "E0", "A0", "A1", 2, 0)])
    frame.loc[1] = {"date": pd.Timestamp("2020-01-08"), "league": "E0", "home": "A1", "away": "A0",
                    "home_goals": np.nan, "away_goals": np.nan, "stage": "dom", "neutral": 0.0}  # fmt: skip
    rating = GlobalRating(countries=COUNTRIES)
    feats = rating.build_features(frame)
    assert feats.loc[1, "g_elo_away"] > feats.loc[1, "g_elo_home"]
    assert rating.games["A0"] == 1


def test_european_results_link_league_scales():
    """ENG clubs beating ESP clubs in Europe lifts *all* ENG clubs (country offset)."""
    rng = np.random.default_rng(1)
    rows = season("E0", "A", "2019-08-01", rng=rng) + season("SP1", "B", "2019-08-01", rng=rng)
    day = pd.Timestamp("2019-10-01")
    for i in range(30):
        rows.append((day + pd.Timedelta(days=i), "UCL", f"A{i % 3}", f"B{i % 3}", 2, 0))
    rows += [("2020-02-01", "E0", "A9", "A8", 0, 0), ("2020-02-01", "SP1", "B9", "B8", 0, 0)]
    rating = GlobalRating(config=GlobalConfig(seed_clubelo=False), countries=COUNTRIES)
    feats = rating.build_features(matches(rows))
    assert rating.off["ENG"] > 0 > rating.off["ESP"]
    # clubs that never played in Europe moved too
    assert rating.elo("A9") - rating.elo("B9") > 50
    last = feats.iloc[-2:]
    assert (last["g_lstr_home"].iloc[0] - last["g_lstr_home"].iloc[1]) > 50
    without = GlobalRating(config=GlobalConfig(seed_clubelo=False, k_country=0.0), countries=COUNTRIES)
    without.build_features(matches(rows))
    assert rating.elo("A9") - rating.elo("B9") > without.elo("A9") - without.elo("B9")


def test_goal_country_offsets_move_with_european_goals():
    rows = [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "UCL", "A0", "B0", 4, 0) for i in range(10)]
    rating = GlobalRating(countries=COUNTRIES)
    rating.build_features(matches(rows))
    assert rating.catt["ENG"] > 0 and rating.cdef["ESP"] < 0
    assert rating._att("A0") > rating._att("B0")


def test_seeding_league_bottom_country_prior_and_stage_prior():
    rows = season("E0", "A", "2019-08-01", n=10)
    rating = GlobalRating(config=GlobalConfig(seed_clubelo=False), countries=COUNTRIES)
    rating.build_features(matches(rows))
    day = np.datetime64("2019-10-01")
    elo, kind = rating.seed("A11", "E0", "dom", day)
    bottom = sorted(rating.elo(f"A{i}") for i in range(10))[:4]
    assert kind == "league" and elo == pytest.approx(np.mean(bottom))
    # club of a rated country appearing only in Europe -> bottom of its country's league
    elo_c, kind_c = rating.seed("A11", "UEL", "league", day)
    assert kind_c == "country" and elo_c == pytest.approx(elo)
    # club of an unrated country: stage priors; then its country's mean minus the newcomer gap
    assert rating.seed("Z1", "UCL", "qual", day) == (1300.0, "stage")
    assert rating.seed("Z1", "UCL", "ko", day) == (1580.0, "stage")
    rating.update(GlobalMatch("Z1", "A0", "UCL", day, 1, 0, "league"))
    elo_z, kind_z = rating.seed("Z2", "UCL", "league", day + np.timedelta64(1, "D"))
    assert kind_z == "country" and elo_z == pytest.approx(rating.elo("Z1") - rating.config.newcomer_gap)


def test_clubelo_seed_uses_last_snapshot_strictly_before_the_day():
    snaps = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "club": ["A0", "A0"], "elo": [1700.0, 1800.0]})
    book = ClubEloBook(snaps)
    assert math.isnan(book.get("A0", np.datetime64("2020-01-01")))
    assert book.get("A0", np.datetime64("2020-01-02")) == 1700.0
    assert book.get("A0", np.datetime64("2020-02-01")) == 1700.0
    assert book.get("A0", np.datetime64("2020-02-02")) == 1800.0
    assert math.isnan(book.get("nobody", np.datetime64("2020-02-02")))
    assert "A0" in book and book.last_snapshot == pd.Timestamp("2020-02-01")
    rating = GlobalRating(countries=COUNTRIES, clubelo=book)
    feats = rating.build_features(matches([("2020-02-01", "E0", "A0", "A1", 0, 0)]))
    assert feats.loc[0, "g_elo_home"] == 1700.0 and rating.seed_kind["A0"] == "clubelo"
    assert rating.confidence("A0") == "high"
    assert rating.confidence("nobody") == "low"


def test_home_advantage_is_learned_per_league_and_neutral_finals_have_none():
    rows = [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "E0", f"A{i % 2}", f"A{1 - i % 2}", 2, 0) for i in range(40)]
    rating = GlobalRating(countries=COUNTRIES)
    rating.build_features(matches(rows))
    assert rating.hfa["E0"] > rating.config.hfa_init
    assert rating._hfa("UCL", neutral=True) == 0.0
    final = rating.features_for("A0", "A1", "UCL", np.datetime64("2020-06-01"), "final", neutral=True)
    assert final["g_hfa"] == 0.0 and final["g_lam_home"] == pytest.approx(final["g_lam_away"])


def test_state_pickles_and_table_lists_ratings():
    rating = GlobalRating(countries=COUNTRIES)
    rating.build_features(matches([("2020-01-01", "E0", "A0", "A1", 1, 0)]))
    clone = pickle.loads(pickle.dumps(rating))
    assert clone.elo("A0") == rating.elo("A0")
    assert clone.league_strength("E0", np.datetime64("2020-01-02"))  # cache rebuilt lazily
    table = rating.table()
    assert list(table["team"]) == ["A0", "A1"]


def test_season_regression_shrinks_toward_the_league_mean():
    rows = season("E0", "A", "2019-08-01", n=10, bias=0.3)
    rows.append(("2020-08-01", "E0", "A0", "A1", 0, 0))
    plain = GlobalRating(config=GlobalConfig(seed_clubelo=False), countries=COUNTRIES)
    plain.build_features(matches(rows))
    shrunk = GlobalRating(config=GlobalConfig(seed_clubelo=False, season_regress=0.5), countries=COUNTRIES)
    shrunk.build_features(matches(rows))
    spread = lambda r: np.std([r.elo(f"A{i}") for i in range(10)])  # noqa: E731
    assert spread(shrunk) < spread(plain)
