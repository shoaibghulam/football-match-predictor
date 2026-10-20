import pandas as pd
import pytest

from matchpredict.predictor.ratings import EloConfig, RatingsEngine


def _matches(rows):
    """rows: (date, league, home, away, hg, ag)."""
    return pd.DataFrame(
        rows, columns=["date", "league", "home", "away", "home_goals", "away_goals"]
    ).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_new_teams_start_at_default_elo():
    feats = RatingsEngine().build_features(_matches([("2020-01-01", "E0", "A", "B", 1, 0)]))
    assert feats.loc[0, "elo_home"] == pytest.approx(EloConfig().initial)
    assert feats.loc[0, "elo_away"] == pytest.approx(EloConfig().initial)


def test_elo_is_zero_sum_and_winner_gains():
    eng = RatingsEngine()
    eng.build_features(_matches([("2020-01-01", "E0", "A", "B", 3, 0)]))
    a, b = eng.elo["A"], eng.elo["B"]
    assert a > EloConfig().initial > b
    assert a + b == pytest.approx(2 * EloConfig().initial)


def test_features_never_see_own_result():
    base = [("2020-01-01", "E0", "A", "B", 2, 0), ("2020-01-08", "E0", "A", "B", 0, 0)]
    changed = [base[0], ("2020-01-08", "E0", "A", "B", 9, 0)]
    f1 = RatingsEngine().build_features(_matches(base))
    f2 = RatingsEngine().build_features(_matches(changed))
    pd.testing.assert_frame_equal(f1, f2)


def test_same_day_matches_do_not_see_each_other():
    rows = [("2020-01-01", "E0", "A", "B", 5, 0), ("2020-01-01", "E0", "A", "C", 0, 0)]
    feats = RatingsEngine().build_features(_matches(rows))
    assert feats.loc[1, "elo_home"] == pytest.approx(EloConfig().initial)
    assert feats.loc[1, "played_home"] == 0


def test_pi_ratings_move_in_direction_of_surprise():
    eng = RatingsEngine()
    eng.build_features(_matches([("2020-01-01", "E0", "A", "B", 3, 0)]))
    assert eng.pi["A"][0] > 0  # A's home rating up
    assert eng.pi["A"][1] > 0  # A's away rating up (cross-learning)
    assert eng.pi["B"][1] < 0  # B's away rating down
    assert eng.pi["B"][0] < 0  # B's home rating down


def test_form_and_rest_days_track_previous_matches():
    rows = [
        ("2020-01-01", "E0", "A", "B", 1, 0),
        ("2020-01-05", "E0", "C", "A", 2, 2),
        ("2020-01-12", "E0", "A", "C", 0, 0),
    ]
    feats = RatingsEngine().build_features(_matches(rows))
    last = feats.iloc[2]
    assert last["form5_home"] == 4  # win + draw
    assert last["rest_home"] == 7
    assert last["played_home"] == 2


def test_output_preserves_input_order_and_index():
    rows = [("2020-01-08", "E0", "A", "B", 1, 0), ("2020-01-01", "E0", "B", "A", 1, 0)]
    m = _matches(rows)
    m.index = [10, 20]
    feats = RatingsEngine().build_features(m)
    assert list(feats.index) == [10, 20]
    assert feats.loc[20, "played_home"] == 0
    assert feats.loc[10, "played_home"] == 1


def test_process_stats_ewma_nan_until_observed():
    m = _matches([("2020-01-01", "E0", "A", "B", 1, 0), ("2020-01-08", "E0", "A", "B", 1, 0)])
    m["home_shots"] = [10.0, None]
    m["away_shots"] = [4.0, None]
    feats = RatingsEngine().build_features(m)
    assert pd.isna(feats.loc[0, "shots_for_home"])
    assert feats.loc[1, "shots_for_home"] == pytest.approx(10.0)
    assert feats.loc[1, "shots_against_away"] == pytest.approx(10.0)


def test_goal_ratings_reward_scoring_and_punish_conceding():
    eng = RatingsEngine()
    eng.build_features(_matches([("2020-01-01", "E0", "A", "B", 4, 0)]))
    assert eng.attack["A"] > 0 and eng.defence["B"] < 0
    assert eng.attack["B"] < 0 and eng.defence["A"] > 0  # B scored fewer than expected


def test_goal_lambdas_reflect_strength():
    rows = [(f"2020-01-{d:02d}", "E0", "A", "B", 3, 0) for d in range(1, 21)]
    feats = RatingsEngine().build_features(_matches(rows))
    first, last = feats.iloc[0], feats.iloc[-1]
    assert last["gr_lambda_home"] > first["gr_lambda_home"]
    assert last["gr_lambda_away"] < first["gr_lambda_away"]
    assert last["elo_fast_diff"] > last["elo_diff"] > 0
