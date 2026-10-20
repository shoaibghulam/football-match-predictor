import numpy as np
import pandas as pd
import pytest

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.predictor.data import load_history, outcomes, season_of, season_start

RAW_HEADER = (
    "Division,MatchDate,MatchTime,HomeTeam,AwayTeam,HomeElo,AwayElo,FTHome,FTAway,FTResult,"
    "HomeShots,AwayShots,HomeTarget,AwayTarget,HomeCorners,AwayCorners,OddHome,OddDraw,OddAway\n"
)
RAW_ROWS = [
    "E0,2020-01-02,15:00:00,C,D,1500,1500,0,2,A,10,12,3,5,4,6,2.5,3.2,2.9",
    "E0,2020-01-01,,A,B,1500,1500,2,1,H,,,,,,,1.9,3.4,4.0",
    "E0,2020-01-02,12:30:00,A,C,1500,1500,1,1,D,8,9,2,3,1,2,,,",
    "E0,2020-01-03,,B,D,1500,1500,,,,,,,,,,2.0,3.0,3.5",
]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "Matches.csv"
    path.write_text(RAW_HEADER + "\n".join(RAW_ROWS) + "\n")
    return path


def test_load_history_has_exact_contract_columns(csv_path):
    df = load_history(csv_path)
    assert tuple(df.columns) == MATCH_COLUMNS


def test_load_history_drops_rows_without_result_and_sorts(csv_path):
    df = load_history(csv_path)
    assert len(df) == 3
    assert list(df["home"]) == ["A", "A", "C"]  # 01-01, then 01-02 12:30 before 15:00
    assert df.index.tolist() == [0, 1, 2]


def test_load_history_maps_values(csv_path):
    df = load_history(csv_path)
    last = df.iloc[2]
    assert df["date"].dtype.kind == "M"
    assert last["league"] == "E0"
    assert last["time"] == "15:00"
    assert df.iloc[0]["time"] is None
    assert last["home_goals"] == 0 and last["away_goals"] == 2
    assert last["home_target"] == 3 and last["away_corners"] == 6
    assert last["odds_home"] == pytest.approx(2.5)
    assert df["home_xg"].isna().all()
    assert np.isnan(df.iloc[1]["odds_home"])


def test_outcomes_encoding():
    df = pd.DataFrame({"home_goals": [2, 1, 0], "away_goals": [1, 1, 3]})
    np.testing.assert_array_equal(outcomes(df), [0, 1, 2])


def test_season_helpers():
    dates = pd.Series(pd.to_datetime(["2024-06-30", "2024-07-01", "2025-01-15"]))
    np.testing.assert_array_equal(season_of(dates), [2023, 2024, 2024])
    assert season_start(2024) == pd.Timestamp("2024-07-01")
