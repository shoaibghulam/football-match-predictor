from datetime import date

import pandas as pd
import pytest
from mp_testkit import FIXTURES

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.ingest import football_data_uk as uk
from matchpredict.ingest.http import FetchError


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_season_code_switches_in_july():
    assert uk.season_code(date(2026, 10, 20)) == "2627"
    assert uk.season_code(date(2027, 3, 1)) == "2627"
    assert uk.season_code(date(2026, 6, 30)) == "2526"
    assert uk.season_code(date(1999, 8, 1)) == "9900"


def test_urls():
    assert uk.results_url("E0", "2627").endswith("/mmz4281/2627/E0.csv")
    assert uk.new_league_url("JAP").endswith("/new/JPN.csv")
    assert uk.new_league_url("SUI").endswith("/new/SWZ.csv")


def test_parse_fixtures_csv_with_bom():
    df = uk.parse_main_csv(_read("fixtures.csv"))
    assert tuple(df.columns) == MATCH_COLUMNS
    assert len(df) == 5
    row = df[df["home"] == "Wolves"].iloc[0]
    assert row["league"] == "E1"
    assert row["date"] == pd.Timestamp("2026-10-20")
    assert row["time"] == "19:45"
    assert (row["odds_home"], row["odds_draw"], row["odds_away"]) == (1.53, 4.1, 5.25)
    assert df["home_goals"].isna().all()
    assert df["date"].dtype == "datetime64[ns]"


def test_parse_results_csv_reads_stats_xg_and_prematch_odds():
    df = uk.parse_main_csv(_read("E0.csv"))
    assert len(df) == 3
    row = df[(df["home"] == "Tottenham")].iloc[0]
    assert (row["home_goals"], row["away_goals"]) == (3.0, 1.0)
    assert (row["home_shots"], row["away_shots"], row["home_target"], row["away_target"]) == (
        16, 12, 5, 1,
    )  # fmt: skip
    assert (row["home_corners"], row["away_corners"]) == (4, 4)
    assert row["home_xg"] == pytest.approx(0.87)
    # Bet365 pre-match (1.45) not closing (B365CH=1.55)
    assert row["odds_home"] == pytest.approx(1.45)


def test_odds_fall_back_to_average_when_bet365_missing():
    csv = "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA,B365CH,B365CD,B365CA\n"
    csv += "E0,01/11/2026,15:00,A,B,,,,2.1,3.2,3.5,9,9,9\n"
    csv += "E0,01/11/2026,15:00,C,D,,,,,,,9,9,9\n"
    df = uk.parse_main_csv(csv)
    a = df[df["home"] == "A"].iloc[0]
    assert (a["odds_home"], a["odds_draw"], a["odds_away"]) == (2.1, 3.2, 3.5)
    c = df[df["home"] == "C"].iloc[0]
    assert pd.isna(c["odds_home"])  # closing odds never used


def test_unknown_divisions_and_blank_rows_dropped():
    csv = "Div,Date,Time,HomeTeam,AwayTeam\nXX,01/11/2026,15:00,A,B\nE0,01/11/26,,C,D\n,,,,\n"
    df = uk.parse_main_csv(csv)
    assert df[["league", "home"]].values.tolist() == [["E0", "C"]]
    assert df.iloc[0]["date"] == pd.Timestamp("2026-11-01")
    assert df.iloc[0]["time"] is None


def test_empty_and_wrong_layout_return_empty_frame():
    assert uk.parse_main_csv(b"").empty
    assert uk.parse_main_csv("Foo,Bar\n1,2\n").empty
    assert tuple(uk.parse_new_league_csv("Foo\n1\n").columns) == MATCH_COLUMNS


def test_latin1_bytes_are_decoded():
    csv = "Div,Date,Time,HomeTeam,AwayTeam\nSP1,01/11/2026,15:00,Alavés,Betis\n".encode("latin-1")
    assert uk.parse_main_csv(csv).iloc[0]["home"] == "Alavés"


def test_parse_new_league_fixtures_maps_country_dedupes_and_skips_unknown():
    df = uk.parse_new_league_csv(_read("new_league_fixtures.csv"))
    assert set(df["league"]) == {"FIN", "JAP", "USA"}
    assert len(df[(df["home"] == "Kawasaki Frontale")]) == 1  # duplicate row removed
    fin = df[df["league"] == "FIN"].iloc[0]
    assert (fin["home"], fin["away"]) == ("PK-35 Vantaa", "Jaro")  # aliased to history name
    assert fin["odds_home"] == pytest.approx(2.75)  # B365 pre-match
    assert df["home_goals"].isna().all()


def test_parse_new_league_results_have_no_closing_odds():
    df = uk.parse_new_league_csv(_read("ARG.csv"))
    assert set(df["league"]) == {"ARG"}
    assert df["home_goals"].notna().all()
    assert df["odds_home"].isna().all()
    assert (df["home"] == "Arsenal Sarandi").any()  # trailing spaces stripped from League ok


def _fake_fetch(mapping: dict[str, bytes]):
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        for suffix, content in mapping.items():
            if url.endswith(suffix):
                return content
        raise FetchError(f"404 for {url}")

    return fetch, calls


def test_fetch_fixtures_combines_both_files():
    fetch, calls = _fake_fetch(
        {"/fixtures.csv": _read("fixtures.csv"), "/new_league_fixtures.csv": _read("new_league_fixtures.csv")}
    )
    result = uk.fetch_fixtures(fetch=fetch, sleep=lambda _: None)
    assert len(calls) == 2
    assert not result.errors
    assert {"E1", "SP1", "JAP", "USA"} <= set(result.frame["league"])
    assert tuple(result.frame.columns) == MATCH_COLUMNS


def test_fetch_results_collects_errors_and_filters_old_new_league_rows():
    fetch, calls = _fake_fetch({"/2627/E0.csv": _read("E0.csv"), "/new/ARG.csv": _read("ARG.csv")})
    result = uk.fetch_results(
        date(2026, 10, 20),
        divisions=["E0", "E1"],
        new_leagues=["ARG"],
        lookback_days=30,
        fetch=fetch,
        sleep=lambda _: None,
    )
    assert len(calls) == 3
    assert len(result.errors) == 1 and "E1" in result.errors[0]
    assert set(result.frame["league"]) == {"E0", "ARG"}
    arg = result.frame[result.frame["league"] == "ARG"]
    assert arg["date"].min() >= pd.Timestamp("2026-09-20")  # 2012 row filtered out
    assert result.frame["home_goals"].notna().all()


def test_fetch_results_records_parse_errors(monkeypatch):
    def boom(_content):
        raise ValueError("bad csv")

    monkeypatch.setattr(uk, "parse_main_csv", boom)
    result = uk.fetch_results(
        date(2026, 10, 20), divisions=["E0"], new_leagues=[], fetch=lambda u: b"x", sleep=lambda _: None
    )
    assert result.frame.empty
    assert "parse error" in result.errors[0]


def test_team_aliases_map_to_history_spelling():
    csv = "Country,League,Date,Time,Home,Away,HG,AG\n"
    csv += "Mexico,Liga MX,18/10/2026,00:07,Guadalajara Chivas,Tigres UANL,1,1\n"
    csv += "Mexico,Liga MX,19/10/2026,00:07,UNAM Pumas,Atlas,2,0\n"
    df = uk.parse_new_league_csv(csv)
    assert df["away"].tolist()[0] == "U.A.N.L.- Tigres"
    assert df["home"].tolist()[1] == "U.N.A.M.- Pumas"


def test_fetch_results_resumes_extra_league_from_since():
    fetch, _ = _fake_fetch({"/new/ARG.csv": _read("ARG.csv")})
    result = uk.fetch_results(
        date(2026, 10, 20),
        divisions=[],
        new_leagues=["ARG"],
        lookback_days=30,
        since={"ARG": date(2012, 1, 1)},
        fetch=fetch,
        sleep=lambda _: None,
    )
    assert result.frame["date"].min() == pd.Timestamp("2012-08-03")
