import datetime as dt

import pandas as pd
import pytest
from mp_testkit import FIXTURES, frame, memory_factory

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.db import repository as repo
from matchpredict.db import seed as seed_mod
from matchpredict.db.models import Match

D = dt.date


@pytest.fixture
def factory():
    return memory_factory()


def _fixture(day="2026-10-24", home="Arsenal", away="Chelsea", **kw):
    return {"date": day, "league": "E0", "home": home, "away": away, "time": "15:00", **kw}


def test_upsert_inserts_then_updates_scheduled_to_finished(factory):
    with factory() as s:
        stats = repo.upsert_matches(s, frame([_fixture(odds_home=2.0, odds_draw=3.4, odds_away=3.6)]))
        assert stats == repo.UpsertStats(inserted=1, updated=0)
        m = s.query(Match).one()
        assert m.status == "scheduled" and m.home_goals is None

    result = _fixture(home_goals=2, away_goals=1, home_xg=1.7, away_xg=0.4, time=None)
    with factory() as s:
        assert repo.upsert_matches(s, frame([result]), source="results") == repo.UpsertStats(0, 1)
        m = s.query(Match).one()
        assert (m.status, m.home_goals, m.away_goals, m.home_xg) == ("finished", 2, 1, 1.7)
        assert m.odds_home == 2.0  # nulls never erase existing values
        assert m.time == "15:00"
        assert m.source == "results"


def test_upsert_is_idempotent_and_never_downgrades_finished(factory):
    finished = frame([_fixture(home_goals=1, away_goals=1)])
    with factory() as s:
        repo.upsert_matches(s, finished)
        assert repo.upsert_matches(s, finished) == repo.UpsertStats(0, 0)
        repo.upsert_matches(s, frame([_fixture(odds_home=1.9, odds_draw=3.5, odds_away=4.0)]))
        m = s.query(Match).one()
        assert m.status == "finished" and m.odds_home == 1.9


def test_upsert_dedupes_and_handles_empty(factory):
    df = frame([_fixture(), _fixture(odds_home=2.5, odds_draw=3.0, odds_away=3.0)])
    with factory() as s:
        assert repo.upsert_matches(s, df).inserted == 1
        assert repo.upsert_matches(s, frame([])) == repo.UpsertStats()
        assert s.query(Match).one().odds_home == 2.5


def test_fixtures_between_filters_and_orders(factory):
    rows = [
        _fixture("2026-10-24", "Arsenal", "Chelsea", time="17:30"),
        _fixture("2026-10-24", "Coventry", "Leeds", time="12:30"),
        _fixture("2026-10-25", "Leeds", "Arsenal"),
        {"date": "2026-10-24", "league": "F1", "home": "Lyon", "away": "Nice"},
        _fixture("2026-11-20", "Chelsea", "Arsenal"),
    ]
    with factory() as s:
        repo.upsert_matches(s, frame(rows))
        got = repo.fixtures_between(s, D(2026, 10, 24), D(2026, 10, 25))
        assert [(m.home, m.time) for m in got] == [
            ("Coventry", "12:30"), ("Arsenal", "17:30"), ("Lyon", None), ("Leeds", "15:00"),
        ]  # fmt: skip
        assert len(repo.fixtures_between(s, D(2026, 10, 24), D(2026, 10, 24), league="F1")) == 1
        assert {m.home for m in repo.fixtures_between(s, D(2026, 10, 1), D(2026, 12, 1), team="Arsenal")} == {
            "Arsenal", "Leeds", "Chelsea",
        }  # fmt: skip


def test_recent_results_teams_and_leagues(factory):
    rows = [
        _fixture(f"2026-09-{d:02d}", "Arsenal", f"Team{d}", home_goals=d % 3, away_goals=1)
        for d in range(1, 15)
    ]
    rows.append(_fixture("2026-10-30", "Arsenal", "Future"))
    rows.append({"date": "2020-01-01", "league": "E1", "home": "Old FC", "away": "Gone FC",
                 "home_goals": 1, "away_goals": 0})  # fmt: skip
    rows.append({"date": "2026-08-01", "league": "E1", "home": "Arsenal", "away": "Leeds",
                 "home_goals": 0, "away_goals": 0})  # fmt: skip
    with factory() as s:
        repo.upsert_matches(s, frame(rows))
        recent = repo.recent_results(s, "Arsenal", 10)
        assert len(recent) == 10 and recent[0].date == D(2026, 9, 14)
        assert all(m.status == "finished" for m in recent)
        names = dict(repo.teams(s))
        assert names["Arsenal"] == "E0"  # league of most recent match
        assert "Old FC" not in names  # outside the window
        assert dict(repo.teams(s, league="E1")) == {"Leeds": "E1"}
        assert [n for n, _ in repo.teams(s, q="team1")] == [f"Team{d}" for d in (1, 10, 11, 12, 13, 14)]
        assert repo.teams(s, q="%") == []  # LIKE wildcards are escaped
        assert repo.leagues(s) == ["E0", "E1"]
        assert repo.latest_finished_by_league(s) == {"E0": D(2026, 9, 14), "E1": D(2026, 8, 1)}
        assert repo.team_exists(s, "Leeds") and not repo.team_exists(s, "Nobody")
        assert repo.count_matches(s, "scheduled") == 1
        assert repo.count_matches(s, "finished", since=D(2026, 9, 1)) == 14
        by_league = repo.known_teams_by_league(s)
        assert "Arsenal" in by_league["E0"]


def test_teams_empty_db(factory):
    with factory() as s:
        assert repo.teams(s) == []
        assert repo.latest_match_date(s) is None


def test_prune_stale_fixtures_only_removes_old_scheduled(factory):
    rows = [
        _fixture("2026-10-01", "A", "B"),
        _fixture("2026-10-01", "C", "D", home_goals=1, away_goals=0),
        _fixture("2026-10-19", "E", "F"),
    ]
    with factory() as s:
        repo.upsert_matches(s, frame(rows))
        assert repo.prune_stale_fixtures(s, before=D(2026, 10, 17)) == 1
        assert {m.home for m in s.query(Match)} == {"C", "E"}


def test_finished_frame_round_trips_contract(factory):
    rows = [
        _fixture("2026-09-01", home_goals=2, away_goals=0, home_shots=10, home_xg=1.2),
        _fixture("2026-10-01", "Leeds", "Chelsea", home_goals=1, away_goals=1),
        _fixture("2026-10-30", "X", "Y"),
    ]
    with factory() as s:
        repo.upsert_matches(s, frame(rows))
        df = repo.finished_frame(s)
        assert tuple(df.columns) == MATCH_COLUMNS and len(df) == 2
        assert df["date"].dtype == "datetime64[ns]"
        assert df.iloc[0]["home_shots"] == 10.0 and pd.isna(df.iloc[0]["away_shots"])
        assert len(repo.finished_frame(s, after=D(2026, 9, 1))) == 1


def test_ingest_runs(factory):
    with factory() as s:
        assert repo.last_successful_run(s) is None
        run = repo.start_run(s)
        repo.finish_run(s, run, "failed", error="boom")
        ok = repo.start_run(s)
        repo.finish_run(s, ok, "success", results_rows=5)
        last = repo.last_successful_run(s)
        assert last.id == ok.id and last.results_rows == 5 and last.finished_at is not None


def test_seed_bulk_then_upsert(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'sub' / 'seed.db').as_posix()}"
    seed_mod.main(["--csv", str(FIXTURES / "history.csv"), "--database-url", db_url])
    factory = seed_mod.create_session_factory(db_url)
    with factory() as s:
        assert repo.count_matches(s) == 5
        m = s.query(Match).filter_by(home="Hibernian").one()
        assert (m.league, m.date, m.time, m.home_goals, m.away_goals, m.home_shots) == (
            "SC0", D(2026, 9, 3), "19:45", 1, 3, 17,
        )  # fmt: skip
        assert m.odds_home == 2.45 and m.source == "history"
    stats = seed_mod.seed(factory, FIXTURES / "history.csv")
    assert stats == repo.UpsertStats(0, 0)


def test_seed_missing_csv_exits(tmp_path):
    with pytest.raises(SystemExit):
        seed_mod.main(["--csv", str(tmp_path / "nope.csv"), "--database-url", "sqlite:///:memory:"])

