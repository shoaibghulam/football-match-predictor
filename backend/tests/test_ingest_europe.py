import datetime as dt
import json

import httpx
import pandas as pd
import pytest
from europe_kit import make_service
from mp_testkit import frame, memory_factory
from sqlalchemy import create_engine, inspect, text

from matchpredict.config import Settings
from matchpredict.db import repository as repo
from matchpredict.db.models import Match
from matchpredict.db.session import init_db, migrate
from matchpredict.ingest import europe as eu
from matchpredict.ingest import run as run_mod
from matchpredict.loader import ServiceHolder, load_europe_service

NOW = dt.datetime(2026, 10, 20, 19, 30, tzinfo=dt.UTC)  # 20:30 UK, PSG v Barcelona in play


def _settings(tmp_path, **kw) -> Settings:
    base = {"database_url": "sqlite:///:memory:", "artifact_path": tmp_path / "m.joblib",
            "europe_artifact_path": tmp_path / "europe.joblib", "v2_artifact_path": tmp_path / "no-v2.joblib", "scheduler_enabled": False,
            "football_data_org_key": None, "openfootball_dir": tmp_path / "of"}  # fmt: skip
    return Settings(**{**base, **kw})


def _org_payload():
    def match(status, utc, home, away, score=None, stage="LEAGUE_STAGE"):
        return {"utcDate": utc, "status": status, "stage": stage, "competition": {"code": "CL"},
                "homeTeam": {"name": home}, "awayTeam": {"name": away},
                "score": {"fullTime": score or {"home": None, "away": None}}}  # fmt: skip

    return {"matches": [
        match("IN_PLAY", "2026-10-20T19:00:00Z", "Paris Saint-Germain FC", "FC Barcelona", {"home": 3, "away": 2}),
        match("FINISHED", "2026-10-20T16:45:00Z", "Celtic FC", "Club Brugge KV", {"home": 1, "away": 0}),
        match("TIMED", "2026-10-21T19:00:00Z", "FC Bayern München", "Arsenal FC"),
    ]}  # fmt: skip


def _tsdb_fetch(url: str) -> bytes:
    if "eventspastleague" in url:
        events = [{"dateEvent": "2026-10-22", "strTimestamp": "2026-10-22T16:45:00", "strHomeTeam": "Kairat Almaty",
                   "strAwayTeam": "Benfica", "strStatus": "NS", "intRound": "3", "strSeason": "2026-2027"}]
        return json.dumps({"events": events}).encode("utf-8")
    return json.dumps({"events": None}).encode("utf-8")


def _client(calls):
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json=_org_payload())

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def europe(tmp_path):
    return ServiceHolder(make_service(), artifact_path=tmp_path / "europe.joblib")


def _seed_domestic(factory):
    with factory() as s:
        repo.upsert_matches(s, frame([
            {"date": "2026-10-18", "league": "SP1", "home": "Barcelona", "away": "Getafe",
             "home_goals": 2, "away_goals": 0},
            {"date": "2026-10-18", "league": "F1", "home": "Paris SG", "away": "Lens",
             "home_goals": 1, "away_goals": 1},
        ]))  # fmt: skip


def test_refresh_with_key_uses_org_for_ucl_and_tsdb_for_others(tmp_path, europe):
    factory = memory_factory()
    _seed_domestic(factory)
    calls = []
    settings = _settings(tmp_path, football_data_org_key="SECRET")
    summary = eu.refresh_european(factory, settings, europe, today=dt.date(2026, 10, 20), fetch=_tsdb_fetch,
                                  client=_client(calls), sleep=lambda _: None)  # fmt: skip
    assert summary.errors == [] and summary.sources == ["football-data.org", "thesportsdb"]
    assert summary.live == 1 and summary.rows == 3 + 2  # UEL + UECL each saw the Kairat event
    assert summary.applied_results >= 1 and (tmp_path / "europe.joblib").exists()
    with factory() as s:
        rows = {(m.league, m.home, m.away): m for m in repo.fixtures_between(s, dt.date(2026, 10, 20), dt.date(2026, 10, 22))}
        live = repo.live_matches(s)
    psg = rows[("UCL", "Paris SG", "Barcelona")]
    assert (psg.status, psg.home_goals, psg.away_goals, psg.stage) == ("live", 3, 2, "league")
    assert rows[("UCL", "Celtic", "Club Brugge")].status == "finished"
    assert rows[("UECL", "FK Kairat", "Benfica")].unmapped is True  # Kairat: raw European name
    assert psg.unmapped is False
    assert [m.home for m in live] == ["Paris SG"]
    assert calls and "SECRET" not in " ".join(calls)


def test_refresh_without_key_falls_back_to_tsdb_and_never_raises(tmp_path):
    factory = memory_factory()
    summary = eu.refresh_european(factory, _settings(tmp_path), None, today=dt.date(2026, 10, 20),
                                  fetch=_tsdb_fetch, sleep=lambda _: None)  # fmt: skip
    assert summary.sources == ["thesportsdb"] and summary.rows == 3
    with factory() as s:
        unmapped = [m.unmapped for m in repo.fixtures_between(s, dt.date(2026, 10, 22), dt.date(2026, 10, 22))]
    assert unmapped == [True, True, True]  # no European model -> raw names kept, flagged

    def boom(_):
        raise RuntimeError("network down")

    bad = eu.refresh_european(factory, _settings(tmp_path), None, today=dt.date(2026, 10, 20), fetch=boom,
                              sleep=lambda _: None)  # fmt: skip
    assert bad.status == "partial" and "RuntimeError" in bad.errors[0]


def test_live_refresh_only_runs_inside_the_window(tmp_path, europe):
    factory = memory_factory()
    settings = _settings(tmp_path, football_data_org_key="K")
    calls = []
    skipped = eu.refresh_live(factory, settings, europe, now=NOW, client=_client(calls), fetch=_tsdb_fetch)
    assert skipped.skipped and calls == []
    with factory() as s:
        repo.upsert_matches(s, frame([{"date": "2026-10-20", "time": "20:00", "league": "UCL",
                                        "home": "Paris SG", "away": "Barcelona"}]))  # fmt: skip
    assert eu.active_leagues(factory, NOW, 3.0) == ("UCL",)
    assert eu.active_leagues(factory, NOW + dt.timedelta(hours=5), 3.0) == ()
    ran = eu.refresh_live(factory, settings, europe, now=NOW, client=_client(calls), fetch=_tsdb_fetch,
                          sleep=lambda _: None)  # fmt: skip
    assert not ran.skipped and len(calls) == 1 and ran.live == 1
    with factory() as s:
        assert repo.live_matches(s)[0].home_goals == 3
    # once live, it stays in the window even far from kick-off
    assert eu.active_leagues(factory, NOW + dt.timedelta(hours=6), 3.0) == ("UCL",)


def test_kickoff_utc():
    m = Match(league="UCL", date=dt.date(2026, 10, 20), time="20:00", home="a", away="b")
    assert eu.kickoff_utc(m) == dt.datetime(2026, 10, 20, 19, 0, tzinfo=dt.UTC)
    m.time = None
    assert eu.kickoff_utc(m) is None


def test_seed_history_and_apply_to_europe(tmp_path, europe):
    cache = tmp_path / "of" / "2025-26"
    cache.mkdir(parents=True)
    (cache / "cl.txt").write_text(
        "= UEFA Champions League 2025/26\n▪ League, Matchday 2\n  Wed Oct 1 2025\n"
        "    21:00  FC Barcelona (ESP)      v Paris Saint-Germain FC (FRA)  1-2 (1-1)\n"
        "           Qarabağ Ağdam FK (AZE)  v FC København (DEN)       2-0 (1-0)\n",
        encoding="utf-8",
    )
    factory = memory_factory()
    settings = _settings(tmp_path)
    stats = eu.seed_history(factory, settings, europe, download=False)
    assert stats.inserted == 2
    with factory() as s:
        got = {(m.home, m.away): m for m in repo.fixtures_between(s, dt.date(2025, 10, 1), dt.date(2025, 10, 1))}
    assert got[("Barcelona", "Paris SG")].unmapped is False and got[("Barcelona", "Paris SG")].stage == "league"
    assert got[("Qarabağ Ağdam FK", "FC København")].unmapped is True
    assert eu.seed_history(factory, settings, europe, download=False).inserted == 0
    assert eu.apply_to_europe(factory, None, settings) == 0
    _seed_domestic(factory)
    assert eu.apply_to_europe(factory, europe, settings) == 2  # the two domestic results


def test_seed_history_downloads(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(eu.openfootball, "download", lambda d, t, fetch: seen.append(d) or [])
    eu.seed_history(memory_factory(), _settings(tmp_path), None, today=dt.date(2026, 10, 21))
    assert seen == [tmp_path / "of"]


def test_run_ingest_includes_the_uefa_step(tmp_path, monkeypatch, europe):
    called = {}

    def fake_refresh(factory, settings, holder, **kw):
        called["holder"] = holder
        return eu.EuropeSummary(rows=4, inserted=3, updated=1, errors=["x"])

    monkeypatch.setattr(eu, "refresh_european", fake_refresh)
    monkeypatch.setattr(run_mod.football_data_uk, "fetch_results",
                        lambda *a, **k: run_mod.football_data_uk.FetchResult(frame=frame([])))  # fmt: skip
    monkeypatch.setattr(run_mod.football_data_uk, "fetch_fixtures",
                        lambda **k: run_mod.football_data_uk.FetchResult(frame=frame([])))  # fmt: skip
    summary = run_mod.run_ingest(memory_factory(), None, _settings(tmp_path), today=dt.date(2026, 10, 20),
                                 sleep=lambda _: None, europe=europe)  # fmt: skip
    assert called["holder"] is europe and summary.europe_rows == 4
    assert summary.inserted == 3 and summary.errors == ["uefa: x"] and summary.status == "partial"


def test_build_mapper_flags_raw_european_names(tmp_path, europe):
    factory = memory_factory()
    _seed_domestic(factory)
    mapper = eu.build_mapper(factory, europe)
    assert mapper.resolve("FC Barcelona") == ("Barcelona", True)
    assert mapper.resolve("Kairat Almaty") == ("FK Kairat", False)
    assert mapper.resolve("Getafe CF", "ESP") == ("Getafe", True)


def test_scheduler_registers_live_job(tmp_path, monkeypatch, europe):
    from matchpredict.ingest import scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "needs_catchup", lambda f: False)
    ran = []
    monkeypatch.setattr(eu, "refresh_live", lambda f, s, e: ran.append(e) or eu.EuropeSummary(skipped=False))
    scheduler = sched_mod.start_scheduler(memory_factory(), ServiceHolder(), _settings(tmp_path), europe=europe)
    try:
        job = scheduler.get_job(sched_mod.LIVE_JOB_ID)
        assert "0:02:00" in str(job.trigger)
        job.func()
        assert ran == [europe]
    finally:
        scheduler.shutdown(wait=False)


# --- DB ----------------------------------------------------------------------------


def test_migrate_adds_missing_columns(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE matches (id INTEGER PRIMARY KEY, league VARCHAR(8), date DATE, "
                          "time VARCHAR(5), home VARCHAR(64), away VARCHAR(64), status VARCHAR(10))"))  # fmt: skip
    added = migrate(engine)
    assert {"matches.minute", "matches.stage", "matches.unmapped", "matches.home_goals"} <= set(added)
    assert migrate(engine) == []
    init_db(engine)
    assert "unmapped" in {c["name"] for c in inspect(engine).get_columns("matches")}


def test_live_status_merge_rules():
    factory = memory_factory()
    base = {"date": "2026-10-20", "time": "20:00", "league": "UCL", "home": "Paris SG", "away": "Barcelona"}

    def upsert(**kw):
        df = frame([{**base, **{k: v for k, v in kw.items() if k not in ("status", "minute")}}])
        if "status" in kw:
            df["status"] = kw["status"]
        if "minute" in kw:
            df["minute"] = kw["minute"]
        with factory() as s:
            repo.upsert_matches(s, df)
            return repo.fixtures_between(s, dt.date(2026, 10, 20), dt.date(2026, 10, 20))[0]

    assert upsert().status == "scheduled"
    m = upsert(home_goals=1, away_goals=0, status="live", minute=30)
    assert (m.status, m.home_goals, m.minute) == ("live", 1, 30)
    m = upsert(home_goals=4, away_goals=2, status="finished")
    assert (m.status, m.home_goals, m.minute) == ("finished", 4, None)
    m = upsert(home_goals=3, away_goals=2, status="live", minute=80)  # stale live poll
    assert (m.status, m.home_goals, m.minute) == ("finished", 4, None)
    with factory() as s:
        assert repo.prune_stale_fixtures(s, dt.date(2026, 10, 25)) == 0
        assert repo.finished_frame(s).empty  # UEFA excluded from the domestic model
        assert len(repo.finished_frame(s, include_europe=True)) == 1


def test_teams_prefer_domestic_league_over_uefa():
    factory = memory_factory()
    with factory() as s:
        repo.upsert_matches(s, frame([
            {"date": "2026-10-18", "league": "SP1", "home": "Barcelona", "away": "Getafe", "home_goals": 1, "away_goals": 0},
            {"date": "2026-10-20", "league": "UCL", "home": "Paris SG", "away": "Barcelona", "home_goals": 1, "away_goals": 0},
        ]))  # fmt: skip
        teams = dict(repo.teams(s))
        assert teams["Barcelona"] == "SP1" and teams["Paris SG"] == "UCL"
        assert repo.kickoffs_between(s, dt.date(2026, 10, 1), dt.date(2026, 10, 30)) == []


def test_load_europe_service(tmp_path, europe):
    assert load_europe_service(tmp_path / "nope.joblib") is None
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"junk")
    assert load_europe_service(bad) is None
    path = tmp_path / "ok.joblib"
    europe.service.save(path)
    assert load_europe_service(path).model_version == "europe-test"
    holder = ServiceHolder(artifact_path=path, loader=load_europe_service)
    assert holder.ensure_loaded() is not None
    pred = holder.predict("Barcelona", "Paris SG", "UCL", pd.Timestamp("2026-10-21"))
    assert pred["model"] == "europe"
