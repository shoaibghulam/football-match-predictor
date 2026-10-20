import json
from datetime import UTC, date, datetime

import httpx
import numpy as np
import pandas as pd
import pytest

from matchpredict.ingest import football_data_org as org
from matchpredict.ingest import openfootball as of
from matchpredict.ingest import thesportsdb as tsdb
from matchpredict.ingest.http import FetchError, NotFoundError
from matchpredict.ingest.teamnames import (
    TeamNameMapper,
    clean_raw,
    log_misses,
    normalise,
)

CANDS = {
    "Barcelona": "ESP", "Paris SG": "FRA", "Bayern Munich": "GER", "Inter": "ITA", "Ath Madrid": "ESP",
    "Man City": "ENG", "Man United": "ENG", "Dortmund": "GER", "Leverkusen": "GER", "Sp Lisbon": "POR",
    "Club Brugge": "BEL", "PSV Eindhoven": "NED", "Lens": "FRA", "Racing Club": "ARG", "Real Madrid": "ESP",
    "Real Sociedad B": "ESP", "Sociedad": "ESP", "Slavia Praha": "CZE", "Wisla": "POL", "AIK": "SWE",
}  # fmt: skip


@pytest.fixture
def mapper():
    return TeamNameMapper(dict(CANDS))


# --- team names --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "country", "expected"),
    [
        ("FC Barcelona", "ESP", "Barcelona"),
        ("Paris Saint-Germain FC", "FRA", "Paris SG"),
        ("Paris Saint-Germain", None, "Paris SG"),
        ("FC Bayern München (GER)", "GER", "Bayern Munich"),
        ("FC Internazionale Milano", "ITA", "Inter"),
        ("Inter Milan", None, "Inter"),
        ("Club Atlético de Madrid", "ESP", "Ath Madrid"),
        ("Manchester City FC", "ENG", "Man City"),
        ("Manchester United", None, "Man United"),
        ("Borussia Dortmund", "GER", "Dortmund"),
        ("Bayer 04 Leverkusen", "GER", "Leverkusen"),
        ("Sporting Clube de Portugal", "POR", "Sp Lisbon"),
        ("Sporting CP", None, "Sp Lisbon"),
        ("Club Brugge KV", "BEL", "Club Brugge"),
        ("PSV", "NED", "PSV Eindhoven"),
        ("Racing Club de Lens", "FRA", "Lens"),
        ("Real Sociedad de Fútbol", "ESP", "Sociedad"),
        ("Slavia Prague", None, "Slavia Praha"),
        ("Wisła Kraków", "POL", "Wisla"),
        ("AIK Solna", "SWE", "AIK"),
    ],
)
def test_mapper_maps_known_variants(mapper, raw, country, expected):
    assert mapper.map(raw, country) == expected


def test_mapper_misses_resolve_and_country_constraint(mapper, caplog):
    assert mapper.map("Qarabağ Ağdam FK", "AZE") is None
    assert mapper.misses == {"Qarabağ Ağdam FK": "AZE"}
    name, ok = mapper.resolve("Qarabağ Ağdam FK (AZE)", "AZE")
    assert (name, ok) == ("Qarabağ Ağdam FK", False)
    assert mapper.resolve("Qarabag Agdam", "AZE") == ("Qarabağ Ağdam FK", False)
    assert mapper.resolve("FC Barcelona", "ESP") == ("Barcelona", True)
    # same normalised name in two countries: the country decides
    m = TeamNameMapper({"Rangers": "SCO", "Rangers FC": "CHI"})
    assert m.map("Rangers", "SCO") == "Rangers"
    assert m.map("Totally Different", None) is None
    m.add("Rangers", "SCO")  # no-op for existing
    with caplog.at_level("INFO"):
        log_misses(mapper, "test")
    assert "unmapped team names" in caplog.text
    assert clean_raw("PSV (NED)") == "PSV" and normalise("FC Köln") == "koln"


def test_mapper_without_candidates_trusts_aliases():
    assert TeamNameMapper({}).map("FC Barcelona") == "Barcelona"
    assert TeamNameMapper({}).map("Nobody") is None


def test_mapper_fuzzy_tie_gives_up():
    m = TeamNameMapper({"Dinamo Alfa": "XXX", "Dinamo Beta": "XXX"})
    assert m.map("Dinamo", "XXX") is None


# --- openfootball ------------------------------------------------------------------

SAMPLE = """= UEFA Champions League 2025/26

# Date       Tue Sep 16 2025 - Sat May 30 2026 (256d)

▪ League, Matchday 1
  Tue Sep 16 2025
    18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)
           PSV (NED)               v Royale Union Saint-Gilloise (BEL)  1-3 (0-2)
  Wed Dec 10
    21:00  FC Barcelona (ESP)      v Paris Saint-Germain FC (FRA)  1-2 (1-1)
▪ Playoffs, Matchday 2
  Wed Feb 25
    21:00  Juventus FC (ITA)       v Galatasaray SK (TUR)     3-2 a.e.t. (3-0, 1-0)
           RB Leipzig (GER)        v Spartak Moskva (RUS)     [cancelled]
           Team X (ESP)            v Team Y (FRA)             3-0    [awarded]
▪ Finals, Final
  Sat May 30
    18:00  Paris Saint-Germain FC (FRA) v Arsenal FC (ENG)         4-3 pen. 1-1 a.e.t. (1-1, 0-1)
"""


def test_parse_text_scores_dates_and_stages():
    df = of.parse_text(SAMPLE, "UCL", "main", "2025-26")
    assert len(df) == 5
    assert df["date"].dt.year.tolist() == [2025, 2025, 2025, 2026, 2026]
    assert df.loc[2, ["home_raw", "away_raw"]].tolist() == ["FC Barcelona", "Paris Saint-Germain FC"]
    assert df.loc[3, ["home_goals", "away_goals"]].tolist() == [3.0, 0.0]  # 90-minute score
    assert df.loc[4, ["home_goals", "away_goals"]].tolist() == [1.0, 1.0]
    assert df["stage"].tolist() == ["league", "league", "league", "ko", "final"]
    assert df.loc[4, "neutral"] and df.loc[1, "time"] == "18:45"
    assert df.loc[0, "home_country"] == "ESP"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2-1 (1-0)", (2, 1)), ("0-0", (0, 0)), ("2-1 a.e.t. (1-1)", (1, 1)), ("2-1 a.e.t.", None),
     ("[cancelled]", None), ("", None), ("x", None)],
)  # fmt: skip
def test_parse_result(text, expected):
    assert of.parse_result(text) == expected


def test_classify_stage_and_dates():
    assert of.classify_stage("1. Round", "qual") == "qual"
    assert of.classify_stage("Round 2", "main") == "qual"
    assert of.classify_stage("Group A", "main") == "league"
    assert of.classify_stage("Gruppe H", "main") == "league"
    assert of.classify_stage("Sechzehntelfinale", "main") == "ko"
    assert of.classify_stage("Final", "main") == "final"
    assert of.classify_stage("Round of 16", "main") == "ko"
    assert of.seasons_until(date(2026, 10, 21))[-1] == "2026-27"
    assert of.seasons_until(date(2026, 3, 1))[-1] == "2025-26"
    covid = "= UEFA Champions League 2019/20\n▪ Final\n  Wed Mar 11 2020\n    21:00  A (ESP) v B (FRA)  1-0\n  Sun Aug 23\n    21:00  C (ESP) v D (FRA)  0-1\n"
    df = of.parse_text(covid, "UCL", "main", "2019-20")
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-03-11", "2020-08-23"]
    nodate = of.parse_text("▪ Group A\n  Tue Feb 30\n    A (ESP) v B (FRA) 1-0\n  Sat Jan 4\n    A (ESP) v B (FRA) 2-0\n", "UCL", "main", "2019-20")
    assert nodate["date"].dt.year.tolist() == [2020]


def test_download_caches_and_marks_404(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if url.endswith("/cl.txt"):
            return SAMPLE.encode("utf-8")
        if "elq" in url:
            raise FetchError("boom")
        raise NotFoundError(url)

    paths = of.download(tmp_path, date(2012, 10, 1), fetch=fetch, sleep=lambda _: None)
    assert [p.parent.name for p in paths] == ["2011-12", "2012-13"]
    first = len(calls)
    of.download(tmp_path, date(2012, 10, 1), fetch=fetch, sleep=lambda _: None)
    # cached past-season files and 404 markers are not re-fetched; the current season and
    # past files that failed transiently (elq 2011-12) are
    assert len(calls) - first == len(of.FILES) + 1
    frame = of.load_cached(tmp_path)
    assert len(frame) == 10 and set(frame["league"]) == {"UCL"}
    assert of.load_cached(tmp_path / "empty").empty


# --- TheSportsDB -------------------------------------------------------------------


def _event(**kw):
    base = {"dateEvent": "2026-10-20", "strTime": "19:00:00", "strTimestamp": "2026-10-20T19:00:00",
            "strHomeTeam": "Paris Saint-Germain", "strAwayTeam": "Barcelona", "intHomeScore": "1",
            "intAwayScore": "0", "strStatus": "1H", "intRound": "3", "strSeason": "2026-2027",
            "strProgress": "34"}  # fmt: skip
    return {**base, **kw}


def test_tsdb_status_minute_and_stage():
    assert tsdb.map_status("1H") == "live" and tsdb.map_status("HT") == "live"
    assert tsdb.map_status("67'") == "live"
    assert tsdb.map_status("FT") == "finished" and tsdb.map_status("AET") == "finished"
    assert tsdb.map_status("NS") == "scheduled" and tsdb.map_status(None) == "scheduled"
    assert tsdb.map_status("PST") is None
    assert tsdb.parse_minute(None, "45+2'") == 45 and tsdb.parse_minute("", None) is None
    ko = datetime(2026, 10, 20, 19, tzinfo=UTC)
    assert tsdb.stage_of({"intRound": "3"}, ko) == "league"
    assert tsdb.stage_of({"intRound": "200"}, ko.replace(month=5)) == "final"
    assert tsdb.stage_of({"intRound": "x"}, ko.replace(month=8)) == "qual"


def test_tsdb_parse_events(mapper):
    events = [
        _event(),
        _event(strHomeTeam="Real Madrid", strAwayTeam="Mystery Town", strStatus="NS", intHomeScore=None,
               intAwayScore=None, strTimestamp=None, dateEvent="2026-10-21", strTime="16:45:00"),
        _event(strStatus="PST"),
        _event(strStatus="FT", intHomeScore=None),
        _event(strStatus="2H", intHomeScore=None, intAwayScore=None, strHomeTeam="Inter", strProgress=None),
        _event(strTimestamp="bad"),
        _event(strTimestamp=None, dateEvent=None),
    ]  # fmt: skip
    rows = tsdb.parse_events(events, "UCL", mapper)
    assert len(rows) == 3
    live, sched, other = rows
    assert live["home"] == "Paris SG" and live["away"] == "Barcelona" and live["status"] == "live"
    assert live["minute"] == 34 and live["time"] == "20:00" and not live["unmapped"]
    assert sched["status"] == "scheduled" and np.isnan(sched["home_goals"]) and sched["unmapped"]
    assert sched["time"] == "17:45" and sched["date"] == pd.Timestamp("2026-10-21")
    assert other["home_goals"] == 0.0 and other["minute"] is None


def test_tsdb_fetch_european_rounds_and_errors(mapper):
    urls = []

    def fetch(url):
        urls.append(url)
        if "id=4481" in url:
            raise FetchError("down")
        if "eventsround" in url:
            return json.dumps({"events": [_event(), _event(strHomeTeam="Inter", strStatus="FT")]}).encode()
        if "eventspastleague" in url:
            return json.dumps({"events": [_event(intRound="3")]}).encode()
        return json.dumps({"events": None}).encode("utf-8")

    sleeps = []
    res = tsdb.fetch_european(mapper, ("UCL", "UEL"), fetch=fetch, sleep=sleeps.append)
    assert len(res.frame) == 2 and set(res.frame["home"]) == {"Paris SG", "Inter"}
    assert any("eventsround.php?id=4480&r=3&s=2026-2027" in u for u in urls)
    assert len(res.errors) == 2 and res.requests == len(urls) and sleeps
    empty = tsdb.fetch_european(mapper, ("UEL",), fetch=fetch, sleep=lambda _: None)
    assert empty.frame.empty and list(empty.frame.columns) == list(tsdb.FRAME_COLUMNS)


def test_tsdb_round_filter_and_round_errors(mapper):
    client = tsdb.TsdbClient(fetch=lambda url: (_ for _ in ()).throw(FetchError("x")) if "eventsround" in url
                             else json.dumps({"events": [_event()]}).encode(), sleep=lambda _: None)  # fmt: skip
    rows, errors = tsdb.fetch_competition(client, "UCL", mapper)
    assert len(rows) == 2 and len(errors) == 1
    rows, errors = tsdb.fetch_competition(client, "UCL", mapper, rounds_only={99})
    assert errors == []


# --- football-data.org CL ----------------------------------------------------------


def _org_match(status="IN_PLAY", **kw):
    base = {
        "utcDate": "2026-10-20T19:00:00Z", "status": status, "stage": "LEAGUE_STAGE",
        "competition": {"code": "CL"}, "minute": "67",
        "homeTeam": {"name": "Paris Saint-Germain FC", "shortName": "PSG", "tla": "PSG"},
        "awayTeam": {"name": "FC Barcelona", "shortName": "Barça", "tla": "FCB"},
        "score": {"fullTime": {"home": 3, "away": 2}, "regularTime": None},
    }  # fmt: skip
    return {**base, **kw}


def test_org_status_stage_and_parse(mapper):
    assert org.org_status("PAUSED") == "live" and org.org_status("TIMED") == "scheduled"
    assert org.org_status("FINISHED") == "finished" and org.org_status("POSTPONED") is None
    assert org.org_stage("QUALIFICATION_ROUND_2") == "qual" and org.org_stage("FINAL") == "final"
    assert org.org_stage("LAST_16") == "ko" and org.org_stage("LEAGUE_STAGE") == "league"
    payload = {"matches": [
        _org_match(),
        _org_match("FINISHED", score={"fullTime": {"home": 2, "away": 2},
                                      "regularTime": {"home": 1, "away": 1}}, utcDate="2026-10-19T19:00:00Z"),
        _org_match("TIMED", utcDate="2026-10-21T19:00:00Z",
                   homeTeam={"name": "FC Bayern München"}, awayTeam={"name": "Mystery FC"}),
        _org_match("POSTPONED"),
        _org_match("FINISHED", score={"fullTime": {"home": None, "away": None}}),
        _org_match(homeTeam={}),
        _org_match(competition={"code": "EL"}),
        _org_match(score={}, minute=None, utcDate="2026-10-18T19:00:00Z"),
    ]}  # fmt: skip
    rows, misses = org.parse_european_matches(payload, mapper)
    assert len(rows) == 4 and misses == ["Mystery FC"]
    live, fin, sched, nil = rows
    assert (live["home"], live["away"], live["status"], live["minute"]) == ("Paris SG", "Barcelona", "live", 67)
    assert (live["home_goals"], live["away_goals"], live["time"]) == (3.0, 2.0, "20:00")
    assert (fin["home_goals"], fin["away_goals"], fin["minute"]) == (1.0, 1.0, None)
    assert sched["status"] == "scheduled" and sched["unmapped"] and np.isnan(sched["home_goals"])
    assert nil["home_goals"] == 0.0 and nil["minute"] is None


def test_fetch_org_european_with_mock_transport(mapper):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params["dateFrom"] == "2026-10-30":
            return httpx.Response(500)
        return httpx.Response(200, json={"matches": [_org_match(), _org_match(homeTeam={"name": "Mystery"})]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    res = org.fetch_org_european("KEY", date(2026, 10, 20), date(2026, 11, 1), mapper, client=client,
                                 sleep=lambda _: None)  # fmt: skip
    assert seen[0].headers["X-Auth-Token"] == "KEY" and seen[0].url.path == "/v4/competitions/CL/matches"
    assert len(res.frame) == 2 and len(res.errors) == 1
    assert "KEY" not in " ".join(res.errors + res.sources)
    empty = org.fetch_org_european("K", date(2026, 10, 30), date(2026, 10, 30), mapper, client=client)
    assert empty.frame.empty
