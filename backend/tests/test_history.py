import datetime as dt
import json

import httpx
import pytest
from mp_testkit import FIXTURES, memory_factory

from matchpredict.db import repository as repo
from matchpredict.db.models import Match
from matchpredict.ingest import history
from matchpredict.ingest.http import FetchError

TODAY = dt.date(2026, 10, 21)
NO_SLEEP = lambda _s: None

E0 = (FIXTURES / "E0.csv").read_bytes()
ARG = (FIXTURES / "ARG.csv").read_bytes()
# early-2000s layout: "HT"/"AT" instead of HomeTeam/AwayTeam, no Time, 2-digit years, old bookmakers
G1_OLD = (
    b"Div,Date,HT,AT,FTHG,FTAG,FTR,GBH,GBD,GBA\n"
    b"G1,26/08/00,Olympiakos,AEK,2,1,H,1.5,3.5,5.0\n"
    b"G1,27/08/00,Panathinaikos,PAOK,0,0,D,1.8,3.2,4.0\n"
    b",,,,,,,,,\n"
)
HTML = b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"><html>300 Multiple Choices</html>'


def _plan(tmp_path, **kw):
    base = {"divisions": ("E0", "G1"), "new_leagues": {"ARG": "ARG"}, "first": 2025}
    return history.plan_files(tmp_path, TODAY, **{**base, **kw})


class FakeSite:
    """Serves E0 / ARG fixtures, an old-format G1 file for 2526, HTML for anything else."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail = fail or set()

    def __call__(self, url: str) -> bytes | None:
        self.calls.append(url)
        if any(url.endswith(f) for f in self.fail):
            raise FetchError(f"boom for {url}")
        if url.endswith("/2627/E0.csv"):
            return E0
        if url.endswith("/2526/G1.csv"):
            return G1_OLD
        if url.endswith("/new/ARG.csv"):
            return ARG
        if url.endswith("/2526/E0.csv"):
            return None  # 404-like
        return HTML


def test_season_codes_start_in_july():
    assert history.season_codes(dt.date(2026, 6, 30), first=2024) == ["2425", "2526"]
    assert history.season_codes(TODAY, first=2024) == ["2425", "2526", "2627"]
    assert history.season_codes(TODAY)[0] == "0001" and len(history.season_codes(TODAY)) == 27


def test_plan_covers_all_divisions_and_extra_countries(tmp_path):
    files = history.plan_files(tmp_path, TODAY)
    assert len(files) == 27 * 22 + 16
    current = [f for f in files if f.refresh and f.kind == "main"]
    assert {f.url.rsplit("/", 2)[1] for f in current} == {"2627"}
    assert all(f.refresh for f in files if f.kind == "new")
    assert files[0].url == "https://football-data.co.uk/mmz4281/0001/E0.csv"
    assert files[0].path == tmp_path / "mmz4281" / "0001" / "E0.csv"
    assert files[-1].url.endswith("/new/USA.csv")


def test_looks_like_csv():
    assert history.looks_like_csv(E0, "main")  # with a UTF-8 BOM
    assert history.looks_like_csv(ARG, "new")
    assert not history.looks_like_csv(ARG, "main")
    assert not history.looks_like_csv(HTML, "main")
    assert not history.looks_like_csv(None, "main") and not history.looks_like_csv(b"", "new")


def test_download_caches_marks_missing_and_resumes(tmp_path):
    files = _plan(tmp_path)
    site = FakeSite()
    result = history.HistoryResult()
    fresh: dict[str, bytes] = {}
    seen = []
    history.download(files, site, result, sleep=NO_SLEEP, progress=lambda d, t: seen.append((d, t)), fresh=fresh)
    assert result.files == 5 and result.downloaded == 3 and result.missing == 2 and not result.errors
    assert seen[-1] == (5, 5)
    assert (tmp_path / "mmz4281" / "2526" / "E0.csv.missing").exists()  # past season: remembered
    assert not (tmp_path / "mmz4281" / "2627" / "G1.csv.missing").exists()  # current: retried
    assert set(fresh) == {f.url for f in files if f.refresh and f.url.endswith(("E0.csv", "ARG.csv"))}

    again, site2 = history.HistoryResult(), FakeSite()
    history.download(files, site2, again, sleep=NO_SLEEP)
    assert again.cached == 1 and again.missing == 2  # 2526/G1 cached, 2526/E0 + 2627/G1 missing
    assert all("/2526/" not in url for url in site2.calls)  # past seasons never re-downloaded


def test_download_records_errors_and_keeps_going(tmp_path):
    files = _plan(tmp_path)
    result = history.HistoryResult()
    history.download(files, FakeSite(fail={"/2627/E0.csv"}), result, sleep=NO_SLEEP)
    assert len(result.errors) == 1 and "boom" in result.errors[0]
    assert result.downloaded == 2


def test_parse_cached_maps_both_layouts_and_keeps_finished_only(tmp_path):
    files = _plan(tmp_path)
    history.download(files, FakeSite(), history.HistoryResult(), sleep=NO_SLEEP)
    frame = history.parse_cached(files)
    assert set(frame["league"]) == {"E0", "G1", "ARG"}
    assert frame["home_goals"].notna().all()
    g1 = frame[frame["league"] == "G1"]
    assert list(g1["home"]) == ["Olympiakos", "Panathinaikos"]
    assert g1["odds_home"].isna().all()  # only Bet365 / market average pre-match odds are used
    e0 = frame[frame["league"] == "E0"].iloc[0]
    assert e0["odds_home"] > 1 and e0["time"] is not None
    assert history.parse_cached([]).empty


def test_seed_from_web_is_idempotent(tmp_path):
    factory = memory_factory()
    cache = tmp_path / "history"
    result = history.seed_from_web(factory, cache, today=TODAY, fetch=FakeSite(), sleep=NO_SLEEP)
    assert result.status == "complete" and result.rows > 0 and result.inserted == result.rows
    assert history.is_complete(cache)
    assert json.loads((cache / history.COMPLETE_MARKER).read_text())["rows"] == result.rows
    with factory() as s:
        assert repo.count_matches(s, "finished") == result.rows
        assert s.query(Match).filter_by(league="E0").first().source == history.SOURCE

    skipped = history.seed_from_web(factory, cache, today=TODAY, fetch=FakeSite(), sleep=NO_SLEEP)
    assert skipped.status == "skipped"
    forced = history.seed_from_web(factory, cache, today=TODAY, fetch=FakeSite(), sleep=NO_SLEEP, force=True)
    assert forced.status == "complete" and forced.inserted == 0  # upsert, no duplicates
    with factory() as s:
        assert repo.count_matches(s, "finished") == result.rows


def test_seed_from_web_partial_leaves_no_marker(tmp_path, monkeypatch):
    factory = memory_factory()
    cache = tmp_path / "history"
    result = history.seed_from_web(factory, cache, today=TODAY, fetch=FakeSite(fail={"/new/ARG.csv"}),
                                   sleep=NO_SLEEP)  # fmt: skip
    assert result.status == "partial" and result.rows > 0 and not history.is_complete(cache)

    monkeypatch.setattr(history, "store", lambda *a: (_ for _ in ()).throw(RuntimeError("db down")))
    failed = history.seed_from_web(factory, cache, today=TODAY, fetch=FakeSite(), sleep=NO_SLEEP)
    assert failed.status == "failed" and "db down" in failed.errors[-1]


def _transport(responses):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return httpx.Response(item[0], content=item[1])

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_http_fetcher_statuses_and_retries():
    client, calls = _transport([(200, b"Div,x"), (300, HTML), (404, b""), (503, b""), (200, b"ok")])
    fetch = history.HttpHistoryFetcher(client, sleep=NO_SLEEP, curl=lambda u: pytest.fail("no curl"))
    assert fetch("https://x/a.csv") == b"Div,x"
    assert fetch("https://x/b.csv") is None
    assert fetch("https://x/c.csv") is None
    assert fetch("https://x/d.csv") == b"ok"  # 503 retried
    assert len(calls) == 5
    client2, _ = _transport([(403, b"")])
    with pytest.raises(FetchError):
        history.HttpHistoryFetcher(client2, sleep=NO_SLEEP)("https://x/e.csv")
    fetch.close()


def test_http_fetcher_falls_back_to_curl_and_sticks():
    boom = httpx.ConnectError("tls handshake failed")
    client, calls = _transport([boom, boom, boom])
    curled = []
    fetch = history.HttpHistoryFetcher(client, sleep=NO_SLEEP, curl=lambda u: curled.append(u) or b"Div,y")
    assert fetch("https://x/a.csv") == b"Div,y"
    assert fetch.use_curl and fetch("https://x/b.csv") == b"Div,y"
    assert len(calls) == 3 and curled == ["https://x/a.csv", "https://x/b.csv"]


def test_memo_fetcher_serves_fresh_files_for_either_host():
    fallback = []
    fetch = history.memo_fetcher({"https://football-data.co.uk/mmz4281/2627/E0.csv": b"E0"},
                                 lambda u: fallback.append(u) or b"net")  # fmt: skip
    assert fetch("https://www.football-data.co.uk/mmz4281/2627/E0.csv") == b"E0"
    assert fetch("https://football-data.co.uk/mmz4281/2627/E0.csv") == b"E0"
    assert fetch("https://www.football-data.co.uk/fixtures.csv") == b"net"
    assert fallback == ["https://www.football-data.co.uk/fixtures.csv"]


def test_parallel_download_matches_sequential(tmp_path):
    seq, par = history.HistoryResult(), history.HistoryResult()
    history.download(_plan(tmp_path / "a"), FakeSite(), seq, sleep=NO_SLEEP)
    fresh: dict[str, bytes] = {}
    history.download(_plan(tmp_path / "b"), FakeSite(), par, sleep=NO_SLEEP, workers=3, fresh=fresh)
    assert (seq.downloaded, seq.missing, seq.cached) == (par.downloaded, par.missing, par.cached) == (3, 2, 0)
    assert len(fresh) == 2
