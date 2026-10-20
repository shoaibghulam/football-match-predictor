from datetime import date

import httpx
import pandas as pd

from matchpredict.contracts import MATCH_COLUMNS
from matchpredict.ingest import football_data_org as org

KNOWN = {
    "E0": ["Man United", "Man City", "Tottenham", "Brighton", "Arsenal", "Coventry"],
    "D1": ["Bayern Munich", "Dortmund", "Leverkusen"],
}


def _match(comp, home, away, utc="2026-10-24T14:00:00Z", status="TIMED", short=None):
    return {
        "competition": {"code": comp},
        "status": status,
        "utcDate": utc,
        "homeTeam": {"name": home, "shortName": short, "tla": None},
        "awayTeam": {"name": away, "shortName": None, "tla": None},
    }


def test_normalise_strips_accents_and_affixes():
    assert org.normalise("FC Bayern München") == "bayern munchen"
    assert org.normalise("Brighton & Hove Albion FC") == "brighton hove albion"


def test_map_team_alias_exact_and_fuzzy():
    assert org.map_team(["Manchester United FC"], KNOWN["E0"]) == "Man United"
    assert org.map_team(["Arsenal"], KNOWN["E0"]) == "Arsenal"
    assert org.map_team(["Borussia Dortmund"], KNOWN["D1"], threshold=60) == "Dortmund"
    assert org.map_team(["Coventry City FC"], KNOWN["E0"], threshold=60) == "Coventry"
    assert org.map_team(["Real Madrid CF"], KNOWN["E0"]) is None
    assert org.map_team(["Nobody"], []) is None
    assert org.map_team([None, "Arsenal"], KNOWN["E0"]) == "Arsenal"


def test_parse_matches_maps_converts_time_and_logs_misses():
    payload = {
        "matches": [
            _match("PL", "Manchester United FC", "Tottenham Hotspur FC"),
            _match("PL", "Arsenal FC", "Unknown Town FC", short="Arsenal"),
            _match("PL", "Manchester City FC", "Brighton & Hove Albion FC", status="FINISHED"),
            _match("CL", "Arsenal FC", "Bayern"),
            _match("BL1", "FC Bayern München", "Bayer 04 Leverkusen", utc="2026-12-05T17:30:00Z"),
        ]
    }
    rows, misses = org.parse_matches(payload, KNOWN)
    assert [(r["league"], r["home"], r["away"]) for r in rows] == [
        ("E0", "Man United", "Tottenham"),
        ("D1", "Bayern Munich", "Leverkusen"),
    ]
    assert rows[0]["time"] == "15:00"  # 14:00 UTC = 15:00 BST
    assert rows[1]["time"] == "17:30"  # GMT in December
    assert misses == ["E0:Unknown Town FC"]


def test_date_chunks_respect_api_limit():
    chunks = org._date_chunks(date(2026, 10, 20), date(2026, 11, 3))
    assert chunks == [(date(2026, 10, 20), date(2026, 10, 29)), (date(2026, 10, 30), date(2026, 11, 3))]


def test_fetch_org_fixtures_sends_token_and_paces_requests():
    requests: list[httpx.Request] = []

    def handler(req):
        requests.append(req)
        if len(requests) == 1:
            return httpx.Response(200, json={"matches": [_match("PL", "Arsenal", "Coventry")]})
        return httpx.Response(429, json={"message": "slow down"})

    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = org.fetch_org_fixtures(
        "secret", date(2026, 10, 20), date(2026, 11, 3), KNOWN, client=client, sleep=sleeps.append
    )
    assert requests[0].headers["X-Auth-Token"] == "secret"
    assert "competitions=PL" in str(requests[0].url)
    assert sleeps == [org.REQUEST_INTERVAL]
    assert len(result.errors) == 1
    assert tuple(result.frame.columns) == MATCH_COLUMNS
    assert result.frame.iloc[0]["date"] == pd.Timestamp("2026-10-24")


def test_fetch_org_fixtures_owns_client(monkeypatch):
    real = httpx.Client
    made = []

    def factory(**kwargs):
        c = real(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"matches": []})))
        made.append(c)
        return c

    monkeypatch.setattr(org.httpx, "Client", factory)
    result = org.fetch_org_fixtures("k", date(2026, 10, 20), date(2026, 10, 21), KNOWN)
    assert result.frame.empty and made[0].is_closed
