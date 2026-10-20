"""League catalogue shared by ingest and the API."""

from __future__ import annotations

from typing import NamedTuple


class League(NamedTuple):
    name: str
    country: str


MAIN_DIVISIONS: tuple[str, ...] = (
    "E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2",
    "I1", "I2", "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1",
)  # fmt: skip

# UEFA club competitions (cross-league; served by predictor/europe.py). Listed first.
EUROPEAN_LEAGUES: tuple[str, ...] = ("UCL", "UEL", "UECL")
EUROPE_COUNTRY = "Europe"

LEAGUES: dict[str, League] = {
    "UCL": League("UEFA Champions League", EUROPE_COUNTRY),
    "UEL": League("UEFA Europa League", EUROPE_COUNTRY),
    "UECL": League("UEFA Conference League", EUROPE_COUNTRY),
    "E0": League("Premier League", "England"),
    "E1": League("Championship", "England"),
    "E2": League("League One", "England"),
    "E3": League("League Two", "England"),
    "EC": League("National League", "England"),
    "SC0": League("Premiership", "Scotland"),
    "SC1": League("Championship", "Scotland"),
    "SC2": League("League One", "Scotland"),
    "SC3": League("League Two", "Scotland"),
    "D1": League("Bundesliga", "Germany"),
    "D2": League("2. Bundesliga", "Germany"),
    "I1": League("Serie A", "Italy"),
    "I2": League("Serie B", "Italy"),
    "SP1": League("La Liga", "Spain"),
    "SP2": League("Segunda División", "Spain"),
    "F1": League("Ligue 1", "France"),
    "F2": League("Ligue 2", "France"),
    "N1": League("Eredivisie", "Netherlands"),
    "B1": League("Pro League", "Belgium"),
    "P1": League("Primeira Liga", "Portugal"),
    "T1": League("Süper Lig", "Turkey"),
    "G1": League("Super League", "Greece"),
    "ARG": League("Liga Profesional", "Argentina"),
    "AUT": League("Bundesliga", "Austria"),
    "BRA": League("Série A", "Brazil"),
    "CHN": League("Super League", "China"),
    "DEN": League("Superliga", "Denmark"),
    "FIN": League("Veikkausliiga", "Finland"),
    "IRL": League("Premier Division", "Ireland"),
    "JAP": League("J1 League", "Japan"),
    "MEX": League("Liga MX", "Mexico"),
    "NOR": League("Eliteserien", "Norway"),
    "POL": League("Ekstraklasa", "Poland"),
    "ROM": League("Liga I", "Romania"),
    "RUS": League("Premier League", "Russia"),
    "SWE": League("Allsvenskan", "Sweden"),
    "SUI": League("Super League", "Switzerland"),
    "USA": League("MLS", "USA"),
}

# Extra-country leagues: our code -> file code on football-data.co.uk (new/{FILE}.csv)
NEW_LEAGUE_FILES: dict[str, str] = {
    "ARG": "ARG", "AUT": "AUT", "BRA": "BRA", "CHN": "CHN", "DEN": "DNK", "FIN": "FIN",
    "IRL": "IRL", "JAP": "JPN", "MEX": "MEX", "NOR": "NOR", "POL": "POL", "ROM": "ROU",
    "RUS": "RUS", "SWE": "SWE", "SUI": "SWZ", "USA": "USA",
}  # fmt: skip

# "Country" column value -> our league code
COUNTRY_TO_CODE: dict[str, str] = {
    LEAGUES[code].country: code for code in NEW_LEAGUE_FILES
}

# Current football-data.co.uk spelling -> the spelling used in our training history,
# per league (found by fuzzy-matching ingested names against Matches.csv).
TEAM_ALIASES: dict[str, dict[str, str]] = {
    "MEX": {"Tigres UANL": "U.A.N.L.- Tigres", "UNAM Pumas": "U.N.A.M.- Pumas"},
    "ROM": {"Dinamo Bucuresti": "Din. Bucuresti"},
    "FIN": {"PK-35": "PK-35 Vantaa"},
    "B1": {"Lommel SK": "Lommel"},
}


def is_european(league: str | None) -> bool:
    """True for the UEFA club competitions (UCL, UEL, UECL)."""
    return league in EUROPEAN_LEAGUES
