"""Canonical team names and countries for the v2 global rating.

The DB stores European (UCL/UEL/UECL) clubs under the names the ingest mapper chose:
domestic DB names, ClubElo names, or raw source names. A few of these differ from the
spelling used by the same club's domestic league rows (e.g. ClubElo "Rapid Wien" vs our
AUT league "SK Rapid"), and different UEFA sources spell some small clubs differently
("Ferencvárosi TC" / "Ferencváros"). ``Canonicaliser`` folds all of them onto one name
so a club has exactly one rating.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from matchpredict.ingest.leagues import EUROPEAN_LEAGUES, LEAGUES
from matchpredict.ingest.teamnames import (
    COUNTRY_CODES,
    clean_raw,
    normalise,
    source_country,
)

# European-source name -> domestic DB name (verified: same club, different spelling).
EURO_TO_DOMESTIC: dict[str, str] = {
    "Rapid Wien": "SK Rapid",
    "HJK Helsinki": "HJK",
    "Ilves Tampere": "Ilves",
    "Vaasa": "VPS",
    "Seinajoki": "SJK",
    "Lech": "Lech Poznan",
    "Craiova": "Univ. Craiova",
    "Otelul Galati": "Otelul",
    "Corvinul Hunedoara": "Corvinul",
    "Universitatea Cluj": "U. Cluj",
    "St Patrick's Athletic": "St. Patricks",
    "Sint-Truiden": "St Truiden",
}

# Spelling variants between UEFA sources (openfootball vs TheSportsDB vs football-data.org).
EURO_VARIANTS: dict[str, str] = {
    "Ferencvárosi TC": "Ferencváros",
    "Paphos FC": "Pafos FC",
    "Pen-y-Bont": "Penybont FC",
    "Petrocub Hîncești": "CS Petrocub",
    "Zirə": "Zira FK",
    "Tobol": "Tobyl Kostanay",
    "Víkingur": "Víkingur Reykjavík",
    "KF Drita": "Drita Gjilan",
    "Levadia Tallinn": "FCI Levadia",
}

# Clubs that only appear in sources without a country (TheSportsDB) -> ClubElo-style code.
MANUAL_COUNTRIES: dict[str, str] = {
    "Torreense": "POR",
    "Ferencváros": "HUN",
    "Dinamo Zagreb": "CRO",
    "Crvena Zvezda": "SRB",
    "Slavia Praha": "CZE",
    "Viktoria Plzeň": "CZE",
    "AC Sparta Praha": "CZE",
    "Shakhtar Donetsk": "UKR",
    "Dinamo Kiev": "UKR",
    "FK Kairat": "KAZ",
    "FC Ararat-Armenia": "ARM",
    "Hapoel Be'er Sheva": "ISR",
    "Omonia Nikosia": "CYP",
    "Levski Sofia": "BUL",
    "NK Celje": "SVN",
    "Slovan Bratislava": "SVK",
    "Sabah FK": "AZE",
    "Qarabağ FK": "AZE",
    "PFC Ludogorets Razgrad": "BUL",
    "Maccabi Tel Aviv": "ISR",
    "Partizan": "SRB",
    "HNK Rijeka": "CRO",
    "Hajduk Split": "CRO",
    "Pafos FC": "CYP",
    "The New Saints": "WAL",
}


def league_country(league: str | None) -> str | None:
    """Country code of a domestic league (None for UEFA competitions / unknown codes)."""
    if league is None or league in EUROPEAN_LEAGUES or league not in LEAGUES:
        return None
    return COUNTRY_CODES.get(LEAGUES[league].country)


@dataclass
class Canonicaliser:
    """Maps any stored team name onto one canonical name; knows each club's country."""

    aliases: dict[str, str] = field(default_factory=dict)
    countries: dict[str, str] = field(default_factory=dict)

    def canonical(self, name: str) -> str:
        seen = 0
        while name in self.aliases and seen < 4:  # short alias chains only
            name = self.aliases[name]
            seen += 1
        return name

    def country(self, name: str) -> str | None:
        return self.countries.get(self.canonical(name))

    @classmethod
    def build(
        cls,
        domestic: Mapping[str, str],
        european_names: Iterable[str],
        source_countries: Mapping[str, str | None] | None = None,
    ) -> Canonicaliser:
        """``domestic``: team -> most frequent domestic league; ``european_names``: every UEFA name
        (with repetitions, so the most frequent spelling wins); ``source_countries``: raw name ->
        source country code (openfootball)."""
        aliases: dict[str, str] = {k: v for k, v in EURO_TO_DOMESTIC.items() if v in domestic}
        aliases.update(EURO_VARIANTS)
        counts = Counter(european_names)
        by_norm: dict[str, list[str]] = {}
        for name, _ in counts.most_common():
            if name in domestic or name in aliases:
                continue
            by_norm.setdefault(normalise(name) or name.casefold(), []).append(name)
        for names in by_norm.values():  # "Omonia Nicosia" / "Omonia Nikosia" -> most frequent
            for other in names[1:]:
                aliases[other] = names[0]
        countries: dict[str, str] = {}
        for team, league in domestic.items():
            code = league_country(league)
            if code:
                countries[team] = code
        for raw, code in (source_countries or {}).items():
            name = clean_raw(raw)
            target = aliases.get(name, name)
            if code and target not in countries:
                countries[target] = source_country(code) or code
        for team, code in MANUAL_COUNTRIES.items():
            countries.setdefault(aliases.get(team, team), code)
        return cls(aliases=aliases, countries=countries)
