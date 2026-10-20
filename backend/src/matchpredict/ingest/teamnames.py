"""Canonical team names across sources (openfootball, football-data.org, TheSportsDB, ClubElo).

The canonical spelling is the football-data.co.uk name used in our DB, which is
also the spelling of ``data/raw/EloRatings.csv`` (ClubElo values joined to
football-data names). Mapping order for a raw name:

1. manual alias table (keyed by the normalised name),
2. exact / case-insensitive match against the candidates,
3. normalised exact match,
4. rapidfuzz match, constrained to the same country when the country is known.

Names that cannot be mapped are returned as ``None``; callers keep the raw name
(cleaned of the ``(CCC)`` country suffix) and flag the row as unmapped.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

FUZZY_THRESHOLD = 88.0
FUZZY_THRESHOLD_NO_COUNTRY = 93.0

# Tokens that carry no identity ("FC Barcelona" == "Barcelona").
_STOPWORDS = frozenset(
    ["fc", "cf", "ac", "sc", "afc", "sl", "ssc", "as", "cd", "rc", "sv", "vfl", "vfb", "fk", "sk", "kv", "bc", "club", "de", "da", "do", "of", "the", "calcio", "cp", "sad", "sfp", "pae", "ofi", "gnk", "nk", "hnk", "ks", "if", "bk", "ff", "ik", "fotball", "tsg", "sportclub", "us", "ud", "sd", "ca", "rcd", "ogc", "losc", "rsc", "kaa", "krc", "jk", "pfc", "tc", "se"]
)
_DIGITS = re.compile(r"\b\d+\b")

# English exonyms -> local spellings used by openfootball for clubs outside ClubElo.
_SYNONYMS = {
    "prague": "praha", "belgrade": "beograd", "kiev": "kyiv", "warsaw": "warszawa",
    "bucharest": "bucuresti", "athens": "athen", "moscow": "moskva",
    "dynamo": "dinamo",
    "nicosia": "nikosia",
    "sint": "st",
    "saint": "st",
}  # fmt: skip

# League country (our LEAGUES catalogue) -> ClubElo-style 3-letter code.
COUNTRY_CODES: dict[str, str] = {
    "England": "ENG", "Scotland": "SCO", "Germany": "GER", "Italy": "ITA", "Spain": "ESP",
    "France": "FRA", "Netherlands": "NED", "Belgium": "BEL", "Portugal": "POR",
    "Turkey": "TUR", "Greece": "GRE", "Austria": "AUT", "Denmark": "DEN", "Finland": "FIN",
    "Ireland": "IRL", "Norway": "NOR", "Poland": "POL", "Romania": "ROM", "Russia": "RUS",
    "Sweden": "SWE", "Switzerland": "SUI", "Argentina": "ARG", "Brazil": "BRA",
    "China": "CHN", "Japan": "JPN", "Mexico": "MEX", "USA": "USA",
}  # fmt: skip

# openfootball / FIFA-style codes -> ClubElo-style codes (identity when missing).
SOURCE_COUNTRY_CODES: dict[str, str] = {"ROU": "ROM", "MCO": "FRA"}

# Normalised raw name -> canonical name. Verified against EloRatings.csv / DB names.
MANUAL_ALIASES: dict[str, str] = {
    "barcelona": "Barcelona",
    "paris saint germain": "Paris SG",
    "psg": "Paris SG",
    "bayern munchen": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    "internazionale milano": "Inter",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "inter": "Inter",
    "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "club atletico madrid": "Ath Madrid",
    "athletic": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "manchester city": "Man City",
    "manchester united": "Man United",
    "borussia dortmund": "Dortmund",
    "bayer leverkusen": "Leverkusen",
    "bayer 04 leverkusen": "Leverkusen",
    "sporting portugal": "Sp Lisbon",
    "sporting": "Sp Lisbon",
    "sporting lisbon": "Sp Lisbon",
    "sporting braga": "Sp Braga",
    "braga": "Sp Braga",
    "sport lisboa e benfica": "Benfica",
    "benfica": "Benfica",
    "porto": "Porto",
    "club brugge": "Club Brugge",
    "brugge": "Club Brugge",
    "psv": "PSV Eindhoven",
    "psv eindhoven": "PSV Eindhoven",
    "ajax": "Ajax",
    "az": "AZ Alkmaar",
    "az alkmaar": "AZ Alkmaar",
    "feyenoord": "Feyenoord",
    "feyenoord rotterdam": "Feyenoord",
    "milan": "Milan",
    "napoli": "Napoli",
    "roma": "Roma",
    "lazio": "Lazio",
    "juventus": "Juventus",
    "atalanta": "Atalanta",
    "real madrid": "Real Madrid",
    "real sociedad": "Sociedad",
    "real sociedad futbol": "Sociedad",
    "real betis": "Betis",
    "real betis balompie": "Betis",
    "sevilla": "Sevilla",
    "valencia": "Valencia",
    "villarreal": "Villarreal",
    "celta vigo": "Celta",
    "celta": "Celta",
    "espanyol barcelona": "Espanol",
    "rayo vallecano madrid": "Vallecano",
    "rayo vallecano": "Vallecano",
    "deportivo alaves": "Alaves",
    "tottenham hotspur": "Tottenham",
    "tottenham": "Tottenham",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "wolverhampton wanderers": "Wolves",
    "leicester city": "Leicester",
    "nottingham forest": "Nott'm Forest",
    "brighton hove albion": "Brighton",
    "olympique marseille": "Marseille",
    "marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "lyon": "Lyon",
    "lille": "Lille",
    "lille osc": "Lille",
    "monaco": "Monaco",
    "stade rennais": "Rennes",
    "rennes": "Rennes",
    "saint etienne": "St Etienne",
    "girondins bordeaux": "Bordeaux",
    "stade brestois 29": "Brest",
    "stade brestois": "Brest",
    "brest": "Brest",
    "racing club lens": "Lens",
    "lens": "Lens",
    "racing lens": "Lens",
    "sporting clube portugal": "Sp Lisbon",
    "sporting clube braga": "Sp Braga",
    "bor monchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "heart midlothian": "Hearts",
    "hearts": "Hearts",
    "wisla krakow": "Wisla",
    "aik solna": "AIK",
    "aik": "AIK",
    "vaasan ps": "Vaasa",
    "vps": "Vaasa",
    "nice": "Nice",
    "montpellier herault": "Montpellier",
    "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "schalke 04": "Schalke 04",
    "schalke": "Schalke 04",
    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    "vfb stuttgart": "Stuttgart",
    "stuttgart": "Stuttgart",
    "werder bremen": "Werder Bremen",
    "hoffenheim": "Hoffenheim",
    "1899 hoffenheim": "Hoffenheim",
    "wolfsburg": "Wolfsburg",
    "union berlin": "Union Berlin",
    "1 union berlin": "Union Berlin",
    "freiburg": "Freiburg",
    "sport club freiburg": "Freiburg",
    "koln": "FC Koln",
    "mainz 05": "Mainz",
    "hertha bsc": "Hertha",
    "celtic": "Celtic",
    "rangers": "Rangers",
    "glasgow rangers": "Rangers",
    "galatasaray": "Galatasaray",
    "fenerbahce": "Fenerbahce",
    "besiktas": "Besiktas",
    "besiktas jk": "Besiktas",
    "trabzonspor": "Trabzonspor",
    "basaksehir": "Buyuksehyr",
    "istanbul basaksehir": "Buyuksehyr",
    "olympiakos piraeus": "Olympiakos",
    "olympiacos": "Olympiakos",
    "olympiakos": "Olympiakos",
    "paok": "PAOK",
    "paok thessaloniki": "PAOK",
    "aek athens": "AEK",
    "aek": "AEK",
    "panathinaikos": "Panathinaikos",
    "red bull salzburg": "Salzburg",
    "salzburg": "Salzburg",
    "rapid wien": "Rapid Wien",
    "rapid vienna": "Rapid Wien",
    "austria wien": "Austria Vienna",
    "sturm graz": "Sturm Graz",
    "lask": "LASK",
    "lask linz": "LASK",
    "copenhagen": "FC Copenhagen",
    "paks": "Paksi SE",
    "red star belgrade": "Crvena Zvezda",
    "red star beograd": "Crvena Zvezda",
    "crvena zvezda": "Crvena Zvezda",
    "kobenhavn": "FC Copenhagen",
    "midtjylland": "Midtjylland",
    "brondby": "Brondby",
    "malmo": "Malmo FF",
    "malmo ff": "Malmo FF",
    "bodo glimt": "Bodo/Glimt",
    "molde": "Molde",
    "rosenborg": "Rosenborg",
    "legia warszawa": "Legia",
    "legia warsaw": "Legia",
    "lech poznan": "Lech",
    "steaua bucuresti": "FCSB",
    "fcsb": "FCSB",
    "cfr 1907 cluj": "CFR Cluj",
    "cfr cluj": "CFR Cluj",
    "zenit petersburg": "Zenit",
    "zenit st petersburg": "Zenit",
    "zenit": "Zenit",
    "cska moskva": "CSKA Moscow",
    "cska moscow": "CSKA Moscow",
    "spartak moskva": "Spartak Moscow",
    "spartak moscow": "Spartak Moscow",
    "lokomotiv moskva": "Lokomotiv Moscow",
    "lokomotiv moscow": "Lokomotiv Moscow",
    "dinamo moskva": "Dynamo Moskva",
    "dynamo moscow": "Dynamo Moskva",
    "krasnodar": "Krasnodar",
    "rubin kazan": "Rubin Kazan",
    "anderlecht": "Anderlecht",
    "rsc anderlecht": "Anderlecht",
    "standard liege": "Standard",
    "genk": "Genk",
    "racing genk": "Genk",
    "gent": "Gent",
    "kaa gent": "Gent",
    "royale union saint gilloise": "St. Gilloise",
    "union saint gilloise": "St. Gilloise",
    "union sg": "St. Gilloise",
    "royal antwerp": "Antwerp",
    "antwerp": "Antwerp",
    "young boys": "Young Boys",
    "bsc young boys": "Young Boys",
    "basel": "Basel",
    "basel 1893": "Basel",
    "zurich": "Zurich",
    "shamrock rovers": "Shamrock Rovers",
    "helsinki": "HJK Helsinki",
    "hjk": "HJK Helsinki",
    "hjk helsinki": "HJK Helsinki",
    "kuopion palloseura": "KuPS",
    "kuopion ps": "KuPS",
    "kups": "KuPS",
}


def clean_raw(name: str) -> str:
    """Strip the openfootball ``(CCC)`` suffix and surrounding whitespace."""
    return re.sub(r"\s*\([A-Z]{3}\)\s*$", "", name).strip()


def normalise(name: str) -> str:
    """Accent-free, lower-case, punctuation-free name without FC/CF/AC/... tokens."""
    text = unicodedata.normalize("NFKD", clean_raw(name))
    text = text.replace("ø", "o").replace("Ø", "o").replace("ß", "ss").replace("ı", "i")
    text = text.replace("ł", "l").replace("Ł", "l").replace("đ", "d").replace("æ", "ae")
    text = text.encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [_SYNONYMS.get(w, w) for w in text.split() if w not in _STOPWORDS]
    return " ".join(words).strip()


def _strip_digits(text: str) -> str:
    return re.sub(r"\s+", " ", _DIGITS.sub(" ", text)).strip()


def source_country(code: str | None) -> str | None:
    if not code:
        return None
    return SOURCE_COUNTRY_CODES.get(code, code)


@dataclass
class TeamNameMapper:
    """Maps raw names to canonical candidates; remembers misses for logging."""

    candidates: Mapping[str, str | None]  # canonical name -> country code (or None)
    aliases: Mapping[str, str] = field(default_factory=lambda: dict(MANUAL_ALIASES))
    misses: dict[str, str | None] = field(default_factory=dict)
    unmapped: set[str] = field(default_factory=set)  # raw names registered by resolve()

    def __post_init__(self) -> None:
        self.candidates = dict(self.candidates)
        self.aliases = {normalise(k) or k: v for k, v in self.aliases.items()}
        self._by_fold: dict[str, str] = {}
        self._by_norm: dict[str, list[str]] = {}
        self._norm_list: dict[str | None, dict[str, str]] = {}
        self._all_norm: dict[str, str] = {}
        self._cache: dict[tuple[str, str | None], str | None] = {}
        for cand in list(self.candidates):
            self._index(cand)

    def _index(self, cand: str) -> None:
        country = self.candidates[cand]
        norm = normalise(cand)
        self._by_fold.setdefault(cand.casefold(), cand)
        for key in {norm, _strip_digits(norm)}:
            if key:
                self._by_norm.setdefault(key, []).append(cand)
        self._norm_list.setdefault(country, {}).setdefault(norm or cand.casefold(), cand)
        self._all_norm.setdefault(norm or cand.casefold(), cand)

    def add(self, name: str, country: str | None = None) -> None:
        """Register a new canonical name (e.g. a club outside ClubElo's coverage)."""
        if name in self.candidates:
            return
        self.candidates[name] = source_country(country)
        self._index(name)
        self._cache = {k: v for k, v in self._cache.items() if v is not None}

    def resolve(self, raw: str, country: str | None = None) -> tuple[str, bool]:
        """(canonical name, mapped?) — unmapped names are registered under their clean raw name."""
        hit = self.map(raw, country)
        if hit is not None:
            return hit, hit not in self.unmapped
        name = clean_raw(raw)
        self.add(name, country)
        self.unmapped.add(name)
        return name, False

    def map(self, raw: str, country: str | None = None) -> str | None:
        key = (raw, country)
        if key not in self._cache:
            self._cache[key] = self._map(raw, source_country(country))
            if self._cache[key] is None:
                self.misses[clean_raw(raw)] = country
        return self._cache[key]

    def _pick(self, names: Iterable[str], country: str | None) -> str | None:
        names = list(dict.fromkeys(names))
        if country is not None:
            same = [n for n in names if self.candidates.get(n) in (country, None)]
            names = same or ([] if len(names) > 1 else names)
        return names[0] if len(names) == 1 else None

    def _map(self, raw: str, country: str | None) -> str | None:
        name = clean_raw(raw)
        norm = normalise(name)
        alias = self.aliases.get(norm) or self.aliases.get(_strip_digits(norm))
        if alias and (alias in self.candidates or not self.candidates):
            return alias
        if name in self.candidates:
            return name
        if name.casefold() in self._by_fold:
            return self._by_fold[name.casefold()]
        for key in (norm, _strip_digits(norm)):
            hit = self._pick(self._by_norm.get(key, []), country)
            if hit:
                return hit
        return self._fuzzy(norm, country)

    def _fuzzy(self, norm: str, country: str | None) -> str | None:
        if not norm:
            return None
        if country is not None:
            pool = {**self._norm_list.get(country, {}), **self._norm_list.get(None, {})}
            threshold = FUZZY_THRESHOLD
        else:
            pool = self._all_norm
            threshold = FUZZY_THRESHOLD_NO_COUNTRY
        if not pool:
            return None
        hits = process.extract(norm, list(pool), scorer=fuzz.token_set_ratio, limit=2)
        if not hits or hits[0][1] < threshold:
            return None
        if len(hits) > 1 and hits[1][1] == hits[0][1]:
            # tie: prefer the closer full-string match, give up when still ambiguous
            a, b = (fuzz.ratio(norm, h[0]) for h in hits[:2])
            if a == b:
                return None
            return pool[hits[0][0] if a > b else hits[1][0]]
        return pool[hits[0][0]]


def log_misses(mapper: TeamNameMapper, source: str) -> None:
    if mapper.misses:
        names = ", ".join(f"{n} ({c})" if c else n for n, c in sorted(mapper.misses.items()))
        log.info("%s: %d unmapped team names: %s", source, len(mapper.misses), names)
