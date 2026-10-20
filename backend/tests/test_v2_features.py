"""v2 names / data frame / feature table / ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from v2_kit import synthetic_frame

from matchpredict.predictor.v2.data import (
    build_canonicaliser,
    dedupe_european,
    domestic_leagues,
    prepare_frame,
    read_db,
    source_countries,
    standardise,
)
from matchpredict.predictor.v2.ensemble import fit_pool, pool_gain, reliability
from matchpredict.predictor.v2.features import (
    CLUBELO_FEATURES,
    LEAGUE_SCALE,
    V1_FEATURES,
    DomesticEngine,
    FeatureSet,
    FeatureState,
    LegBook,
    build_table,
    fixture_row,
)
from matchpredict.predictor.v2.global_rating import ClubEloBook, GlobalRating
from matchpredict.predictor.v2.names import Canonicaliser, league_country


def test_canonicaliser_aliases_variants_and_countries():
    domestic = {"SK Rapid": "AUT", "Arsenal": "E0", "Lech Poznan": "POL"}
    names = Canonicaliser.build(
        domestic,
        ["Rapid Wien", "Omonia Nikosia", "Omonia Nikosia", "Omonia Nicosia", "Ferencvárosi TC", "Lech"],
        {"Omonia Nikosia (CYP)": "CYP", "Dinamo Zagreb": "CRO"},
    )
    assert names.canonical("Rapid Wien") == "SK Rapid"
    assert names.canonical("Lech") == "Lech Poznan"
    assert names.canonical("Omonia Nicosia") == "Omonia Nikosia"
    assert names.canonical("Ferencvárosi TC") == "Ferencváros"
    assert names.country("Arsenal") == "ENG" and names.country("SK Rapid") == "AUT"
    assert names.country("Omonia Nicosia") == "CYP"
    assert names.country("Ferencvárosi TC") == "HUN"  # manual table
    assert league_country("UCL") is None and league_country("XX") is None and league_country("SP1") == "ESP"


def test_standardise_dedupe_and_prepare():
    raw = synthetic_frame(seasons=[2020], euro_per_season=2)
    dup = raw[raw["league"] == "UCL"].iloc[[0]].assign(source="thesportsdb")
    near = dup.assign(date=dup["date"] + pd.Timedelta(days=1))
    raw = pd.concat([raw, dup, near], ignore_index=True)
    frame = standardise(raw)
    assert set(frame["comp"]) == {"DOM", "UCL"}
    assert (frame.loc[frame["comp"] == "DOM", "stage"] == "dom").all()
    deduped = dedupe_european(frame)
    assert (deduped["comp"] == "UCL").sum() == 2
    assert set(deduped.loc[deduped["comp"] == "UCL", "source"]) == {"openfootball"}
    names = build_canonicaliser(frame)
    prepared = prepare_frame(raw, names)
    assert prepared["date"].is_monotonic_increasing
    assert domestic_leagues(frame)["A1"] == "E0"
    assert source_countries(None, None) == {}


def test_read_db_reads_finished_matches(tmp_path):
    from matchpredict.db import repository
    from matchpredict.db.session import create_session_factory

    url = f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
    factory = create_session_factory(url)
    with factory() as session:
        repository.upsert_matches(session, synthetic_frame(seasons=[2020], euro_per_season=2).drop(columns=["stage", "source"]))
    frame = read_db(url)
    assert len(frame) == 182 and {"stage", "source", "status"} <= set(frame.columns)


def _state(frame: pd.DataFrame, book: ClubEloBook | None = None) -> tuple[FeatureState, Canonicaliser]:
    names = build_canonicaliser(standardise(frame))
    return FeatureState(GlobalRating(countries=names.countries, clubelo=book), DomesticEngine(), clubelo=book), names


def test_feature_table_blocks_and_european_rows():
    raw = synthetic_frame(seasons=[2019, 2020])
    state, names = _state(raw)
    frame = prepare_frame(raw, names)
    table = build_table(frame, state)
    assert len(table) == len(frame)
    for col in FeatureSet().columns(market=True):
        assert col in table.columns
    euro = table["is_euro"] > 0
    assert euro.sum() == 24
    assert table.loc[euro, LEAGUE_SCALE].isna().all().all()
    assert table.loc[euro, "mkt_home"].isna().all()
    assert table.loc[~euro, "mkt_home"].notna().all()
    assert table.loc[~euro, "elo_home"].notna().all()
    assert table[CLUBELO_FEATURES].isna().all().all()  # no ClubElo book
    # UEFA results feed form / rest, not the per-league v1 Elo
    assert state.domestic.played["A1"] > 36
    assert "UCL" not in state.domestic.league_teams


def test_feature_set_ablation_columns():
    full = FeatureSet()
    v1 = FeatureSet(global_elo=False, global_goals=False, clubelo=False, context=False)
    assert v1.columns() == V1_FEATURES + ["league"]
    assert v1.cat_columns() == ["league"]
    assert "g_elo_diff" in full.columns() and "stage" in full.cat_columns()
    assert "mkt_home" in full.columns(market=True)


def test_fixture_row_matches_the_batch_row_for_an_upcoming_match():
    raw = synthetic_frame(seasons=[2019, 2020])
    snaps = pd.DataFrame({"date": ["2019-07-01"], "club": ["A1"], "elo": [1700.0]})
    book = ClubEloBook(snaps)
    state, names = _state(raw, book)
    frame = prepare_frame(raw, names)
    build_table(frame, state)
    day = frame["date"].max() + pd.Timedelta(days=3)
    serving = fixture_row(state, "A1", "A2", "E0", day, "dom", False, (2.0, 3.4, 4.0))
    upcoming = frame.iloc[[0]].copy()
    upcoming[["date", "home", "away", "home_goals", "away_goals"]] = [day, "A1", "A2", np.nan, np.nan]
    upcoming[["odds_home", "odds_draw", "odds_away"]] = [2.0, 3.4, 4.0]
    batch = build_table(upcoming.reset_index(drop=True), state)
    cols = FeatureSet().columns(market=True)
    pd.testing.assert_frame_equal(serving[cols].reset_index(drop=True), batch[cols].reset_index(drop=True), check_dtype=False)
    assert serving.loc[0, "ce_home"] == 1700.0


def test_leg_book_detects_second_legs():
    legs = LegBook()
    legs.add("2021-02-16", "UCL", "A1", "B1", 2.0, 0.0)
    assert legs.context("B1", "A1", "UCL", "ko", "2021-03-09") == (1.0, -2.0)
    assert legs.context("B1", "A1", "UCL", "league", "2021-03-09") == (0.0, 0.0)
    assert legs.context("B1", "A1", "UCL", "ko", "2021-05-30") == (0.0, 0.0)
    assert legs.context("B1", "A1", "E0", "ko", "2021-03-09") == (0.0, 0.0)
    legs.add("2021-06-30", "UCL", "C", "D", 1.0, 1.0)
    assert len(legs.recent) == 1  # old entries pruned


def test_log_pool_recovers_temperature_and_prefers_the_better_member():
    rng = np.random.default_rng(0)
    n = 4000
    logits = rng.normal(size=(n, 3))
    true = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in true])
    flat = np.exp(logits / 2) / np.exp(logits / 2).sum(axis=1, keepdims=True)  # under-confident
    noise = np.full((n, 3), 1 / 3)
    pool = fit_pool(["flat", "noise"], [flat, noise], y)
    assert pool.weights[0] == pytest.approx(2.0, rel=0.15)
    gains = pool_gain(pool, [flat, noise], y)
    assert gains["pool"] < gains["flat"] < gains["noise"]
    assert pool.describe()["members"]["flat"] > 1.5
    rel = reliability(pool.predict([flat, noise]), y)
    assert set(rel) == {"home", "draw", "away"} and sum(r["n"] for r in rel["home"]) == n
    with pytest.raises(RuntimeError):
        type(pool)(names=["x"]).predict([flat])
