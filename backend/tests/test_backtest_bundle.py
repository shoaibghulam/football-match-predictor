import datetime as dt

from mp_testkit import frame, memory_factory
from sqlalchemy import insert, select

from matchpredict.db import backtest_bundle as bb
from matchpredict.db import repository as repo
from matchpredict.db.models import Match
from matchpredict.db.results_models import BacktestPrediction

SOURCE_MATCHES = [
    {"date": "2024-08-17", "league": "E0", "home": "Nottm Forest", "away": "Bournemouth", "home_goals": 1, "away_goals": 1},
    {"date": "2024-08-17", "league": "E0", "home": "Arsenal", "away": "Wolves", "home_goals": 2, "away_goals": 0},
    {"date": "2024-03-07", "league": "ARG", "home": "Tigre", "away": "Lanus", "home_goals": 0, "away_goals": 1},
    {"date": "2024-08-20", "league": "UCL", "home": "Celtic", "away": "Ajax", "home_goals": 1, "away_goals": 2},
    {"date": "2024-08-24", "league": "E0", "home": "Chelsea", "away": "Everton", "home_goals": 3, "away_goals": 0},
]  # fmt: skip


def _source_db():
    factory = memory_factory()
    with factory() as s:
        repo.upsert_matches(s, frame(SOURCE_MATCHES))
        ids = {m.home: m.id for m in s.scalars(select(Match))}
        rows = [
            {"match_id": ids[home], "season": 2024, "model": "market", "model_version": "v2-backtest-2024/25",
             "prob_home": p, "prob_draw": 0.25, "prob_away": round(0.75 - p, 6)}
            for home, p in (("Nottm Forest", 0.412345), ("Arsenal", 0.7), ("Tigre", 0.3), ("Celtic", 0.2),
                            ("Chelsea", 0.6))
        ]  # fmt: skip
        s.execute(insert(BacktestPrediction), rows)
        s.commit()
    return factory


def _target_db():
    """Fresh-install spellings: Nott'm Forest, ARG date not swapped, UCL a day later, no Chelsea game."""
    factory = memory_factory()
    with factory() as s:
        repo.upsert_matches(s, frame([
            {"date": "2024-08-17", "league": "E0", "home": "Nott'm Forest", "away": "Bournemouth", "home_goals": 1, "away_goals": 1},
            {"date": "2024-08-17", "league": "E0", "home": "Arsenal", "away": "Wolves", "home_goals": 2, "away_goals": 0},
            {"date": "2024-07-03", "league": "ARG", "home": "Tigre", "away": "Lanus", "home_goals": 0, "away_goals": 1},
            {"date": "2024-08-21", "league": "UCL", "home": "Celtic", "away": "Ajax", "home_goals": 1, "away_goals": 2},
        ]))  # fmt: skip
    return factory


def test_export_then_import_joins_by_key_with_fallbacks(tmp_path):
    path = tmp_path / "bt.csv.gz"
    assert bb.export_bundle(_source_db(), path) == 5
    bundle = bb.read_bundle(path)
    assert list(bundle.columns) == bb.COLUMNS and "match_id" not in bundle.columns
    assert bundle.loc[bundle["home"] == "Nottm Forest", "prob_home"].item() == 0.4123  # 4 dp

    target = _target_db()
    result = bb.import_bundle(target, path)
    assert result == bb.ImportResult("imported", rows=5, joined=4, unmatched=1)  # Chelsea game absent
    with target() as s:
        got = {s.get(Match, b.match_id).home: b.prob_home for b in s.scalars(select(BacktestPrediction))}
    assert got == {"Nott'm Forest": 0.4123, "Arsenal": 0.7, "Tigre": 0.3, "Celtic": 0.2}

    assert bb.import_bundle(target, path).status == "skipped"  # idempotent
    with target() as s:
        assert len(s.scalars(select(BacktestPrediction)).all()) == 4

    with target() as s:  # the first ingest adds the missing match -> re-import picks it up
        repo.upsert_matches(s, frame([SOURCE_MATCHES[-1]]))
    again = bb.import_bundle(target, path)
    assert again.status == "imported" and again.joined == 5 and again.unmatched == 0
    assert bb.import_bundle(target, path) == bb.ImportResult("skipped", rows=5, joined=5)


def test_import_without_bundle_or_rows(tmp_path):
    assert bb.import_bundle(memory_factory(), tmp_path / "missing.csv.gz").status == "empty"
    empty = tmp_path / "empty.csv.gz"
    bb.export_bundle(memory_factory(), empty)
    assert bb.import_bundle(memory_factory(), empty).status == "empty"


def test_ambiguous_partial_matches_are_not_used():
    import pandas as pd

    rows = pd.DataFrame([{"league": "E0", "date": dt.date(2024, 8, 17), "home": "X", "away": "Wolves", "season": 2024,
                          "model": "base", "model_version": "v", "prob_home": 0.3, "prob_draw": 0.3, "prob_away": 0.4}])  # fmt: skip
    matches = pd.DataFrame([
        {"match_id": 1, "league": "E0", "date": dt.date(2024, 8, 17), "home": "A", "away": "Wolves"},
        {"match_id": 2, "league": "E0", "date": dt.date(2024, 8, 17), "home": "B", "away": "Wolves"},
    ])  # fmt: skip
    assert bb.join_matches(rows, matches)["match_id"].isna().all()
    assert bb._swap_day_month(dt.date(2024, 3, 31)) is None
