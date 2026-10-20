import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from europe_kit import make_service, table_and_book
from mp_testkit import frame

from matchpredict.predictor import europe_train as T
from matchpredict.predictor.europe import (
    EloBook,
    EloConfig,
    EuropeService,
    MatchContext,
    elo_baseline_probs,
    elo_expectation,
    feature_row,
    infer_stage,
    margin_multiplier,
    rescale_to_1x2,
    score_matrix,
    summarise_matrix,
)

D = pd.Timestamp


@pytest.fixture(scope="module")
def service():
    return make_service()


def test_elo_helpers():
    assert elo_expectation(0) == pytest.approx(0.5)
    assert elo_expectation(400) == pytest.approx(10 / 11)
    assert [margin_multiplier(g) for g in (0, 1, 2, 3, 5)] == [1.0, 1.0, 1.5, 1.75, 2.0]
    assert infer_stage("UCL", D("2026-08-01")) == "qual"
    assert infer_stage("UCL", D("2026-10-21")) == "league"
    assert infer_stage("UEL", D("2027-03-10")) == "ko"
    assert infer_stage("E0", D("2027-03-10")) == "league"


def test_elo_book_snapshot_priors_and_timestamps():
    book = EloBook(EloConfig())
    book.apply_snapshot(D("2026-09-01"), [("Barcelona", "ESP", 1970.0), ("Paris SG", "FRA", 1920.0)])
    assert book.covered == {"Barcelona", "Paris SG"}
    # matches older than the snapshot do not move covered ratings
    assert book.update(D("2026-08-01"), "Barcelona", "Paris SG", 3, 0) is False
    assert book.update(D("2026-10-01"), "Barcelona", "Paris SG", 1, 2) is True
    assert book.ratings["Barcelona"] < 1970 < 1920 + 60 or book.ratings["Paris SG"] > 1920
    # uncovered priors: stage default, then country peers
    assert book.ensure("Kairat", "KAZ", "qual") == 1300.0
    assert book.ensure("Astana", "KAZ", "league") == pytest.approx(1300.0)
    assert book.ensure("New", None, "ko") == 1600.0
    assert book._k("Kairat") == 40.0 and book._k("Barcelona") == 20.0
    clone = EloBook.from_state(book.state())
    assert clone.ratings == book.ratings and clone.covered == book.covered


def test_feature_row_and_matrix_helpers():
    ctx = MatchContext("UECL", "ko", neutral=True, leg2=True, agg_gd=5, domestic=False)
    row = feature_row(1900, 1700, False, True, ctx)
    assert row["elo_diff"] == 2.0 and row["diff_x_unc"] == 2.0 and row["agg_gd"] == 3.0
    assert row["comp_uecl"] == 1.0 and row["st_ko"] == 1.0 and row["neutral"] == 1.0
    m = score_matrix(1.5, 1.0)
    assert m.sum() == pytest.approx(1.0)
    target = np.array([0.5, 0.3, 0.2])
    r = rescale_to_1x2(m, target)
    idx = np.arange(r.shape[0])
    assert r[idx[:, None] > idx[None, :]].sum() == pytest.approx(0.5)
    assert np.trace(r) == pytest.approx(0.3)
    summary = summarise_matrix(r)
    assert len(summary["top_scores"]) == 5 and 0 < summary["btts"] < 1
    base = elo_baseline_probs(np.array([0.0, 400.0]), np.array([True, False]), 0.25, 60)
    assert base.sum(axis=1) == pytest.approx([1, 1]) and base[1, 0] > base[0, 0]


def test_replay_marks_second_legs_and_uncovered():
    hist, table, book, euro_log = table_and_book()
    assert len(table) == len(hist) and set(T.FEATURES) <= set(table.columns)
    assert table["leg2"].sum() >= 0 and table["unc_home"].max() == 1.0
    assert "FK Kairat" in book.ratings and "FK Kairat" not in book.covered
    assert book.ratings["Arsenal"] == 2020.0  # final snapshot applied
    assert len(euro_log) == len(hist)


def test_replay_detects_two_legged_tie():
    rows = pd.DataFrame([
        {"date": D("2025-02-10"), "time": "20:00", "league": "UCL", "season": "2024-25", "stage": "ko",
         "home": "A", "away": "B", "home_country": "ESP", "away_country": "FRA",
         "home_goals": 3.0, "away_goals": 0.0, "neutral": False},
        {"date": D("2025-02-17"), "time": "20:00", "league": "UCL", "season": "2024-25", "stage": "ko",
         "home": "B", "away": "A", "home_country": "FRA", "away_country": "ESP",
         "home_goals": 1.0, "away_goals": 1.0, "neutral": False},
    ])  # fmt: skip
    table, _, _ = T.replay(rows, [])
    assert table["leg2"].tolist() == [0.0, 1.0]
    assert table["agg_gd"].tolist() == [0.0, -3.0]


def test_walk_forward_beats_base_rates_and_formats():
    _, table, _, _ = table_and_book()
    report = T.walk_forward(table, None, seasons=("2024-25", "2025-26"), variants=T.VARIANTS[:2])
    overall = report["overall"]
    assert set(overall) >= {"elo_baseline", "base_rates", "europe_lr"}
    assert overall["europe_lr"]["rps"] < overall["base_rates"]["rps"]
    assert T.best_variant(report, T.VARIANTS[:2]).name in {"europe_lr", "europe_lr_full"}
    assert "| all | europe_lr |" in T.format_report(report)
    assert T.best_variant({"overall": {}}).name == T.VARIANTS[0].name


def test_service_predict_contract_and_names(service):
    pred = service.predict("FC Barcelona", "Paris SG", "UCL", D("2026-10-21"))
    assert set(pred) == {"probs", "expected_goals", "over_2_5", "btts", "top_scores", "model",
                         "model_version", "confidence"}  # fmt: skip
    assert pred["model"] == "europe" and pred["confidence"] == "high"
    assert sum(pred["probs"].values()) == pytest.approx(1.0, abs=1e-3)
    assert pred["probs"]["home"] > pred["probs"]["away"]
    assert len(pred["top_scores"]) == 5
    strong = service.predict("Arsenal", "FK Kairat", "UCL", D("2026-10-21"))
    assert strong["probs"]["home"] > 0.6 and strong["expected_goals"]["home"] > strong["expected_goals"]["away"]
    assert service.resolve("barcelona") == "Barcelona"
    assert service.strength("Arsenal") == pytest.approx(2020.0)
    assert service.strength("Nobody United") is None
    assert service.team_ratings("Arsenal") == {"elo": pytest.approx(2020.0)}
    with pytest.raises(KeyError):
        service.team_ratings("Nobody United")
    assert service.known_teams()["Barcelona"] == "UCL"


def test_unknown_clubs_get_low_confidence_prior_in_uefa_only(service):
    pred = service.predict("Totally Unknown FC", "Benfica", "UECL", D("2026-10-22"))
    assert pred["confidence"] == "low" and pred["probs"]["away"] > pred["probs"]["home"]
    assert service.competition_prior("UECL") == pytest.approx(service.book.config.prior_main - 50)
    assert service.competition_prior("UCL") < 2000
    with pytest.raises(KeyError):
        service.predict("Totally Unknown FC", "Benfica", "E0", D("2026-10-22"))


def test_update_results_domestic_and_european_and_second_leg():
    svc = make_service()
    before = svc.book.ratings["Barcelona"]
    rows = frame([
        {"date": "2026-09-20", "league": "SP1", "home": "Barcelona", "away": "Getafe",
         "home_goals": 0, "away_goals": 3},
        {"date": "2026-08-01", "league": "SP1", "home": "Barcelona", "away": "Getafe",
         "home_goals": 5, "away_goals": 0},  # older than the snapshot: ignored for Barcelona
        {"date": "2027-02-17", "league": "UCL", "home": "Celtic", "away": "Lens",
         "home_goals": 2, "away_goals": 0},
        {"date": "2027-02-18", "league": "UCL", "home": "Paris SG", "away": "Arsenal"},  # unplayed
    ])  # fmt: skip
    applied = svc.update_results(rows)
    assert applied == 3  # Getafe is new so the August game still seeds it
    assert svc.book.ratings["Barcelona"] < before
    assert svc.book.league["Getafe"] == "SP1"
    assert svc.last_result_date == D("2027-02-17")
    ctx = svc.context("Lens", "Celtic", "UCL", D("2027-02-24"))
    assert ctx.leg2 and ctx.agg_gd == -2.0 and ctx.stage == "ko"
    assert svc.update_results(rows.iloc[0:0]) == 0


def test_save_load_roundtrip(tmp_path, service):
    path = tmp_path / "europe.joblib"
    service.save(path)
    loaded = EuropeService.load(path)
    a = service.predict("Barcelona", "Paris SG", "UCL", D("2026-10-21"))
    assert loaded.predict("Barcelona", "Paris SG", "UCL", D("2026-10-21")) == a
    import joblib

    joblib.dump({"artifact_version": 99}, path)
    with pytest.raises(ValueError):
        EuropeService.load(path)


def test_candidates_mapping_and_coverage(tmp_path):
    elo = tmp_path / "elo.csv"
    elo.write_text('"date","club","country","elo"\n"2026-09-01","Barcelona","ESP","1970"\n'
                   '"2026-09-01","Paris SG","FRA","1920"\n', encoding="utf-8")  # fmt: skip
    cands = T.candidates_from(elo, [("Getafe", "SP1"), ("Raw Club", "UCL")])
    assert cands == {"Barcelona": "ESP", "Paris SG": "FRA", "Getafe": "ESP"}
    snaps = T.load_snapshots(elo)
    assert snaps[0][0] == D("2026-09-01") and len(snaps[0][1]) == 2
    from matchpredict.ingest.teamnames import TeamNameMapper

    raw = pd.DataFrame([{"home_raw": "FC Barcelona", "home_country": "ESP", "away_raw": "FK Kairat",
                         "away_country": "KAZ"}])  # fmt: skip
    mapped = T.map_history(raw, TeamNameMapper(cands))
    assert mapped.loc[0, "home"] == "Barcelona" and mapped.loc[0, "away"] == "FK Kairat"
    cov = T.mapping_coverage(mapped, {"ESP", "FRA"})
    assert cov["covered_country_appearances_mapped_pct"] == 100.0
    assert cov["appearances_mapped_pct"] == 50.0 and cov["unmapped_countries"] == {"KAZ": 1}
    assert T.outcome_codes(np.array([2, 1, 0]), np.array([1, 1, 3])).tolist() == [0, 1, 2]


def test_domestic_rows(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("Division,MatchDate,HomeElo,AwayElo,FTHome,FTAway\nE0,2020-01-01,1800,1700,2,1\n"
                   "E0,2020-01-02,,1700,1,1\n", encoding="utf-8")  # fmt: skip
    dom = T.domestic_rows(csv)
    assert len(dom) == 1 and dom.loc[0, "domestic"] == 1.0 and dom.loc[0, "elo_diff"] == 1.0
    _, table, _, _ = table_and_book()
    model = T._fit(T.VARIANTS[2], table, dom)
    assert model.predict_proba(table.head(3)).shape == (3, 3)


def test_train_end_to_end_with_tiny_inputs(tmp_path, monkeypatch):
    cache = tmp_path / "of"
    (cache / "2024-25").mkdir(parents=True)
    lines = ["= UEFA Champions League 2024/25", "▪ League, Matchday 1", "  Tue Sep 17 2024"]
    for i in range(30):
        lines.append(f"    21:00  Club A{i % 5} (ESP) v Club B{i % 4} (FRA)  {i % 3}-{(i + 1) % 2} (0-0)")
    (cache / "2024-25" / "cl.txt").write_text("\n".join(lines), encoding="utf-8")
    elo = tmp_path / "elo.csv"
    elo.write_text('"date","club","country","elo"\n"2024-07-01","Club A1","ESP","1800"\n', encoding="utf-8")
    monkeypatch.setattr(T, "OPENFOOTBALL_DIR", cache)
    monkeypatch.setattr(T, "ELO_CSV", elo)
    monkeypatch.setattr(T, "MATCHES_CSV", tmp_path / "missing.csv")
    svc, report = T.train(download=False, use_db=False, artifact=tmp_path / "e.joblib",
                          metrics_path=tmp_path / "m.json", today=__import__("datetime").date(2026, 10, 21))  # fmt: skip
    assert (tmp_path / "e.joblib").exists() and report["per_season"] == {}
    assert json.loads((tmp_path / "m.json").read_text())["history_matches"] == 30
    assert svc.predict("Club A1", "Club B2", "UCL", D("2026-10-21"))["model"] == "europe"


def test_db_inputs_reads_sqlite(tmp_path):
    db = tmp_path / "x.db"
    from matchpredict.db import repository
    from matchpredict.db.session import create_session_factory

    factory = create_session_factory(f"sqlite:///{db.as_posix()}")
    with factory() as s:
        repository.upsert_matches(s, frame([{"date": "2026-10-01", "league": "E0", "home": "Arsenal",
                                              "away": "Chelsea", "home_goals": 1, "away_goals": 0}]))  # fmt: skip
    teams, recent = T._db_inputs(f"sqlite:///{db.as_posix()}")
    assert ("Arsenal", "E0") in teams and len(recent) == 1


ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "europe.joblib"
REAL_FIXTURES = [  # real UEFA fixtures of 2026-10-21/22 (football-data.org / TheSportsDB names)
    ("UCL", "FC Bayern München", "Arsenal FC"), ("UCL", "Real Madrid CF", "RB Leipzig"),
    ("UCL", "Como 1907", "Manchester United FC"), ("UCL", "Real Betis Balompié", "Feyenoord Rotterdam"),
    ("UCL", "Sporting Clube de Portugal", "LASK Linz"), ("UCL", "Club Brugge KV", "Racing Club de Lens"),
    ("UCL", "Lille", "Galatasaray"), ("UCL", "Aston Villa", "Viking"), ("UCL", "Inter", "Shakhtar Donetsk"),
    ("UECL", "FK Kairat", "Panathinaikos"), ("UEL", "Dinamo Zagreb", "Nijmegen"),
    ("UEL", "FC Ararat-Armenia", "AZ Alkmaar"), ("UEL", "Ferencváros", "Torreense"),
    ("UEL", "Lech", "Sunderland"), ("UEL", "Juventus", "Rennes"), ("UEL", "Olympiakos", "AC Sparta Praha"),
    ("UEL", "Omonia Nicosia", "Benfica"), ("UEL", "Celtic", "Celta"),
]  # fmt: skip


@pytest.mark.skipif(not ARTIFACT.exists(), reason="trained europe artifact not present")
def test_real_artifact_prices_every_real_uefa_fixture():
    svc = EuropeService.load(ARTIFACT)
    preds = [svc.predict(h, a, lg, D("2026-10-22")) for lg, h, a in REAL_FIXTURES]
    assert len(preds) == len(REAL_FIXTURES)  # 100% coverage
    assert all(p["model"] == "europe" and abs(sum(p["probs"].values()) - 1) < 1e-2 for p in preds)
    for name, canon in (("FC Barcelona", "Barcelona"), ("Paris Saint-Germain FC", "Paris SG"),
                        ("FC Bayern München", "Bayern Munich"), ("Lech", "Lech")):  # fmt: skip
        assert svc.resolve(name) == canon


def test_sample_fixture_coverage_is_total_even_for_unknown_clubs(service):
    names = [(lg, h, a) for lg, h, a in REAL_FIXTURES]
    preds = [service.predict(h, a, lg, D("2026-10-22")) for lg, h, a in names]
    assert len(preds) == len(names)
    assert {p["confidence"] for p in preds} <= {"high", "low"}

