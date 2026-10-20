import numpy as np
import pandas as pd
import pytest

from matchpredict.predictor.features import (
    BASE_FEATURES,
    CAT_FEATURES,
    MARKET_FEATURES,
    build_feature_table,
    market_features,
)
from matchpredict.predictor.metrics import ece, log_loss
from matchpredict.predictor.model import OutcomeModel, TemperatureScaler, softmax


def _logits_dataset(n=4000, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(0, 1, (n, 3))
    p = softmax(true_logits)
    y = np.array([rng.choice(3, p=row) for row in p])
    return true_logits * scale, y  # overconfident by factor `scale`


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[1, 0] == pytest.approx(1.0)


def test_temperature_scaler_fixes_overconfidence():
    logits, y = _logits_dataset()
    scaler = TemperatureScaler().fit(logits, y)
    assert scaler.temperature == pytest.approx(3.0, rel=0.15)
    assert log_loss(scaler.transform(logits), y) < log_loss(softmax(logits), y)


def test_ece_is_small_for_calibrated_and_large_for_overconfident():
    logits, y = _logits_dataset(n=20000, scale=1.0)
    good = ece(softmax(logits), y)
    bad = ece(softmax(logits * 4), y)
    assert len(good) == 3
    assert max(good) < 0.02
    assert min(bad) > max(good)


def test_market_features_devig_and_missing():
    feats = market_features(np.array([[2.0, 3.5, 4.0], [np.nan, 3.0, 3.0]]))
    assert list(feats.columns) == [c for c in MARKET_FEATURES if c not in BASE_FEATURES]
    assert feats.iloc[0, :3].sum() == pytest.approx(1.0)
    assert feats.iloc[1].isna().all()


def _history(n_rounds=40, seed=3):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    strength = dict(zip(teams, rng.normal(0, 0.5, len(teams))))
    rows = []
    day = pd.Timestamp("2018-08-01")
    for r in range(n_rounds):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            lh = np.exp(0.35 + strength[h] - strength[a])
            la = np.exp(0.05 + strength[a] - strength[h])
            rows.append((day + pd.Timedelta(days=7 * r), h, a, rng.poisson(lh), rng.poisson(la)))
    df = pd.DataFrame(rows, columns=["date", "home", "away", "home_goals", "away_goals"])
    df["league"] = "E0"
    df["time"] = None
    for c in ("odds_home", "odds_draw", "odds_away"):
        df[c] = 3.0
    return df.astype({"home_goals": float, "away_goals": float})


def test_build_feature_table_columns():
    df = _history(n_rounds=5)
    table, engine = build_feature_table(df)
    assert set(MARKET_FEATURES) <= set(table.columns)
    assert len(table) == len(df)
    assert table["league"].iloc[0] == "E0"
    assert engine.played["T0"] == 5


def test_outcome_model_fit_predict_and_calibrate():
    df = _history()
    table, _ = build_feature_table(df)
    y = np.where(df.home_goals > df.away_goals, 0, np.where(df.home_goals == df.away_goals, 1, 2))
    split = len(df) * 3 // 4
    model = OutcomeModel(BASE_FEATURES, CAT_FEATURES, iterations=50, learning_rate=0.1)
    model.fit(table.iloc[:split], y[:split], table.iloc[split:], y[split:])
    probs = model.predict_proba(table.iloc[split:])
    assert probs.shape == (len(df) - split, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert model.temperature > 0
    assert model.best_iteration >= 1
    refit = model.refit_full(table, y)
    assert refit.temperature == model.temperature
    assert refit.predict_proba(table.iloc[:2]).shape == (2, 3)


def test_outcome_model_requires_fit():
    with pytest.raises(RuntimeError):
        OutcomeModel(BASE_FEATURES, CAT_FEATURES).predict_proba(pd.DataFrame())


def test_outcome_model_with_market_baseline_learns_residual():
    df = _history()
    table, _ = build_feature_table(df)
    rng = np.random.default_rng(5)
    table[["mkt_home", "mkt_draw", "mkt_away"]] = rng.dirichlet([4, 3, 3], len(table))
    table.loc[table.index[:5], "mkt_home"] = np.nan  # rows without odds are skipped in training
    y = np.where(df.home_goals > df.away_goals, 0, np.where(df.home_goals == df.away_goals, 1, 2))
    split = len(df) * 3 // 4
    model = OutcomeModel(BASE_FEATURES, CAT_FEATURES, iterations=30, learning_rate=0.1,
                         baseline_cols=["mkt_home", "mkt_draw", "mkt_away"])
    model.fit(table.iloc[:split], y[:split], table.iloc[split:], y[split:])
    probs = model.predict_proba(table.iloc[split:])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    refit = model.refit_full(table, y)
    assert np.isfinite(refit.predict_proba(table.iloc[10:12])).all()
