import numpy as np
import pandas as pd
import pytest

from matchpredict.predictor.goals import (
    MAX_GOALS,
    GoalsModel,
    btts,
    dc_tau,
    fit_rho,
    outcome_probs,
    over_line,
    rescale_to_1x2,
    score_matrix,
    top_scores,
)


def test_score_matrix_shape_and_normalised():
    m = score_matrix(np.array([1.5, 0.8]), np.array([1.1, 2.0]), rho=-0.1)
    assert m.shape == (2, MAX_GOALS + 1, MAX_GOALS + 1)
    np.testing.assert_allclose(m.sum(axis=(1, 2)), 1.0, atol=1e-9)
    assert (m >= 0).all()


def test_rho_zero_is_independent_poisson():
    m = score_matrix(np.array([1.0]), np.array([1.0]), rho=0.0)[0]
    assert m[0, 0] == pytest.approx(np.exp(-2.0), rel=1e-6)


def test_negative_rho_inflates_draws_00_and_11():
    base = score_matrix(np.array([1.3]), np.array([1.1]), rho=0.0)[0]
    dc = score_matrix(np.array([1.3]), np.array([1.1]), rho=-0.1)[0]
    assert dc[0, 0] > base[0, 0] and dc[1, 1] > base[1, 1]
    assert dc[1, 0] < base[1, 0]


def test_dc_tau_identity_outside_low_scores():
    assert dc_tau(np.array([3]), np.array([2]), np.array([1.2]), np.array([1.0]), -0.1)[0] == 1.0


def test_fit_rho_recovers_negative_dependence():
    rng = np.random.default_rng(0)
    lh, la = np.full(20000, 1.3), np.full(20000, 1.1)
    true = score_matrix(lh[:1], la[:1], rho=-0.12)[0].ravel()
    idx = rng.choice(true.size, size=20000, p=true / true.sum())
    hg, ag = np.divmod(idx, MAX_GOALS + 1)
    assert fit_rho(hg, ag, lh, la) == pytest.approx(-0.12, abs=0.03)


def test_market_summaries():
    m = score_matrix(np.array([1.4]), np.array([1.2]), rho=-0.05)[0]
    probs = outcome_probs(m)
    assert probs.sum() == pytest.approx(1.0)
    assert 0 < over_line(m, 2.5) < 1
    assert 0 < btts(m) < 1
    tops = top_scores(m, k=5)
    assert len(tops) == 5
    assert tops[0][1] >= tops[-1][1]
    assert tops[0][0] in {"1-1", "1-0"}


def test_rescale_to_1x2_matches_target():
    m = score_matrix(np.array([1.4]), np.array([1.2]), rho=-0.05)[0]
    target = np.array([0.6, 0.25, 0.15])
    r = rescale_to_1x2(m, target)
    np.testing.assert_allclose(outcome_probs(r), target, atol=1e-9)
    assert r.sum() == pytest.approx(1.0)


def _synthetic(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, n)
    lh = np.exp(0.3 + 0.4 * strength)
    la = np.exp(0.1 - 0.4 * strength)
    return (
        pd.DataFrame({"diff": strength, "noise": rng.normal(size=n), "league": rng.choice(["E0", "D1"], n)}),
        rng.poisson(lh),
        rng.poisson(la),
    )


def test_goals_model_learns_strength_direction():
    X, hg, ag = _synthetic()
    X["empty"] = np.nan  # all-missing columns (e.g. xG history) are ignored
    model = GoalsModel(feature_cols=["diff", "noise", "empty"], cat_cols=["league"], max_iter=60).fit(X, hg, ag)
    assert model.used_cols == ["diff", "noise"]
    new = pd.DataFrame({"diff": [2.0, -2.0], "noise": [0, 0], "empty": [1.0, 1.0], "league": ["E0", "XX"]})
    lh, la = model.predict_lambdas(new)
    assert lh[0] > lh[1] and la[0] < la[1]
    assert -0.3 < model.rho < 0.3
    matrices = model.score_matrices(X.iloc[:3])
    assert matrices.shape == (3, MAX_GOALS + 1, MAX_GOALS + 1)


def test_goals_model_requires_fit():
    with pytest.raises(RuntimeError):
        GoalsModel(feature_cols=["diff"]).predict_lambdas(pd.DataFrame({"diff": [1.0]}))
