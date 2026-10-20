import numpy as np
import pytest

from matchpredict.predictor.metrics import devig, rps


def test_rps_perfect_prediction_is_zero():
    assert rps(np.array([[1.0, 0, 0], [0, 0, 1.0]]), np.array([0, 2])) == 0


def test_rps_worst_prediction_is_one():
    assert rps(np.array([[0, 0, 1.0]]), np.array([0])) == pytest.approx(1.0)


def test_rps_penalises_distance_ordinally():
    near = rps(np.array([[0, 1.0, 0]]), np.array([0]))
    far = rps(np.array([[0, 0, 1.0]]), np.array([0]))
    assert near == pytest.approx(0.5)
    assert far > near


def test_devig_normalises_and_handles_missing():
    p = devig(np.array([[2.0, 3.5, 4.0], [np.nan, 3.0, 3.0]]))
    assert p[0].sum() == pytest.approx(1.0)
    assert p[0, 0] > p[0, 2]
    assert np.isnan(p[1]).all()
