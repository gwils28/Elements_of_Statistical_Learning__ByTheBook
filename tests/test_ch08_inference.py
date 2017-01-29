"""Tests du module esl.ch08_inference (chapitre 8)."""

import numpy as np
import pytest

from esl import ch08_inference as inf


@pytest.fixture
def smooth_data():
    rng = np.random.RandomState(0)
    x = np.sort(rng.uniform(0, 3, 50))
    y = 1 + 1.6 * np.sin(2.6 * x + 0.9) + 0.9 * rng.randn(50)
    knots = inf.bspline_knots(np.quantile(x, [0.25, 0.5, 0.75]), (0, 3))
    return x, y, knots


# --------------------------------------------------------------------------- B-splines

def test_bspline_partition_of_unity(smooth_data):
    x, _, knots = smooth_data
    H = inf.bspline_basis(np.linspace(0, 3, 101), knots)
    assert H.shape == (101, 7)
    assert np.all(H >= -1e-12)
    np.testing.assert_allclose(H.sum(1), 1.0, atol=1e-12)
