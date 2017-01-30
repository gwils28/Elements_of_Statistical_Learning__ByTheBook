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


def test_bspline_penalty_exact_on_polynomials(smooth_data):
    _, _, knots = smooth_data
    Om = inf.bspline_penalty(knots)
    xs = np.linspace(0, 3, 60)
    H = inf.bspline_basis(xs, knots)
    # coefficients représentant exactement 1, x et x^2 (polynômes de degré <= 3)
    for f, integral in [(lambda t: np.ones_like(t), 0.0), (lambda t: t, 0.0),
                        (lambda t: t ** 2, 4.0 * 3)]:          # int_0^3 (2)^2 = 12
        c = np.linalg.lstsq(H, f(xs), rcond=None)[0]
        assert np.allclose(H @ c, f(xs), atol=1e-10)
        assert c @ Om @ c == pytest.approx(integral, abs=1e-8)
    assert np.allclose(Om, Om.T)
    assert np.linalg.eigvalsh(Om).min() > -1e-8
