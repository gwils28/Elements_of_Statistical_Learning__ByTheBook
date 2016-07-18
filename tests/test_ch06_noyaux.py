import numpy as np
import pytest
from scipy import integrate, stats

from esl import ch06_noyaux as nk


def _data(n=80, seed=0):
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(4 * x) + rng.normal(0, 1 / 3, n)
    return x, y


def test_kernels_integrals():
    assert integrate.quad(nk.epanechnikov, -1, 1)[0] == pytest.approx(1.0)
    assert integrate.quad(nk.gaussian, -np.inf, np.inf)[0] == pytest.approx(1.0)
    assert integrate.quad(nk.tricube, -1, 1)[0] == pytest.approx(81 / 70)
    assert nk.epanechnikov(1.5) == 0 and nk.tricube(-1.2) == 0


def test_nw_equals_local_constant():
    x, y = _data()
    x0 = np.linspace(0.05, 0.95, 15)
    a = nk.nadaraya_watson(x0, x, y, lam=0.2, kernel="tricube")
    b = nk.local_regression(x0, x, y, lam=0.2, degree=0, kernel="tricube")
    assert np.allclose(a, b)
