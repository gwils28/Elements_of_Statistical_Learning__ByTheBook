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
