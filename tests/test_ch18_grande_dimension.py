import numpy as np
import pytest
from scipy import optimize, stats

from esl import ch18_grande_dimension as g


def _classif_data(rng, N=40, p=60, K=3):
    y = np.repeat(np.arange(K), N // K)
    X = rng.randn(len(y), p)
    X[:, :5] += y[:, None]
    return X, y


def test_soft_threshold():
    x = np.array([-3.0, -0.5, 0.0, 0.2, 2.0])
    np.testing.assert_allclose(g.soft_threshold(x, 1.0), [-2, 0, 0, 0, 1])
