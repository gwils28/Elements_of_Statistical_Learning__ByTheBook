import numpy as np
import pytest
from scipy import stats

from esl import ch18_grande_dimension_complements as m


def test_sigmoid_stable():
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    s = m.sigmoid(z)
    assert np.all(np.isfinite(s))
    assert np.allclose(s, 1 / (1 + np.exp(-np.clip(z, -50, 50))))


def test_separable_loglik_tends_to_zero():
    x = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    vals = [m.logistic_loglik(0.0, np.array([b]), x, y) for b in (1, 10, 100)]
    assert vals[0] < vals[1] < vals[2] < 0
    assert vals[2] > -1e-10
    it, path, ll = m.logistic_gradient_ascent(x, y, n_iter=2000)
    assert np.all(np.diff(path[:, 1]) > 0)        # beta croît sans cesse
    assert np.all(np.diff(ll) > 0)
