import numpy as np
import pytest

from esl import ch02_supervise as c2
from esl import datasets


def test_least_squares_matches_normal_equations():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 3))
    y = 1 + X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=50)
    beta = c2.least_squares_fit(X, y)
    A = c2.add_intercept(X)
    np.testing.assert_allclose(beta, np.linalg.solve(A.T @ A, A.T @ y))
    # résidus orthogonaux aux colonnes (éq. (2.5))
    np.testing.assert_allclose(A.T @ (y - c2.least_squares_predict(X, beta)), 0, atol=1e-10)
