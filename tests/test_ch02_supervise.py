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


def test_weighted_least_squares():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    w = rng.uniform(0.5, 2, size=30)
    beta = c2.least_squares_fit(X, y, weights=w)
    A = c2.add_intercept(X)
    W = np.diag(w)
    np.testing.assert_allclose(beta, np.linalg.solve(A.T @ W @ A, A.T @ W @ y))


def test_knn_regress_limits():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=40)
    # k = 1 sur l'échantillon d'apprentissage : on retrouve y
    np.testing.assert_allclose(c2.knn_regress(X, y, X, 1), y)
    # k = N : moyenne globale
    np.testing.assert_allclose(c2.knn_regress(X, y, X[:5], 40), np.full(5, y.mean()))
