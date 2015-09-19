"""Tests du module esl.ch03_regression (chapitre 3)."""

from itertools import combinations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LinearRegression, Ridge, lars_path

from esl import ch03_regression as cr


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    N, p = 60, 6
    X = rng.randn(N, p) @ (np.eye(p) + 0.4 * rng.randn(p, p))
    beta = np.array([2.0, -1.0, 0.0, 0.5, 0.0, 1.5])
    y = 1.0 + X @ beta + rng.randn(N)
    return X, y


@pytest.fixture
def prostate():
    return cr.load_prostate_standardized()


def test_gram_schmidt_qr(data):
    X, _ = data
    A = cr.add_intercept(X)
    Q, R = cr.gram_schmidt_qr(A)
    assert np.allclose(Q.T @ Q, np.eye(A.shape[1]))
    assert np.allclose(np.triu(R), R)
    assert np.allclose(Q @ R, A)
    Q2, R2 = np.linalg.qr(A)
    S = np.sign(np.diag(R2))
    assert np.allclose(R, S[:, None] * R2)


def test_back_substitution():
    rng = np.random.RandomState(1)
    R = np.triu(rng.randn(5, 5)) + 5 * np.eye(5)
    b = rng.randn(5)
    assert np.allclose(R @ cr.back_substitution(R, b), b)


def test_ols_matches_sklearn(data):
    X, y = data
    fit = cr.ols_fit(X, y)
    lr = LinearRegression().fit(X, y)
    assert np.isclose(fit["coef"][0], lr.intercept_)
    assert np.allclose(fit["coef"][1:], lr.coef_)
    assert np.allclose(cr.ols_qr(cr.add_intercept(X), y), fit["coef"])
