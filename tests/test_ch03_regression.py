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


def test_table_3_2(prostate):
    Xtr, ytr, Xte, yte, _ = prostate
    fit = cr.ols_fit(Xtr, ytr)
    book_coef = [2.46, 0.68, 0.26, -0.14, 0.21, 0.31, -0.29, -0.02, 0.27]
    book_z = [27.60, 5.37, 2.75, -1.40, 2.06, 2.47, -1.87, -0.15, 1.74]
    assert np.allclose(fit["coef"], book_coef, atol=0.006)
    assert np.allclose(fit["z"], book_z, atol=0.006)
    pred = fit["coef"][0] + Xte @ fit["coef"][1:]
    assert np.isclose(np.mean((yte - pred) ** 2), 0.521, atol=5e-4)


def test_f_test_equals_z2_and_book(prostate):
    Xtr, ytr, _, _, _ = prostate
    full = cr.ols_fit(Xtr, ytr)
    N = len(ytr)
    # retrait d'une seule variable : F = z^2 (exercice 3.1)
    for j in range(8):
        keep = [k for k in range(8) if k != j]
        small = cr.ols_fit(Xtr[:, keep], ytr)
        F, _ = cr.f_test(small["rss"], full["rss"], 7, 8, N)
        assert np.isclose(F, full["z"][j + 1] ** 2)
    # éq. (3.16) : F = 1.67, p = 0.17
    small = cr.ols_fit(Xtr[:, [0, 1, 3, 4]], ytr)
    F, pv = cr.f_test(small["rss"], full["rss"], 4, 8, N)
    assert abs(F - 1.67) < 0.01 and abs(pv - 0.17) < 0.01


def test_successive_orthogonalization(data):
    X, y = data
    A = cr.add_intercept(X)
    Z, G = cr.successive_orthogonalization(A)
    assert np.allclose(Z @ G, A)
    D = Z.T @ Z
    assert np.allclose(D - np.diag(np.diag(D)), 0, atol=1e-8)
    zp = Z[:, -1]
    beta = cr.ols_fit(X, y)["coef"]
    assert np.isclose(zp @ y / (zp @ zp), beta[-1])


def test_hat_matrix(data):
    X, _ = data
    H = cr.hat_matrix(cr.add_intercept(X))
    assert np.allclose(H, H.T)
    assert np.allclose(H @ H, H)
    assert np.isclose(np.trace(H), X.shape[1] + 1)
