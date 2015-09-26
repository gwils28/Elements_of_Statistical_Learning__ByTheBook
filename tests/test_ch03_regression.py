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


def test_best_subset_bruteforce(data):
    X, y = data
    subsets, rss = cr.best_subset(X, y)
    assert np.all(np.diff(rss) <= 1e-12)
    for k in (1, 3):
        best = min(combinations(range(X.shape[1]), k),
                   key=lambda S: cr.ols_fit(X[:, list(S)], y)["rss"])
        assert set(best) == set(subsets[k])
    assert np.isclose(rss[-1], cr.ols_fit(X, y)["rss"])


def test_forward_stepwise_greedy(data):
    X, y = data
    order, rss = cr.forward_stepwise(X, y)
    chosen = []
    for k, j in enumerate(order):
        cand = [i for i in range(X.shape[1]) if i not in chosen]
        r = [cr.ols_fit(X[:, chosen + [i]], y)["rss"] for i in cand]
        assert cand[int(np.argmin(r))] == j
        chosen.append(j)
        assert np.isclose(min(r), rss[k + 1])


def test_backward_stepwise(data):
    X, y = data
    removed, rss = cr.backward_stepwise(X, y)
    full = cr.ols_fit(X, y)
    j = removed[0]
    keep = [k for k in range(X.shape[1]) if k != j]
    assert np.isclose(rss[1] - rss[0], full["z"][j + 1] ** 2 * full["sigma2"])
    assert np.isclose(rss[1], cr.ols_fit(X[:, keep], y)["rss"])


def test_forward_stagewise_converges_to_ols(data):
    X, y = data
    # convergence lente (variables corrélées) mais certaine
    path = cr.forward_stagewise(X, y, n_steps=30000)
    assert np.allclose(path[-1], cr.ols_fit(X, y)["coef"][1:], atol=1e-6)
    # FS_eps : les pas ont tous la même taille eps en norme L1
    path = cr.forward_stagewise(X, y, eps=0.01, n_steps=200)
    assert np.allclose(np.abs(np.diff(path, axis=0)).sum(1), 0.01)


def test_ridge_matches_sklearn_and_df(data):
    X, y = data
    for lam in (0.1, 5.0, 100.0):
        b0, b = cr.ridge_fit(X, y, lam)
        sk = Ridge(alpha=lam).fit(X, y)
        assert np.allclose(b, sk.coef_) and np.isclose(b0, sk.intercept_)
    Xc = X - X.mean(0)
    d = np.linalg.svd(Xc, compute_uv=False)
    lam = 7.0
    H = Xc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T)
    assert np.isclose(cr.ridge_df(d, lam), np.trace(H))
    assert np.isclose(cr.ridge_df(d, cr.ridge_lambda_for_df(X, 3.5)), 3.5)


def test_lasso_matches_sklearn(data):
    X, y = data
    N = len(y)
    for lam in (1.0, 20.0, 80.0):
        b0, b = cr.lasso_cd(X, y, lam)
        sk = Lasso(alpha=lam / N, tol=1e-12, max_iter=100000).fit(X, y)
        assert np.allclose(b, sk.coef_, atol=1e-6)
        assert np.isclose(b0, sk.intercept_, atol=1e-6)
    _, b = cr.lasso_cd(X, y, cr.lasso_lambda_max(X, y) * 1.0001)
    assert np.all(b == 0)


def test_soft_threshold():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert np.allclose(cr.soft_threshold(z, 1.0), [-2, 0, 0, 0, 2])


@pytest.mark.parametrize("method", ["lar", "lasso"])
def test_lars_matches_sklearn(prostate, method):
    Xtr, ytr, _, _, _ = prostate
    coefs, corr, _ = cr.lars(Xtr, ytr, method=method)
    Xc, yc = Xtr - Xtr.mean(0), ytr - ytr.mean()
    _, _, sk = lars_path(Xc, yc, method=method)
    assert coefs.shape == sk.T.shape
    assert np.allclose(coefs, sk.T, atol=1e-8)
    assert np.allclose(coefs[-1], cr.ols_fit(Xtr, ytr)["coef"][1:])
