"""Tests du module esl.ch03_regression_complements (exercices 3.16 à 3.30)."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge, lars_path

from esl import ch03_regression_complements as rc


def make_data(N=60, p=6, seed=0, rho=0.5):
    rng = np.random.RandomState(seed)
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.randn(N, p) @ np.linalg.cholesky(cov).T
    beta = rng.randn(p)
    y = X @ beta + rng.randn(N)
    X, _, _, _ = rc.standardize(X, unit_norm=True)
    return X, y - y.mean()


def test_standardize():
    X, _ = make_data()
    Xs, _, _, _ = rc.standardize(X)
    assert np.allclose(Xs.mean(0), 0) and np.allclose(Xs.std(0), 1)
    Xu, _, _, _ = rc.standardize(X, unit_norm=True)
    assert np.allclose(np.linalg.norm(Xu, axis=0), 1)


def test_orthonormal_estimators():
    rng = np.random.RandomState(1)
    X = np.linalg.qr(rng.randn(40, 5))[0]
    y = rng.randn(40)
    b_ls = X.T @ y
    assert np.allclose(rc.ridge(X, y, 2.0), b_ls / 3)
    lam = 0.3
    assert np.allclose(rc.lasso_cd(X, y, lam), rc.soft_threshold(b_ls, lam), atol=1e-8)
    hb = rc.hard_threshold(b_ls, 2)
    assert np.count_nonzero(hb) == 2
    assert np.min(np.abs(hb[hb != 0])) >= np.max(np.abs(b_ls[hb == 0]))


def test_ridge_vs_sklearn_and_svd():
    X, y = make_data()
    b = rc.ridge(X, y, 0.7)
    sk = Ridge(alpha=0.7, fit_intercept=False).fit(X, y).coef_
    assert np.allclose(b, sk)
    assert np.allclose(rc.ridge_svd_path(X, y, [0.7])[:, 0], b)


def test_ridge_df_inverse():
    X, _ = make_data()
    d = np.linalg.svd(X, compute_uv=False)
    for df in [0.5, 2.0, 5.5]:
        lam = rc.ridge_lambda_for_df(d, df)
        assert np.isclose(rc.ridge_df(d, lam), df)
    assert rc.ridge_lambda_for_df(d, X.shape[1]) == 0.0


def test_lasso_cd_vs_sklearn():
    X, y = make_data()
    N = len(y)
    lam = 0.5
    b = rc.lasso_cd(X, y, lam)
    sk = Lasso(alpha=lam / N, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y).coef_
    assert np.allclose(b, sk, atol=1e-6)


@pytest.mark.parametrize("method", ["lar", "lasso"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lars_vs_sklearn(method, seed):
    X, y = make_data(N=50, p=8, seed=seed, rho=0.5)
    res = rc.lars(X, y, method=method)
    _, _, coefs = lars_path(X, y, method=method)
    assert res["coefs"].shape == coefs.shape
    assert np.allclose(res["coefs"], coefs, atol=1e-8)


def test_lar_ends_at_least_squares_and_equal_correlations():
    X, y = make_data(N=50, p=6, seed=4)
    res = rc.lars(X, y)
    b_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(res["coefs"][:, -1], b_ls)
    # corrélations absolues liées sur l'ensemble actif à chaque nœud
    for k, A in enumerate(res["actives"][:-1]):
        c = np.abs(X.T @ (y - X @ res["coefs"][:, k + 1]))
        assert np.allclose(c[A], c[A][0])
        assert np.all(c <= c[A][0] + 1e-10)


def test_lasso_path_matches_cd():
    X, y = make_data(N=50, p=8, seed=3, rho=0.8)
    res = rc.lars(X, y, method="lasso")
    assert any(e[0] == "sortie" for e in res["events"])  # le chemin contient une sortie
    for k in range(1, res["coefs"].shape[1] - 1):
        lam = res["lambdas"][k]
        assert np.allclose(rc.lasso_cd(X, y, lam), res["coefs"][:, k], atol=1e-7)


def test_interpolate_path():
    X, y = make_data()
    res = rc.lars(X, y, method="lasso")
    b = rc.interpolate_path(res["coefs"], [0.0, 1.0])
    assert np.allclose(b[:, 0], 0) and np.allclose(b[:, 1], res["coefs"][:, -1])
    half = rc.interpolate_path(res["coefs"], [0.5])[:, 0]
    assert np.isclose(np.abs(half).sum(), 0.5 * np.abs(res["coefs"][:, -1]).sum())


def test_pcr_full_is_least_squares():
    X, y = make_data()
    b = rc.pcr(X, y)
    assert np.allclose(b[:, -1], np.linalg.lstsq(X, y, rcond=None)[0])


def test_pls_vs_sklearn_and_ls():
    X, y = make_data(N=60, p=5, seed=2)
    Xs, _, _, _ = rc.standardize(X)
    out = rc.pls(Xs, y)
    assert np.allclose(out["betas"][:, -1], np.linalg.lstsq(Xs, y, rcond=None)[0])
    for m in (1, 2, 3):
        sk = PLSRegression(n_components=m, scale=False).fit(Xs, y)
        assert np.allclose(Xs @ out["betas"][:, m] + y.mean(), sk.predict(Xs).ravel(), atol=1e-8)
    Z = out["Z"]
    G = Z.T @ Z
    assert np.allclose(G - np.diag(np.diag(G)), 0, atol=1e-8)


def test_pls_equals_conjugate_gradient():
    X, y = make_data(N=60, p=6, seed=7)
    Xs, _, _, _ = rc.standardize(X)
    yc = y - y.mean()
    b_pls = rc.pls(Xs, yc)["betas"]
    b_cg = rc.conjugate_gradient(Xs.T @ Xs, Xs.T @ yc)
    assert np.allclose(b_pls, b_cg, atol=1e-8)


def test_pls_norm_increases():
    X, y = make_data(N=40, p=8, seed=8, rho=0.9)
    b = rc.pls(X, y)["betas"]
    norms = np.linalg.norm(b, axis=0)
    assert np.all(np.diff(norms) >= -1e-10)
