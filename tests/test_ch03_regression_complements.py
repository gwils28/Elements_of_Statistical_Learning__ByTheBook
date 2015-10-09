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
