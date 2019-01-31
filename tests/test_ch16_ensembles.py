import numpy as np
import pytest

from esl.ch16_ensembles import (
    adaboost_shrunk, block_correlated_data, extract_rules, forward_stagewise,
    isle_ensemble, l1_arc_length, lasso_cd, lasso_path_cd, lasso_path_fast, logistic_irls,
    ridge_path_df, rule_matrix, standardize, tree_predictions)


def _data(n=80, p=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    beta = np.r_[3, -2, 1.5, np.zeros(p - 3)]
    y = X @ beta + rng.randn(n)
    return X, y


def test_standardize():
    X, _ = _data()
    Xs, m, s = standardize(X)
    assert np.allclose(Xs.mean(0), 0) and np.allclose(Xs.std(0), 1)
    Xs2, _, _ = standardize(X, m, s)
    assert np.allclose(Xs, Xs2)


def test_lasso_cd_matches_sklearn():
    from sklearn.linear_model import Lasso
    X, y = _data()
    for lam in (0.01, 0.1, 0.5):
        b0, b = lasso_cd(X, y, lam, tol=1e-10)
        sk = Lasso(alpha=lam, tol=1e-12, max_iter=100000).fit(X, y)
        assert np.allclose(b, sk.coef_, atol=1e-6)
        assert np.isclose(b0, sk.intercept_, atol=1e-6)


def test_lasso_kkt_and_path():
    X, y = _data()
    lams, b0s, coefs = lasso_path_cd(X, y, n_lambdas=20, tol=1e-10)
    assert np.allclose(coefs[0], 0)            # lam_max annule tout
    Xc, yc = X - X.mean(0), y - y.mean()
    for lam, b in zip(lams, coefs):
        g = Xc.T @ (yc - Xc @ b) / len(y)      # conditions KKT
        act = b != 0
        assert np.allclose(g[act], lam * np.sign(b[act]), atol=1e-6)
        assert np.all(np.abs(g[~act]) <= lam + 1e-6)
