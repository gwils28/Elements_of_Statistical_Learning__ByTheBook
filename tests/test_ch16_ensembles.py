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


def test_stagewise_equals_lasso_orthogonal_design():
    # base orthonormée : forward stagewise (eps -> 0) = lasso (soft-thresholding)
    rng = np.random.RandomState(1)
    Q, _ = np.linalg.qr(rng.randn(50, 5))
    y = Q @ np.array([3.0, -2.0, 1.0, 0.5, 0.0]) + 0.1 * rng.randn(50)
    eps, M = 1e-3, 3000
    path = forward_stagewise(Q, y, eps=eps, n_steps=M)
    a = path[-1]
    t = np.abs(a).sum()
    z = Q.T @ y
    # solution lasso de borne t : seuillage doux de z au niveau gamma
    from scipy.optimize import brentq
    gam = brentq(lambda g: np.maximum(np.abs(z) - g, 0).sum() - t, 0, np.abs(z).max())
    lasso = np.sign(z) * np.maximum(np.abs(z) - gam, 0)
    assert np.allclose(a, lasso, atol=5 * eps)
    assert np.isclose(np.abs(path[1]).sum(), eps)


def test_stagewise_reaches_least_squares():
    X, y = _data(p=4)
    Xs, _, _ = standardize(X)
    yc = y - y.mean()
    path = forward_stagewise(Xs, yc, eps=1e-3, n_steps=20000)
    ls = np.linalg.lstsq(Xs, yc, rcond=None)[0]
    assert np.allclose(path[-1], ls, atol=1e-2)


def test_ridge_df():
    X, y = _data()
    Xc, yc = X - X.mean(0), y - y.mean()
    lams, coefs = ridge_path_df(Xc, yc, [2.0, 5.0])
    for lam, b in zip(lams, coefs):
        direct = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
        assert np.allclose(b, direct)
    d2 = np.linalg.svd(Xc, compute_uv=False) ** 2
    assert np.isclose(np.sum(d2 / (d2 + lams[0])), 2.0)


def test_block_correlated_data():
    rng = np.random.RandomState(0)
    X, y, beta, sigma = block_correlated_data(20000, n_blocks=3, block_size=4, rng=rng)
    C = np.corrcoef(X.T)
    assert np.allclose(C[0, 1:4], 0.95, atol=0.01)
    assert np.allclose(C[0, 4:], 0.0, atol=0.03)
    assert np.sum(beta != 0) == 3
    assert np.isclose(sigma ** 2, 0.72 * np.sum(beta ** 2))
    resid = y - X @ beta
    assert np.isclose(resid.var() / (X @ beta).var(), 0.72, rtol=0.05)


def test_arc_length():
    t = np.linspace(0, 1, 50)[:, None]
    mono = np.hstack([t, -2 * t ** 2])
    assert np.isclose(l1_arc_length(mono)[-1], np.abs(mono[-1]).sum())
    wave = np.hstack([np.sin(4 * t), t])
    assert l1_arc_length(wave)[-1] > np.abs(wave[-1]).sum() + 0.5


def test_leaf_rules_reproduce_tree():
    from sklearn.tree import DecisionTreeRegressor
    X, y = _data(n=200, p=3)
    tree = DecisionTreeRegressor(max_leaf_nodes=6).fit(X, y)
    rules, leaf = extract_rules(tree)
    R = rule_matrix(rules, X)
    assert np.allclose(R[:, leaf].sum(1), 1)          # les feuilles partitionnent
    coef = np.linalg.lstsq(R[:, leaf], y, rcond=None)[0]
    assert np.allclose(R[:, leaf] @ coef, tree.predict(X))
    assert len(rules) == 2 * leaf.sum() - 2


def test_logistic_on_leaves_gives_proportions():
    rng = np.random.RandomState(0)
    g = rng.randint(3, size=300)
    y = (rng.rand(300) < np.array([0.2, 0.5, 0.7])[g]).astype(float)
    Z = np.eye(3)[g]
    beta = logistic_irls(Z, y)
    p = 1 / (1 + np.exp(-beta))
    assert np.allclose(p, [y[g == k].mean() for k in range(3)])


def test_isle_and_adaboost_shapes():
    X, y = _data(n=100, p=4)
    f0, trees = isle_ensemble(X, y, n_trees=5, eta=0.5, nu=0.1,
                              rng=np.random.RandomState(0))
    assert len(trees) == 5 and tree_predictions(trees, X).shape == (100, 5)
    ys = np.sign(y - np.median(y))
    out = adaboost_shrunk(X, ys, n_trees=20, nu=0.5)
    assert np.all(np.abs(out["margin"]) <= 1 + 1e-12)
    assert np.all(out["alpha"] > 0)


def test_lasso_path_fast_matches_cd():
    X, y = _data()
    lams, b0, c = lasso_path_cd(X, y, n_lambdas=15, tol=1e-10)
    lams2, b02, c2 = lasso_path_fast(X, y, lambdas=lams)
    assert np.allclose(c, c2, atol=1e-4) and np.allclose(b0, b02, atol=1e-4)
