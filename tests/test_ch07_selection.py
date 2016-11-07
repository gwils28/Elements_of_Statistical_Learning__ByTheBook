"""Tests du module esl.ch07_selection."""

import itertools

import numpy as np
import pytest

from esl import ch07_selection as sel


def _data(n=30, p=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X @ np.arange(1, p + 1) + rng.randn(n)
    return X, y


def test_hat_matrix_projector_and_trace():
    X, _ = _data()
    H = sel.hat_matrix(X)
    assert np.allclose(H, H.T)
    assert np.allclose(H @ H, H)
    assert np.isclose(sel.df_trace(H), X.shape[1] + 1)


def test_ols_fit_predict_matches_lstsq():
    X, y = _data()
    pred = sel.ols_fit_predict(X, y, X)
    assert np.allclose(pred, sel.hat_matrix(X) @ y)


def test_loo_shortcut_ols():
    X, y = _data()
    S = sel.hat_matrix(X)
    brute = np.array([y[i] - sel.ols_fit_predict(np.delete(X, i, 0), np.delete(y, i),
                                                 X[i:i + 1])[0] for i in range(len(y))])
    assert np.allclose(sel.loo_residuals(y, S), brute)
    # exercice 7.3 (b)
    assert np.all(np.abs(brute) >= np.abs(y - S @ y) - 1e-12)


def test_smoothing_spline_properties():
    rng = np.random.RandomState(1)
    x = np.sort(rng.rand(25))
    S = sel.smoothing_spline_matrix(x, 1e-3)
    assert np.allclose(S, S.T)
    # reproduit les fonctions affines
    assert np.allclose(S @ np.ones(25), 1)
    assert np.allclose(S @ x, x)
    ev = np.linalg.eigvalsh(S)
    assert ev.min() > 0 and ev.max() < 1 + 1e-10
    assert 2 < sel.df_trace(S) < 25
    # lam -> 0 : interpolation ; lam grand : droite des moindres carrés
    xg = np.linspace(0, 1, 25)
    assert np.allclose(sel.smoothing_spline_matrix(xg, 1e-12), np.eye(25), atol=1e-6)
    assert np.allclose(sel.smoothing_spline_matrix(x, 1e9), sel.hat_matrix(x), atol=1e-4)


def test_smoothing_spline_predict_and_loo():
    rng = np.random.RandomState(2)
    x = np.sort(rng.rand(20))
    y = np.sin(6 * x) + 0.3 * rng.randn(20)
    lam = 1e-3
    S = sel.smoothing_spline_matrix(x, lam)
    assert np.allclose(sel.smoothing_spline_predict(x, y, lam, x), S @ y)
    brute = np.array([y[i] - sel.smoothing_spline_predict(np.delete(x, i), np.delete(y, i),
                                                          lam, x[i])[0] for i in range(20)])
    assert np.allclose(sel.loo_residuals(y, S), brute, atol=1e-8)


def test_knn_smoother():
    X, y = _data(n=40)
    k = 5
    S = sel.knn_smoother_matrix(X, k)
    assert np.allclose(S.sum(1), 1)
    assert np.isclose(sel.df_trace(S), 40 / k)
    assert np.allclose(S @ y, sel.knn_regression(X, y, X, k))


def test_gcv_and_criteria():
    X, y = _data()
    S = sel.hat_matrix(X)
    n, d = len(y), 4
    err = np.mean((y - S @ y) ** 2)
    assert np.isclose(sel.gcv_score(y, S), err / (1 - d / n) ** 2)
    assert np.isclose(sel.cp_statistic(err, d, 1.0, n), err + 2 * d / n)
    assert np.isclose(sel.bic_gaussian(err, d, 2.0, n), n / 2 * err + np.log(n) * d)
    assert np.isclose(sel.aic_loglik(-10.0, 3, 20), 1.0 + 0.3)


def test_df_monte_carlo_ols():
    X, _ = _data(n=20)
    H = sel.hat_matrix(X)
    df = sel.df_monte_carlo(lambda y: H @ y, np.zeros(20), 1.0, n_rep=4000,
                            rng=np.random.RandomState(0))
    assert abs(df - 4) < 0.3


def test_kfold_partition():
    folds = sel.kfold_indices(23, 5, np.random.RandomState(0))
    allidx = np.sort(np.concatenate(folds))
    assert np.array_equal(allidx, np.arange(23))
    assert {len(f) for f in folds} <= {4, 5}


def test_cross_val_loo_equals_shortcut():
    X, y = _data()
    n = len(y)
    folds = [np.array([i]) for i in range(n)]
    cv, _ = sel.cross_val_error(sel.ols_fit_predict, X, y, folds=folds)
    assert np.isclose(cv, np.mean(sel.loo_residuals(y, sel.hat_matrix(X)) ** 2))


def test_no_information_rate_binary():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 50)
    yhat = rng.randint(0, 2, 50)
    p1, q1 = y.mean(), yhat.mean()
    assert np.isclose(sel.no_information_rate(y, yhat), p1 * (1 - q1) + (1 - p1) * q1)


def test_bootstrap_1nn_independent_labels():
    rng = np.random.RandomState(0)
    n = 100
    X = rng.randn(n, 2)
    y = np.repeat([0, 1], n // 2)

    def nn1(Xtr, ytr, Xte):
        d2 = ((Xte[:, None, :] - Xtr[None]) ** 2).sum(-1)
        return ytr[d2.argmin(1)]

    r = sel.bootstrap_error_estimates(nn1, X, y, B=50, rng=rng, loss=sel.zero_one_loss)
    assert r["err"] == 0
    assert abs(r["boot"] - 0.184) < 0.05
    assert abs(r["loo1"] - 0.5) < 0.08
    assert np.isclose(r["632"], 0.632 * r["loo1"])
    assert r["632plus"] >= r["632"]


def test_best_subset_and_forward():
    X, y = _data(n=40, p=4)
    bs = sel.best_subset(X, y)
    assert len(bs) == 5
    rss = [r for _, r in bs]
    assert all(a >= b - 1e-9 for a, b in zip(rss, rss[1:]))
    for k in range(5):
        brute = min(sel.rss(X, y, c) for c in itertools.combinations(range(4), k))
        assert np.isclose(bs[k][1], brute)
    order = sel.forward_stepwise(X, y)
    assert sorted(order) == [0, 1, 2, 3]
    assert order[0] == bs[1][0][0]
