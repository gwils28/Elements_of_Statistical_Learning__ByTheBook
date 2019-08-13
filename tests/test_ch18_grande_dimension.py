import numpy as np
import pytest
from scipy import optimize, stats

from esl import ch18_grande_dimension as g


def _classif_data(rng, N=40, p=60, K=3):
    y = np.repeat(np.arange(K), N // K)
    X = rng.randn(len(y), p)
    X[:, :5] += y[:, None]
    return X, y


def test_soft_threshold():
    x = np.array([-3.0, -0.5, 0.0, 0.2, 2.0])
    np.testing.assert_allclose(g.soft_threshold(x, 1.0), [-2, 0, 0, 0, 1])


def test_nsc_delta_zero_is_diagonal_lda():
    rng = np.random.RandomState(0)
    X, y = _classif_data(rng)
    m = g.NearestShrunkenCentroids(delta=0.0).fit(X, y)
    np.testing.assert_allclose(m.shrunken_centroids_, m.centroids_)
    # LDA diagonale écrite à la main
    s = m.s_ + m.s0_
    sc = np.stack([-(((X - m.centroids_[k]) / s) ** 2).sum(1) + 2 * np.log(np.mean(y == k))
                   for k in range(3)], axis=1)
    np.testing.assert_allclose(m.discriminant(X), sc)
    np.testing.assert_allclose(m.predict_proba(X).sum(1), 1.0)


def test_nsc_large_delta_removes_all_features():
    rng = np.random.RandomState(1)
    X, y = _classif_data(rng)
    m = g.NearestShrunkenCentroids(delta=1e6).fit(X, y)
    assert len(m.active_features_) == 0
    np.testing.assert_allclose(m.shrunken_centroids_, np.tile(X.mean(0), (3, 1)))
    # le nombre de gènes actifs décroît avec delta
    counts = [len(m.set_delta(d).active_features_) for d in np.linspace(0, 6, 13)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_nsc_lasso_variant_solves_criterion():
    """Exercice 18.2 : avec s0 = 0 et m_k = 1/sqrt(N_k), mu_jk minimise le critère."""
    rng = np.random.RandomState(2)
    X, y = _classif_data(rng, N=30, p=8)
    lam = 1.3
    m = g.NearestShrunkenCentroids(delta=lam, s0=0.0, mk="lasso").fit(X, y)
    xbar = X.mean(0)
    for j in range(3):
        for k in range(3):
            xk = X[y == k, j]
            sj = m.s_[j]
            f = lambda mu: 0.5 * ((xk - xbar[j] - mu) ** 2).sum() / sj ** 2 \
                + lam * np.sqrt(len(xk)) * abs(mu) / sj
            res = optimize.minimize_scalar(f, bounds=(-5, 5), method="bounded",
                                           options={"xatol": 1e-10})
            assert abs(res.x - (m.shrunken_centroids_[k, j] - xbar[j])) < 1e-5


def test_ridge_svd_matches_direct():
    rng = np.random.RandomState(3)
    X, y = rng.randn(15, 80), rng.randn(15)
    for lam in [0.1, 1.0, 50.0]:
        np.testing.assert_allclose(g.ridge_svd(X, y, lam), g.ridge_direct(X, y, lam), atol=1e-10)
    path = g.ridge_svd(X, y, [0.1, 1.0])
    assert path.shape == (2, 80)
    R, V = g.svd_reduce(X)
    np.testing.assert_allclose(R @ V.T, X, atol=1e-12)
    np.testing.assert_allclose(V.T @ V, np.eye(15), atol=1e-12)


def test_ridge_df_limits():
    rng = np.random.RandomState(4)
    X = rng.randn(10, 30)
    assert abs(g.ridge_df(X, 1e-10) - 10) < 1e-6
    assert g.ridge_df(X, 1e10) < 1e-6
    df = g.ridge_df(X, [0.1, 1, 10])
    assert np.all(np.diff(df) < 0)
