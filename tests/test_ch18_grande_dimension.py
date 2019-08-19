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


def test_lasso_cd_matches_sklearn():
    from sklearn.linear_model import ElasticNet, Lasso
    rng = np.random.RandomState(5)
    X, y = rng.randn(20, 50), rng.randn(20)
    lam = 2.0
    b = g.lasso_cd(X, y, lam, tol=1e-12, n_iter=5000)
    sk = Lasso(alpha=lam / 20, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y)
    np.testing.assert_allclose(b, sk.coef_, atol=1e-6)
    assert np.count_nonzero(b) <= 20
    b2 = g.lasso_cd(X, y, lam, alpha=0.5, tol=1e-12, n_iter=5000)
    sk2 = ElasticNet(alpha=lam / 20, l1_ratio=0.5, fit_intercept=False, tol=1e-12,
                     max_iter=100000).fit(X, y)
    np.testing.assert_allclose(b2, sk2.coef_, atol=1e-6)


def test_fused_lasso_signal_optimality():
    rng = np.random.RandomState(6)
    n = 20
    y = np.r_[np.zeros(7), 2 * np.ones(6), np.zeros(7)] + 0.3 * rng.randn(n)
    lam1, lam2 = 0.2, 1.0
    b = g.fused_lasso_signal(y, lam1, lam2, n_iter=20000, tol=1e-12)
    obj = lambda v: 0.5 * ((y - v) ** 2).sum() + lam1 * np.abs(v).sum() \
        + lam2 * np.abs(np.diff(v)).sum()
    # aucune petite perturbation aléatoire ne diminue l'objectif
    f0 = obj(b)
    for _ in range(200):
        assert obj(b + 1e-3 * rng.randn(n)) >= f0 - 1e-9
    # comparaison à une reformulation lisse sous contraintes linéaires
    D = np.diff(np.eye(n), axis=0)
    def f(z):
        v, s, t = z[:n], z[n:2 * n], z[2 * n:]
        return 0.5 * ((y - v) ** 2).sum() + lam1 * s.sum() + lam2 * t.sum()
    A = np.block([[np.eye(n), np.eye(n), np.zeros((n, n - 1))],
                  [-np.eye(n), np.eye(n), np.zeros((n, n - 1))],
                  [D, np.zeros((n - 1, n)), np.eye(n - 1)],
                  [-D, np.zeros((n - 1, n)), np.eye(n - 1)]])
    z0 = np.r_[y, np.abs(y) + 1, np.abs(D @ y) + 1]
    res = optimize.minimize(f, z0, method="SLSQP",
                            constraints=[{"type": "ineq", "fun": lambda z: A @ z,
                                          "jac": lambda z: A}],
                            options={"maxiter": 1000, "ftol": 1e-12})
    assert f0 <= res.fun + 1e-6


def test_kernel_tools():
    rng = np.random.RandomState(7)
    X, y = _classif_data(rng, N=30, p=10)
    X0 = rng.randn(5, 10)
    K, K0, k00 = X @ X.T, X0 @ X.T, (X0 ** 2).sum(1)
    classes, D2 = g.kernel_centroid_distances(K, K0, k00, y)
    direct = np.stack([((X0 - X[y == c].mean(0)) ** 2).sum(1) for c in classes], axis=1)
    np.testing.assert_allclose(D2, direct, atol=1e-10)
    np.testing.assert_allclose(g.kernel_distances(K, K0, k00),
                               ((X0[:, None, :] - X[None]) ** 2).sum(2), atol=1e-10)
    # double centrage de -Delta^2/2 = produits scalaires centrés
    Dist2 = ((X[:, None] - X[None]) ** 2).sum(2)
    Xc = X - X.mean(0)
    np.testing.assert_allclose(g.double_center(-Dist2 / 2), Xc @ Xc.T, atol=1e-10)
    # ACP noyau = ACP classique (au signe près)
    Z, U, d = g.kernel_pca(K, 3)
    Uc, dc, Vt = np.linalg.svd(Xc, full_matrices=False)
    np.testing.assert_allclose(np.abs(Z), np.abs(Uc[:, :3] * dc[:3]), atol=1e-8)


def test_univariate_scores_and_spca():
    rng = np.random.RandomState(8)
    X, y = rng.randn(30, 20), rng.randn(30)
    sc = g.univariate_scores(X, y)
    for j in range(3):
        r = stats.linregress(X[:, j], y)
        assert abs(sc[j] - r.slope / r.stderr) < 1e-8
    # avec toutes les variables, SPCA = régression sur la 1re composante principale
    m = g.SupervisedPCA(n_features=20).fit(X, y)
    Xc = X - X.mean(0)
    U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
    z = Xc @ Vt[0]
    pred = y.mean() + z * (z @ (y - y.mean())) / (z @ z)
    np.testing.assert_allclose(m.predict(X), pred, atol=1e-10)
    m2 = g.SupervisedPCA(threshold=1.0).fit(X, y)
    assert set(m2.selected_) == set(np.flatnonzero(np.abs(sc) > 1.0))


def test_t_statistics_match_scipy():
    rng = np.random.RandomState(9)
    X = rng.randn(25, 40)
    grp = np.zeros(25, bool)
    grp[:8] = True
    t = g.t_statistics(X, grp)
    ref = stats.ttest_ind(X[grp], X[~grp], axis=0).statistic
    np.testing.assert_allclose(t, ref, atol=1e-10)
    tp = g.permutation_t(X, grp, 30, np.random.RandomState(0), chunk=7)
    assert tp.shape == (30, 40)


def test_benjamini_hochberg_and_bonferroni():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej, L, thr = g.benjamini_hochberg(p, 0.05)
    # p_(j) < 0.05 j / 10 : vrai pour j = 1, 2 seulement
    assert L == 2 and thr == 0.008 and rej.sum() == 2
    assert g.bonferroni(p, 0.05).sum() == 1
    # cohérence avec les p-valeurs ajustées
    adj = g.bh_adjusted(p)
    np.testing.assert_array_equal(adj <= 0.05, rej)
    np.testing.assert_allclose(adj[:2], [0.01, 0.04])
    rej0, L0, _ = g.benjamini_hochberg(np.ones(5), 0.1)
    assert L0 == 0 and not rej0.any()
