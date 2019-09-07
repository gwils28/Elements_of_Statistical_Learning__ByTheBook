import numpy as np
import pytest
from scipy import stats

from esl import ch18_grande_dimension_complements as m


def test_sigmoid_stable():
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    s = m.sigmoid(z)
    assert np.all(np.isfinite(s))
    assert np.allclose(s, 1 / (1 + np.exp(-np.clip(z, -50, 50))))


def test_separable_loglik_tends_to_zero():
    x = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    vals = [m.logistic_loglik(0.0, np.array([b]), x, y) for b in (1, 10, 100)]
    assert vals[0] < vals[1] < vals[2] < 0
    assert vals[2] > -1e-10
    it, path, ll = m.logistic_gradient_ascent(x, y, n_iter=2000)
    assert np.all(np.diff(path[:, 1]) > 0)        # beta croît sans cesse
    assert np.all(np.diff(ll) > 0)


def test_logistic_ridge_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = rng.randn(80, 5)
    y = (X[:, 0] + rng.randn(80) > 0).astype(int)
    lam = 0.7
    b0, b = m.logistic_ridge(X, y, lam)
    sk = LogisticRegression(C=1 / (2 * lam), tol=1e-12, max_iter=10000).fit(X, y)
    assert np.allclose(b, sk.coef_[0], atol=1e-5)
    assert np.isclose(b0, sk.intercept_[0], atol=1e-5)


def test_ridge_via_R_and_cv_identical():
    rng = np.random.RandomState(0)
    N, p = 30, 200
    X = rng.randn(N, p)
    y = X[:, :3].sum(axis=1) + rng.randn(N)
    R, V = m.svd_reduce(X)
    assert np.allclose(R @ V.T, X)
    b0, b = m.ridge_fit(X, y, 2.0)
    c0, th = m.ridge_fit(R, y, 2.0)
    assert np.isclose(b0, c0) and np.allclose(b, V @ th)
    # comparaison à la formule primale
    Xc, yc = X - X.mean(0), y - y.mean()
    assert np.allclose(b, np.linalg.solve(Xc.T @ Xc + 2.0 * np.eye(p), Xc.T @ yc))
    folds = m.kfold_indices(N, 5, rng)
    lams = [0.1, 1.0, 10.0]
    assert np.allclose(m.ridge_cv_errors(X, y, lams, folds),
                       m.ridge_cv_errors(R, y, lams, folds))


def test_kernel_logistic_equals_primal():
    rng = np.random.RandomState(1)
    N, p = 25, 60
    X = rng.randn(N, p)
    y = (X[:, 0] > 0).astype(int)
    lam = 0.5
    b0, b = m.logistic_ridge(X, y, lam)
    t0, alpha = m.kernel_logistic_ridge(X @ X.T, y, lam)
    X0 = rng.randn(7, p)
    assert np.isclose(b0, t0, atol=1e-6)
    assert np.allclose(X0 @ b + b0, (X0 @ X.T) @ alpha + t0, atol=1e-6)


def test_dw_knn_score():
    Xtr = np.array([[0.0], [1.0], [3.0], [4.0]])
    ytr = np.array([1, 1, -1, -1])
    s = m.dw_knn_score(Xtr, ytr, np.array([[0.5], [2.0], [3.5]]))
    assert s[0] > 0 and np.isclose(s[1], 0) and s[2] < 0
    assert np.isclose(s[0], np.log(2.5 / 0.5))
    s2 = m.dw_knn_score(Xtr, ytr, np.array([[2.0]]), priors=(0.8, 0.2))
    assert np.isclose(s2[0], np.log(0.8 / 0.2))
    s3 = m.dw_knn_score(Xtr, ytr, np.array([[0.5]]), k=2, dim=3)
    assert np.isclose(s3[0], 3 * np.log(3.5 / 0.5))


def test_roc_auc_matches_mann_whitney():
    rng = np.random.RandomState(0)
    s = np.r_[rng.randn(50) + 1, rng.randn(60)]
    lab = np.r_[np.ones(50), np.zeros(60)]
    _, _, auc = m.roc_curve(s, lab)
    U = stats.mannwhitneyu(s[:50], s[50:]).statistic
    assert np.isclose(auc, U / (50 * 60))


def test_kernel_pca_matches_linear_pca():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 50) + 3.0
    X0 = rng.randn(4, 50) + 3.0
    U, d, Z = m.kernel_pca(X @ X.T, n_components=5)
    xm = X.mean(0)
    _, s, Vt = np.linalg.svd(X - xm, full_matrices=False)
    Zlin = (X - xm) @ Vt[:5].T
    sign = np.sign(np.sum(Z * Zlin, axis=0))
    assert np.allclose(Z * sign, Zlin, atol=1e-8)
    z0 = m.kernel_pca_project(X @ X.T, U, d, X @ X0.T)
    assert np.allclose(z0.T * sign, (X0 - xm) @ Vt[:5].T, atol=1e-8)
    # un point d'apprentissage est projeté sur sa propre ligne de Z
    z1 = m.kernel_pca_project(X @ X.T, U, d, X @ X[2])
    assert np.allclose(z1, Z[2])


def test_two_sample_t_matches_scipy():
    rng = np.random.RandomState(0)
    X = rng.randn(30, 20)
    g = np.r_[np.zeros(12, bool), np.ones(8, bool)]
    t = m.two_sample_t(X, g)
    ref = stats.ttest_ind(X[:, g], X[:, ~g], axis=1).statistic
    assert np.allclose(t, ref)
    tp = m.permutation_t(X, g, 5, rng)
    assert tp.shape == (30, 5)


def test_bh_equals_plugin_and_bonferroni():
    rng = np.random.RandomState(0)
    M, N = 400, 20
    X = rng.randn(M, N)
    g = np.r_[np.zeros(10, bool), np.ones(10, bool)]
    X[:40, g] += 2.0
    t = m.two_sample_t(X, g)
    tp = m.permutation_t(X, g, 50, rng)
    p = m.pooled_pvalues(t, tp)
    for alpha in (0.05, 0.1, 0.2, 0.4):
        L, rej = m.benjamini_hochberg(p, alpha)
        assert rej.sum() == L
        assert np.array_equal(rej, m.plugin_reject(t, tp, alpha))
    # FDR plug-in au seuil |t|_(l) = M p_(l) / l
    a = np.sort(np.abs(t))[::-1]
    fdr, R, _ = m.plugin_fdr(t, tp, a[:10])
    assert np.array_equal(R, np.arange(1, 11))
    assert np.allclose(fdr, M * np.sort(p)[:10] / np.arange(1, 11))
    assert m.bonferroni_reject(p, 0.05).sum() <= m.benjamini_hochberg(p, 0.05)[0]
    assert 0.5 < m.pi0_quartile_estimate(t, tp) <= 1.0
