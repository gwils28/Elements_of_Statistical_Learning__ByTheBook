"""Tests du module esl.ch12_svm (chapitre 12)."""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR

from esl.ch12_svm import (FDA, LDA, MDA, SVMClassifier, SVMRegressor, binomial_deviance,
                          eps_insensitive, hinge_loss, huber_loss, huberized_hinge, kmeans,
                          natural_spline_basis, poly2_features, polynomial_kernel, qp_interior_point,
                          rbf_kernel, second_diff_penalty, simulate_waveform, smo_solve,
                          waveform_bayes_posterior)


@pytest.fixture(scope="module")
def two_class():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(40, 2) + [1, 1], rng.randn(40, 2) - [1, 1]])
    y = np.r_[np.ones(40), -np.ones(40)]
    return X, y


@pytest.mark.parametrize("kernel,C,kw", [
    ("linear", 0.1, {}), ("linear", 10.0, {}), ("rbf", 1.0, {"gamma": 0.5}),
    ("rbf", 100.0, {"gamma": 2.0}), ("poly", 1.0, {"degree": 2, "gamma": 1.0}),
])
def test_svc_matches_sklearn(two_class, kernel, C, kw):
    X, y = two_class
    mine = SVMClassifier(C=C, kernel=kernel, **kw).fit(X, y)
    sk = SVC(C=C, kernel=kernel, coef0=1.0, tol=1e-8, **kw).fit(X, y)
    Z = np.random.RandomState(1).randn(50, 2) * 2
    np.testing.assert_allclose(mine.decision_function(Z), sk.decision_function(Z), atol=2e-4)
    assert set(mine.support_) == set(sk.support_)


@pytest.mark.parametrize("solver", ["smo", "ip"])
def test_kkt_conditions(two_class, solver):
    X, y = two_class
    C = 1.0
    s = SVMClassifier(C=C, kernel="rbf", gamma=1.0, solver=solver,
                      tol=1e-7 if solver == "smo" else None).fit(X, y)
    a, m = s.alpha_, y * s.decision_function(X)
    assert np.all(a >= -1e-12) and np.all(a <= C + 1e-12)
    assert abs(a @ y) < 1e-8
    assert np.all(m[a < 1e-6] >= 1 - 1e-4)              # alpha = 0  => y f >= 1
    assert np.all(m[a > C - 1e-6] <= 1 + 1e-4)          # alpha = C  => y f <= 1
    free = (a > 1e-5) & (a < C - 1e-5)
    np.testing.assert_allclose(m[free], 1, atol=1e-4)    # 0 < alpha < C => y f = 1


def test_smo_and_interior_point_agree(two_class):
    X, y = two_class
    Q = (y[:, None] * y[None, :]) * rbf_kernel(X, X, 0.7)
    p = -np.ones(len(y))
    a1, b1, _ = smo_solve(Q, p, y, 2.0, tol=1e-8)
    a2, b2, _ = qp_interior_point(Q, p, y, 2.0)
    obj = lambda a: 0.5 * a @ Q @ a + p @ a
    assert abs(obj(a1) - obj(a2)) < 1e-7
    assert abs(b1 - b2) < 1e-4


def test_primal_equals_dual(two_class):
    X, y = two_class
    s = SVMClassifier(C=3.0, kernel="poly", degree=3).fit(X, y)
    K = s.K_(X, X)
    w = s.alpha_ * y
    dual = s.alpha_.sum() - 0.5 * w @ K @ w
    assert abs(s.primal_objective() - dual) < 1e-6 * (1 + abs(dual))


def test_svr_matches_sklearn():
    rng = np.random.RandomState(0)
    x = np.sort(rng.rand(50) * 6)[:, None]
    t = np.sin(x[:, 0]) + 0.2 * rng.randn(50)
    a = SVMRegressor(C=5.0, epsilon=0.15, kernel="rbf", gamma=1.0).fit(x, t)
    b = SVR(C=5.0, epsilon=0.15, gamma=1.0, tol=1e-8).fit(x, t)
    np.testing.assert_allclose(a.predict(x), b.predict(x), atol=1e-4)
    # résidus des non-supports dans le tube
    inside = np.setdiff1d(np.arange(50), a.support_)
    assert np.all(np.abs(t[inside] - a.predict(x[inside])) <= 0.15 + 1e-6)


def test_kernels():
    rng = np.random.RandomState(0)
    A, B = rng.randn(4, 3), rng.randn(5, 3)
    np.testing.assert_allclose(polynomial_kernel(A, B, 2), poly2_features(A) @ poly2_features(B).T)
    K = rbf_kernel(A, A, 0.3)
    np.testing.assert_allclose(np.diag(K), 1)
    Z = rng.randn(30, 2)
    assert np.linalg.eigvalsh(rbf_kernel(Z, Z, 1.0)).min() > -1e-10


def test_losses():
    m = np.array([-2.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(hinge_loss(m), [3, 1, 0, 0])
    np.testing.assert_allclose(binomial_deviance(m), np.log1p(np.exp(-m)))
    np.testing.assert_allclose(huberized_hinge(m), [8, 1, 0, 0])
    np.testing.assert_allclose(eps_insensitive([0.5, -2.0], 1.0), [0, 1])
    np.testing.assert_allclose(huber_loss([0.5, 3.0], 1.0), [0.125, 2.5])


def test_fda_linear_is_lda():
    rng = np.random.RandomState(0)
    X, g = simulate_waveform(300, rng)
    Xt, _ = simulate_waveform(200, rng)
    f = FDA().fit(X, g)
    l = LDA().fit(X, g)
    N, K = len(g), 3
    Wi = np.linalg.inv(l.W_ * (N - K) / N)
    maha = np.column_stack([np.einsum("ij,jk,ik->i", Xt - m, Wi, Xt - m) for m in l.means_])
    d = f.distances(Xt) + 2 * np.log(f.priors_) - maha
    assert d.std(axis=1).max() < 1e-8
    # LDA écrite à la main = sklearn (qui normalise W par N au lieu de N - K)
    l.W_ = l.W_ * (N - K) / N
    l.Winv_ = np.linalg.inv(l.W_)
    sk = LinearDiscriminantAnalysis().fit(X, g)
    diff = sk.decision_function(Xt) - l.decision_function(Xt)
    assert diff.std(axis=1).max() < 1e-8
