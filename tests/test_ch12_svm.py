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
