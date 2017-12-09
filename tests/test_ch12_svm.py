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
