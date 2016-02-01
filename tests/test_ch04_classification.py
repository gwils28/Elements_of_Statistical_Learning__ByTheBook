"""Tests du module esl.ch04_classification (chapitre 4)."""

import numpy as np
import pytest
from scipy import linalg
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression

from esl import datasets
from esl import ch04_classification as c4


@pytest.fixture(scope="module")
def gauss3():
    rng = np.random.RandomState(0)
    means = np.array([[0, 0, 0], [2, 1, 0], [0, 3, 1]], float)
    X = np.vstack([rng.randn(60, 3) + m for m in means])
    y = np.repeat([0, 1, 2], 60)
    return X, y


@pytest.fixture(scope="module")
def vowel():
    return datasets.load_vowel("train") + datasets.load_vowel("test")


@pytest.fixture(scope="module")
def separable():
    rng = np.random.RandomState(1)
    X = rng.randn(40, 2)
    y = np.where(X @ np.array([1.0, -1.0]) + 0.3 > 0, 1, -1)
    X = X + 0.4 * y[:, None] * np.array([1.0, -1.0]) / np.sqrt(2)   # marge nette
    return X, y


def test_indicator_rows_sum_to_one(gauss3):
    X, y = gauss3
    m = c4.indicator_regression_fit(X, y)
    _, F = c4.indicator_regression_predict(m, X, return_fitted=True)
    np.testing.assert_allclose(F.sum(axis=1), 1.0, atol=1e-10)


def test_lda_matches_sklearn(gauss3):
    X, y = gauss3
    m = c4.lda_fit(X, y)
    sk = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
    np.testing.assert_array_equal(c4.lda_predict(m, X), sk.predict(X))
    # différences de discriminants = log-rapports de densités a posteriori
    from scipy.stats import multivariate_normal as mvn
    D = c4.lda_decision_function(m, X[:5])
    logf = np.column_stack([mvn(m["means"][k], m["cov"]).logpdf(X[:5]) + np.log(m["priors"][k])
                            for k in range(3)])
    np.testing.assert_allclose(D - D[:, :1], logf - logf[:, :1], atol=1e-8)


def test_qda_matches_sklearn(gauss3):
    X, y = gauss3
    m = c4.qda_fit(X, y)
    sk = QuadraticDiscriminantAnalysis().fit(X, y)
    np.testing.assert_array_equal(c4.qda_predict(m, X), sk.predict(X))


def test_rda_limits(gauss3):
    X, y = gauss3
    m0 = c4.qda_fit(X, y, alpha=0.0)
    lda = c4.lda_fit(X, y)
    np.testing.assert_array_equal(c4.qda_predict(m0, X), c4.lda_predict(lda, X))


def test_vowel_error_rates_table_4_1(vowel):
    Xtr, ytr, Xte, yte = vowel
    lda = c4.lda_fit(Xtr, ytr)
    qda = c4.qda_fit(Xtr, ytr)
    reg = c4.indicator_regression_fit(Xtr, ytr)
    assert round(c4.error_rate(yte, c4.lda_predict(lda, Xte)), 2) == 0.56
    assert round(c4.error_rate(yte, c4.qda_predict(qda, Xte)), 2) == 0.53
    assert round(c4.error_rate(yte, c4.indicator_regression_predict(reg, Xte)), 2) == 0.67


def test_reduced_rank_full_equals_lda(vowel):
    Xtr, ytr, Xte, _ = vowel
    rr = c4.reduced_rank_lda_fit(Xtr, ytr)
    np.testing.assert_array_equal(c4.reduced_rank_lda_predict(rr, Xte, 10),
                                  c4.lda_predict(c4.lda_fit(Xtr, ytr), Xte))


def test_reduced_rank_normalisation(gauss3):
    X, y = gauss3
    rr = c4.reduced_rank_lda_fit(X, y)
    V = rr["scalings"]
    assert V.shape == (3, 2)
    np.testing.assert_allclose(V.T @ rr["cov"] @ V, np.eye(2), atol=1e-10)
    assert rr["eigvals"][0] >= rr["eigvals"][1] > 0


def test_logistic_irls_table_4_2():
    df = datasets.load_saheart()
    cols = ["sbp", "tobacco", "ldl", "famhist", "obesity", "alcohol", "age"]
    r = c4.logistic_irls(df[cols].values, df["chd"].values)
    expected = [-4.130, 0.006, 0.080, 0.185, 0.939, -0.035, 0.001, 0.043]
    np.testing.assert_allclose(np.round(r["coef"], 3), expected, atol=1e-3)
    np.testing.assert_allclose(r["se"][[0, 4]], [0.964, 0.225], atol=1e-3)
    assert np.all(np.diff(r["history"]) >= -1e-9)        # Newton monotone ici


def test_multilogit_matches_sklearn(gauss3):
    X, y = gauss3
    r = c4.multilogit_newton(X, y)
    sk = LogisticRegression(C=1e10, max_iter=5000, tol=1e-10).fit(X, y)
    np.testing.assert_allclose(c4.multilogit_predict_proba(r, X), sk.predict_proba(X), atol=1e-4)


def test_multilogit_two_classes_equals_binary(gauss3):
    X, y = gauss3
    mask = y < 2
    r = c4.multilogit_newton(X[mask], y[mask])
    b = c4.logistic_irls(X[mask], (y[mask] == 0).astype(float))  # classe 0 contre référence 1
    np.testing.assert_allclose(r["coef"][0], b["coef"], atol=1e-6)
