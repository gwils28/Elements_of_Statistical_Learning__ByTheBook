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
