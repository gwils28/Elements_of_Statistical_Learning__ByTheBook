"""Tests du module esl.ch03_regression (chapitre 3)."""

from itertools import combinations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LinearRegression, Ridge, lars_path

from esl import ch03_regression as cr


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    N, p = 60, 6
    X = rng.randn(N, p) @ (np.eye(p) + 0.4 * rng.randn(p, p))
    beta = np.array([2.0, -1.0, 0.0, 0.5, 0.0, 1.5])
    y = 1.0 + X @ beta + rng.randn(N)
    return X, y


@pytest.fixture
def prostate():
    return cr.load_prostate_standardized()
