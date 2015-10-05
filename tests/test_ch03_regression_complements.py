"""Tests du module esl.ch03_regression_complements (exercices 3.16 à 3.30)."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge, lars_path

from esl import ch03_regression_complements as rc


def make_data(N=60, p=6, seed=0, rho=0.5):
    rng = np.random.RandomState(seed)
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.randn(N, p) @ np.linalg.cholesky(cov).T
    beta = rng.randn(p)
    y = X @ beta + rng.randn(N)
    X, _, _, _ = rc.standardize(X, unit_norm=True)
    return X, y - y.mean()


def test_standardize():
    X, _ = make_data()
    Xs, _, _, _ = rc.standardize(X)
    assert np.allclose(Xs.mean(0), 0) and np.allclose(Xs.std(0), 1)
    Xu, _, _, _ = rc.standardize(X, unit_norm=True)
    assert np.allclose(np.linalg.norm(Xu, axis=0), 1)
