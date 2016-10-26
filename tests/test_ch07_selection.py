"""Tests du module esl.ch07_selection."""

import itertools

import numpy as np
import pytest

from esl import ch07_selection as sel


def _data(n=30, p=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X @ np.arange(1, p + 1) + rng.randn(n)
    return X, y


def test_hat_matrix_projector_and_trace():
    X, _ = _data()
    H = sel.hat_matrix(X)
    assert np.allclose(H, H.T)
    assert np.allclose(H @ H, H)
    assert np.isclose(sel.df_trace(H), X.shape[1] + 1)


def test_ols_fit_predict_matches_lstsq():
    X, y = _data()
    pred = sel.ols_fit_predict(X, y, X)
    assert np.allclose(pred, sel.hat_matrix(X) @ y)


def test_loo_shortcut_ols():
    X, y = _data()
    S = sel.hat_matrix(X)
    brute = np.array([y[i] - sel.ols_fit_predict(np.delete(X, i, 0), np.delete(y, i),
                                                 X[i:i + 1])[0] for i in range(len(y))])
    assert np.allclose(sel.loo_residuals(y, S), brute)
    # exercice 7.3 (b)
    assert np.all(np.abs(brute) >= np.abs(y - S @ y) - 1e-12)
