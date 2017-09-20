"""Tests du module esl.ch11_reseaux (réseaux de neurones et PPR)."""

import numpy as np
import pytest
from scipy.interpolate import make_smoothing_spline

from esl.ch11_reseaux import (MLP, PPR, SmoothingSpline, backprop, df_to_lambda,
                              forward, gradient_check, init_params, n_weights,
                              one_hot, pack, sigmoid, softmax, unpack)


def _data(N=15, p=3, K=2, seed=0, classif=False):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, p)
    if classif:
        Y = one_hot(rng.randint(0, K, N), K)
    else:
        Y = rng.randn(N, K)
    return X, Y


@pytest.mark.parametrize("output,loss,classif", [
    ("identity", "squared", False),
    ("softmax", "cross_entropy", True),
    ("softmax", "squared", True),
    ("sigmoid", "squared", True),
    ("sigmoid", "cross_entropy", True),
])
@pytest.mark.parametrize("M", [0, 1, 4])
@pytest.mark.parametrize("lam", [0.0, 0.3])
def test_gradient_check(output, loss, classif, M, lam):
    X, Y = _data(K=3, classif=classif)
    params = init_params(X.shape[1], M, Y.shape[1], np.random.RandomState(1))
    assert gradient_check(params, X, Y, output, loss, lam) < 1e-7


def test_pack_unpack_roundtrip():
    rng = np.random.RandomState(0)
    for M in (0, 3):
        params = init_params(5, M, 2, rng)
        theta = pack(params)
        assert theta.size == n_weights(5, M, 2)
        back = unpack(theta, 5, M, 2)
        assert np.allclose(pack(back), theta)
