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


def test_n_weights_formula():
    # éq. (11.8) et réseau Net-2 du tableau 11.1 : 256 entrées, 12 cachées, 10 sorties
    assert n_weights(256, 12, 10) == 3214
    assert n_weights(256, 0, 10) == 2570


def test_sigmoid_softmax():
    v = np.linspace(-800, 800, 11)
    s = sigmoid(v)
    assert np.all(np.isfinite(s))
    assert np.allclose(s + sigmoid(-v), 1.0)
    T = np.random.RandomState(0).randn(6, 4) * 50
    P = softmax(T)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(softmax(T + 7.0), P)           # invariance par translation


def test_small_weights_nearly_linear():
    """Ex. 11.2 : poids d'entrée proches de 0 -> modèle presque linéaire."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 2)
    params = init_params(2, 5, 1, rng)
    errs = []
    for eps in (1e-1, 1e-2):
        alpha = params["alpha"] * eps
        alpha[:, 0] = 0.0                              # constantes nulles
        pr = {"alpha": alpha, "beta": params["beta"]}
        f = forward(pr, X)["F"][:, 0]
        A = np.column_stack([np.ones(len(X)), X])
        coef = np.linalg.lstsq(A, f, rcond=None)[0]
        errs.append(np.max(np.abs(A @ coef - f)))
    # sigma''(0) = 0 : l'écart à la linéarité décroît comme eps^3 (facteur 1000 attendu)
    assert errs[1] < errs[0] * 3e-3
