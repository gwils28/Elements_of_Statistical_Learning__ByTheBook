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
