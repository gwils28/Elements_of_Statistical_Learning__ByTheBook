"""Tests du module esl.ch13_prototypes (chapitre 13)."""

import numpy as np
import pytest

from esl import ch13_prototypes as cp


def _blobs(rng, n=60):
    X = np.vstack([rng.randn(n, 2) * 0.3 + [-2, 0], rng.randn(n, 2) * 0.3 + [2, 0]])
    y = np.repeat([0, 1], n)
    return X, y


def test_sq_dists():
    rng = np.random.RandomState(0)
    A, B = rng.randn(5, 3), rng.randn(4, 3)
    ref = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    assert np.allclose(cp.sq_dists(A, B), ref)
