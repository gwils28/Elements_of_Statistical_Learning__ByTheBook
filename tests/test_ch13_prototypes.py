"""Tests du module esl.ch13_prototypes (chapitre 13)."""

import numpy as np
import pytest

from esl import ch13_prototypes as cp


def _blobs(rng, n=60):
    X = np.vstack([rng.randn(n, 2) * 0.3 + [-2, 0], rng.randn(n, 2) * 0.3 + [2, 0]])
    y = np.repeat([0, 1], n)
    return X, y
