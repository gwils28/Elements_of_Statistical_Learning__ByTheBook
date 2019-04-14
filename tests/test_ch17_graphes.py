"""Tests du module esl.ch17_graphes (chapitre 17)."""
import warnings

import numpy as np
import pytest

from esl import ch17_graphes as g


def _cov(seed=0, N=60, p=7):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, p) @ rng.randn(p, p)
    X /= X.std(axis=0)
    return g.empirical_covariance(X)
