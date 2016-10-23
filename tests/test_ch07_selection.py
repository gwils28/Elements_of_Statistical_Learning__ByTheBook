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
