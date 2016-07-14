import numpy as np
import pytest
from scipy import integrate, stats

from esl import ch06_noyaux as nk


def _data(n=80, seed=0):
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(4 * x) + rng.normal(0, 1 / 3, n)
    return x, y
