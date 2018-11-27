"""Tests du module esl.ch15_forets (chapitre 15 : forêts aléatoires)."""

import itertools

import numpy as np
import pytest

from esl import ch15_forets as rf


def test_variance_moyenne_correlee_simulation():
    rng = np.random.RandomState(0)
    B, rho, s2 = 5, 0.3, 2.0
    C = s2 * (rho * np.ones((B, B)) + (1 - rho) * np.eye(B))
    Z = rng.multivariate_normal(np.zeros(B), C, size=200000)
    assert Z.mean(axis=1).var() == pytest.approx(
        rf.variance_moyenne_correlee(s2, rho, B), rel=0.02)
    # formule exacte : 1^T C 1 / B^2
    assert np.ones(B) @ C @ np.ones(B) / B ** 2 == pytest.approx(
        rf.variance_moyenne_correlee(s2, rho, B))
