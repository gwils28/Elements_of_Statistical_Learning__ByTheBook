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


def test_rho_minimal_annule_variance():
    for B in (2, 5, 10):
        assert rf.variance_moyenne_correlee(1.0, rf.rho_minimal(B), B) == pytest.approx(0)


def test_correlation_bootstrap():
    rng = np.random.RandomState(1)
    n = 5
    x = rng.randn(40000, n)
    i1 = rng.randint(0, n, (40000, n))
    i2 = rng.randint(0, n, (40000, n))
    m1 = np.take_along_axis(x, i1, 1).mean(1)
    m2 = np.take_along_axis(x, i2, 1).mean(1)
    assert np.corrcoef(m1, m2)[0, 1] == pytest.approx(
        rf.correlation_moyennes_bootstrap(n), abs=0.02)
    assert m1.var() == pytest.approx(rf.variance_moyenne_bootstrap(1.0, n), rel=0.03)


def test_proba_variable_pertinente_valeurs_du_livre():
    # en-têtes de la figure 15.7 : (2, 5) -> 0.52 ; (2, 25) -> 0.34
    for bruit, attendu in [(5, 0.52), (25, 0.34), (50, 0.25), (100, 0.19), (150, 0.15)]:
        p = 2 + bruit
        m = int(np.floor(np.sqrt(p)))
        assert rf.proba_variable_pertinente(p, 2, m) == pytest.approx(attendu, abs=0.01)
