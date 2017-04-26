"""Tests du module esl.ch09_arbres (modèles additifs, CART, PRIM, MARS)."""

import numpy as np
import pytest

from esl.ch09_arbres import (
    GAM, MARS, ArbreCART, MatriceLisseur, SplineLisseur, auc, backfitting,
    impurity, in_box, local_linear_matrix, logistic_irls, prim, prim_peel,
    roc_curve,
)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


# --- lisseurs ---------------------------------------------------------------

def test_spline_reproduit_les_droites(rng):
    x = np.sort(rng.uniform(0, 1, 60))
    sm = SplineLisseur(x, df=5)
    S = sm.matrix()
    lin = 2.0 - 3.0 * x
    assert np.allclose(S @ lin, lin, atol=1e-6)
    assert np.allclose(S, S.T, atol=1e-8)
    assert abs(np.trace(S) - 5) < 1e-3


def test_spline_egale_spline_de_lissage_scipy(rng):
    from scipy.interpolate import make_smoothing_spline
    x = np.sort(rng.uniform(0, 1, 40))
    y = np.sin(6 * x) + 0.3 * rng.randn(40)
    lam = 1e-4
    ours = SplineLisseur(x, lam=lam).fit(y)
    ref = make_smoothing_spline(x, y, lam=lam)(x)
    assert np.allclose(ours, ref, atol=1e-5)


def test_spline_valeurs_propres_ridge(rng):
    x = rng.uniform(0, 1, 30)
    S = SplineLisseur(x, lam=1e-3, ridge=1e-2).matrix()
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    assert ev.max() < 1 and ev.min() > -1e-8


def test_lineaire_local_reproduit_les_droites(rng):
    x = rng.uniform(0, 1, 50)
    S = local_linear_matrix(x, 0.1)
    assert np.allclose(S @ (1 + 4 * x), 1 + 4 * x)
    assert np.allclose(S.sum(1), 1)
