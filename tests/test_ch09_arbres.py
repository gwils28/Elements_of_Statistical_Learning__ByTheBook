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
