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
