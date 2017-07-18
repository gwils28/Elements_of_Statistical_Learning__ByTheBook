"""Tests du module esl.ch10_boosting (chapitre 10)."""

import numpy as np
import pytest

from esl.ch10_boosting import (
    AdaBoostM1, GradientBoosting, GradientBoostingMulticlass, RegressionTree,
    loss_binomial_deviance, loss_exponential, loss_hinge, loss_huber,
    loss_misclassification, loss_squared_margin, partial_dependence_bruteforce,
    simulate_nested_spheres)


@pytest.fixture
def spheres():
    rng = np.random.RandomState(0)
    X, y = simulate_nested_spheres(400, rng=rng)
    Xt, yt = simulate_nested_spheres(1000, rng=rng)
    return X, y, Xt, yt


def test_simulation_equilibree():
    X, y = simulate_nested_spheres(20000, rng=np.random.RandomState(1))
    assert X.shape == (20000, 10)
    assert abs(np.mean(y == 1) - 0.5) < 0.02


def test_pertes_passent_par_0_1():
    for L in (loss_exponential, loss_binomial_deviance, loss_squared_margin, loss_hinge):
        assert np.isclose(L(0.0), 1.0)
    assert loss_misclassification(-1) == 1 and loss_misclassification(2) == 0
    r = np.linspace(-3, 3, 13)
    assert np.allclose(loss_huber(r, 10.0), r ** 2)
    # raccord C^1 au point delta
    assert np.isclose(loss_huber(1.0, 1.0), 1.0)
