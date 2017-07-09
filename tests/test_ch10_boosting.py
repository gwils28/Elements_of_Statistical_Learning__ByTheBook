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
