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


def test_souche_trouve_le_seuil():
    rng = np.random.RandomState(0)
    X = rng.rand(300, 3)
    y = np.where(X[:, 1] > 0.37, 2.0, -1.0)
    t = RegressionTree(2).fit(X, y)
    assert t.feature[0] == 1
    assert abs(t.threshold[0] - 0.37) < 0.02
    assert np.allclose(t.predict(X), y)


def test_arbre_egal_sklearn():
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.RandomState(1)
    X = rng.randn(200, 4)
    y = np.sin(2 * X[:, 0]) + X[:, 2] ** 2 + 0.1 * rng.randn(200)
    w = rng.rand(200)
    t = RegressionTree(8).fit(X, y, w)
    sk = DecisionTreeRegressor(max_leaf_nodes=8, random_state=0).fit(X, y, sample_weight=w)
    assert np.allclose(t.predict(X), sk.predict(X))


def test_gain_egal_baisse_somme_carres():
    rng = np.random.RandomState(2)
    X = rng.randn(150, 3)
    y = X[:, 0] + rng.randn(150)
    t = RegressionTree(5).fit(X, y)
    sse0 = np.sum((y - y.mean()) ** 2)
    sse = np.sum((y - t.predict(X)) ** 2)
    assert np.isclose(t.improvement.sum(), sse0 - sse)


def test_adaboost_egal_sklearn(spheres):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    X, y, Xt, yt = spheres
    ab = AdaBoostM1(30).fit(X, y)
    sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=30).fit(X, y)
    # SAMME avec K = 2 classes coïncide avec AdaBoost.M1
    assert np.allclose(ab.alphas_, sk.estimator_weights_)
    assert np.all(ab.predict(Xt) == sk.predict(Xt))
    assert np.all(ab.errs_ < 0.5)


def test_adaboost_poids_mal_classes_moitie(spheres):
    """Après la mise à jour, les mal classés portent exactement la moitié du poids."""
    X, y, _, _ = spheres
    ab = AdaBoostM1(3).fit(X, y)
    w = ab.weights_history_[0]
    miss = ab.trees_[0].predict(X) != y
    assert np.isclose(w[miss].sum(), 0.5)


@pytest.mark.parametrize("loss", ["ls", "lad", "huber"])
def test_gb_regression_perte_decroit(loss):
    rng = np.random.RandomState(3)
    X = rng.rand(300, 2)
    y = np.sin(6 * X[:, 0]) + 0.2 * rng.randn(300)
    gb = GradientBoosting(loss, 60, 4, 0.5).fit(X, y)
    if loss == "ls":
        assert np.all(np.diff(gb.train_loss_) <= 1e-10)
    assert gb.train_loss_[-1] < gb.loss_value(y, np.full(300, gb.f0_))


def test_gb_ls_egal_sklearn():
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.RandomState(4)
    X = rng.randn(200, 3)
    y = X[:, 0] * X[:, 1] + rng.randn(200) * 0.3
    gb = GradientBoosting("ls", 20, 2, 0.3).fit(X, y)
    sk = GradientBoostingRegressor(n_estimators=20, max_depth=1, learning_rate=0.3).fit(X, y)
    assert np.allclose(gb.predict(X), sk.predict(X), atol=1e-8)


def test_gb_deviance_init_et_proba(spheres):
    X, y, Xt, yt = spheres
    y01 = (y + 1) // 2
    gb = GradientBoosting("deviance", 50, 4, 0.2).fit(X, y01)
    assert np.isclose(gb.f0_, np.log(y01.mean() / (1 - y01.mean())))
    p = gb.predict_proba(Xt)
    assert np.all((p > 0) & (p < 1))
    assert np.mean(gb.predict(Xt) != (yt + 1) // 2) < 0.4


def test_dependance_partielle_arbres_egale_moyenne(spheres):
    X, y, _, _ = spheres
    gb = GradientBoosting("ls", 15, 6, 0.5).fit(X, y.astype(float))
    for S in ([0], [2, 5]):
        XS = np.random.RandomState(5).randn(7, len(S))
        exact = gb.partial_dependence(XS, S, X=X)
        slow = partial_dependence_bruteforce(gb.decision_function, X, S, XS)
        assert np.allclose(exact, slow)
        # le parcours pondéré de Friedman n'est qu'approché (X_S, X_C indépendants ici)
        fast = gb.partial_dependence(XS, S)
        assert np.max(np.abs(fast - slow)) < 0.1
