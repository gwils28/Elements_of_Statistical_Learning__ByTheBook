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


def test_arbre_complet_interpole():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    y = rng.randn(60)
    a = rf.ArbreAleatoire(m=None, nmin=1).fit(X, y)
    assert np.allclose(a.predict(X), y)
    assert a.n_feuilles == 60


def test_arbre_nmin_et_comparaison_sklearn():
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.RandomState(0)
    X = rng.rand(200, 2)
    y = np.sin(6 * X[:, 0]) + 0.1 * rng.randn(200)
    a = rf.ArbreAleatoire(m=None, nmin=5).fit(X, y)
    feuilles = a.apply(X)
    assert np.bincount(feuilles)[np.unique(feuilles)].min() >= 5
    sk = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(X, y)
    # même critère (SCE) : erreur d'apprentissage comparable
    e1 = np.mean((a.predict(X) - y) ** 2)
    e2 = np.mean((sk.predict(X) - y) ** 2)
    assert e1 == pytest.approx(e2, rel=0.15)


def test_premiere_coupure_identique_a_sklearn():
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.RandomState(3)
    X = rng.randn(100, 4)
    y = 2 * (X[:, 2] > 0.3) + 0.2 * rng.randn(100)
    a = rf.ArbreAleatoire(m=None).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=1).fit(X, y)
    assert a.variable[0] == sk.tree_.feature[0]
    assert a.seuil[0] == pytest.approx(sk.tree_.threshold[0])


def test_foret_classification_et_oob():
    rng = np.random.RandomState(0)
    X, y = rf.simuler_spheres(300, p=4, rng=rng)
    f = rf.ForetAleatoire(B=30, m=2, classification=True).fit(X, y)
    Xt, yt = rf.simuler_spheres(500, p=4, rng=rng)
    err = np.mean(f.predict(Xt) != yt)
    assert err < 0.3
    P = f.predictions_arbres(X)
    oob = rf.erreur_oob_cumulee(P, f.dans_sac, y)
    assert oob.shape == (30,)
    assert abs(oob[-1] - err) < 0.1
    # environ 36,8 % d'observations hors sac par arbre
    assert (~f.dans_sac).mean() == pytest.approx(np.exp(-1), abs=0.02)
    assert f.importance_gini.max() == pytest.approx(100)


def test_erreur_test_cumulee_derniere_valeur():
    rng = np.random.RandomState(0)
    P = (rng.rand(11, 50) > 0.4).astype(float)
    y = rng.randint(0, 2, 50)
    e = rf.erreur_test_cumulee(P, y)
    assert e[-1] == pytest.approx(np.mean((P.mean(0) > 0.5) != y))
