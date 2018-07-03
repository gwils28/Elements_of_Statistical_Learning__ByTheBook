"""Tests du module esl.ch14_non_supervise (chapitre 14)."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from esl import ch14_non_supervise as ns


# --- règles d'association ---------------------------------------------------

def _apriori_force_brute(Z, t, taille_max):
    out = {}
    for k in range(1, taille_max + 1):
        for S in combinations(range(Z.shape[1]), k):
            s = Z[:, list(S)].all(1).mean()
            if s >= t:
                out[S] = s
    return out


def test_apriori_egal_force_brute():
    rng = np.random.RandomState(0)
    Z = rng.uniform(size=(300, 8)) < np.linspace(0.3, 0.8, 8)
    Z[:, 1] |= Z[:, 0]
    a = ns.apriori(Z, 0.15, taille_max=4)
    b = _apriori_force_brute(Z, 0.15, 4)
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k])


def test_regles_confiance_lift():
    rng = np.random.RandomState(1)
    Z = rng.uniform(size=(500, 5)) < 0.6
    f = ns.apriori(Z, 0.2)
    R = ns.regles_association(f)
    for _, r in R.head(20).iterrows():
        A, B = list(r.antecedent), list(r.consequent)
        sAB = Z[:, A + B].all(1).mean()
        sA = Z[:, A].all(1).mean()
        sB = Z[:, B].all(1).mean()
        assert r.confiance == pytest.approx(sAB / sA)
        assert r.lift == pytest.approx(sAB / (sA * sB))


def test_codage_disjonctif():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 3]})
    Z, noms, var = ns.codage_disjonctif(df, ["a"], ["b"])
    assert Z.shape == (4, 5)
    assert np.all(Z[:, :2].sum(1) == 1) and np.all(Z[:, 2:].sum(1) == 1)


# --- non supervisé comme supervisé ------------------------------------------

def test_reference_permutation_marges():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 3))
    X0 = ns.echantillon_reference(X, "permutation", rng)
    assert np.allclose(np.sort(X, 0), np.sort(X0, 0))


def test_logistique_irls_sans_penalite():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    B = np.column_stack([np.ones(200), X])
    beta = ns.regression_logistique(B, y)
    sk = LogisticRegression(C=1e10, tol=1e-12, max_iter=10000).fit(X, y)
    assert np.allclose(beta, np.r_[sk.intercept_, sk.coef_[0]], atol=1e-4)


def test_splines_naturelles_lineaires_hors_noeuds():
    x = np.linspace(-5, 5, 400)
    N = ns.base_splines_naturelles(x, np.linspace(-1, 1, 5))
    gauche, droite = x < -1, x > 1
    for m in (gauche, droite):
        d2 = np.diff(N[m], 2, axis=0)
        assert np.abs(d2).max() < 1e-8


# --- classification ---------------------------------------------------------

def test_kmeans_point_fixe_et_sklearn():
    from sklearn.cluster import KMeans
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(m, 0.5, size=(40, 2)) for m in ((0, 0), (4, 0), (0, 4))])
    lab, C, W = ns.kmeans(X, 3, n_init=10, rng=rng)
    for k in range(3):
        assert np.allclose(C[k], X[lab == k].mean(0))
    assert np.all(ns.distances_carrees(X, C).argmin(1) == lab)
    sk = KMeans(3, n_init=10, random_state=0).fit(X)
    assert W == pytest.approx(sk.inertia_, rel=1e-8)
    assert ns.indice_rand_ajuste(lab, sk.labels_) == pytest.approx(1.0)


def test_kmedoides_optimal_petit_cas():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(12, 2))
    D = np.sqrt(ns.distances_carrees(X))
    lab, med, crit = ns.kmedoides(D, 2, n_init=20, rng=rng)
    best = min(D[:, list(m)].min(1).sum() for m in combinations(range(12), 2))
    assert crit == pytest.approx(best)


@pytest.mark.parametrize("methode", ["single", "complete", "average"])
def test_hierarchique_comme_scipy(methode):
    rng = np.random.RandomState(3)
    X = rng.normal(size=(25, 4))
    d = pdist(X)
    Z = ns.classification_hierarchique(squareform(d), methode)
    Zs = hierarchy.linkage(d, method=methode)
    assert np.allclose(Z[:, 2], Zs[:, 2])
    C = ns.dissimilarites_cophenetiques(Z)
    assert np.allclose(squareform(C, checks=False), hierarchy.cophenet(Zs))


def test_cophenetique_ultrametrique():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(15, 2))
    C = ns.dissimilarites_cophenetiques(ns.classification_hierarchique(
        np.sqrt(ns.distances_carrees(X)), "average"))
    for i, j, k in combinations(range(15), 3):
        assert C[i, j] <= max(C[i, k], C[j, k]) + 1e-12
