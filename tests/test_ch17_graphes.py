"""Tests du module esl.ch17_graphes (chapitre 17)."""
import warnings

import numpy as np
import pytest

from esl import ch17_graphes as g


def _cov(seed=0, N=60, p=7):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, p) @ rng.randn(p, p)
    X /= X.std(axis=0)
    return g.empirical_covariance(X)


# --- Graphes ---------------------------------------------------------------

def test_cliques_figure_17_2():
    # (a) chemin X - Y - Z ; (b) triangle X,Y,W + Z isolé ; (c) cycle de 4
    assert g.maximal_cliques(g.adjacency_from_edges(3, [(0, 1), (1, 2)])) == [(0, 1), (1, 2)]
    A = g.adjacency_from_edges(4, [(0, 1), (0, 3), (1, 3)])
    assert g.maximal_cliques(A) == [(0, 1, 3), (2,)]
    A = g.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert g.maximal_cliques(A) == [(0, 1), (0, 3), (1, 2), (2, 3)]


def test_cliques_figure_17_8():
    A = g.adjacency_from_edges(6, [(0, 1), (0, 3), (0, 5), (1, 5), (2, 3), (4, 5)])
    assert g.maximal_cliques(A) == [(0, 1, 5), (0, 3), (2, 3), (4, 5)]


def test_separation():
    # figure 17.2 (d) : X - Y - Z - W
    A = g.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert g.separates(A, [0], [2, 3], [1])
    assert g.separates(A, [1], [3], [2])
    assert not g.separates(A, [0], [3], [])
    # cycle (c) : {X, Z} sépare Y et W
    C = g.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert g.separates(C, [1], [3], [0, 2])
    assert not g.separates(C, [1], [3], [0])


def test_components():
    A = g.adjacency_from_edges(5, [(0, 1), (3, 4)])
    assert g.connected_components(A) == [[0, 1], [2], [3, 4]]


# --- Gaussien --------------------------------------------------------------

def test_algorithme_17_1_exemple_whittaker():
    S = np.array([[10, 1, 5, 4], [1, 10, 2, 6], [5, 2, 10, 3], [4, 6, 3, 10.]])
    A = g.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    W, Theta = g.fit_known_graph(S, A)
    assert abs(W[0, 2] - 1.31) < 5e-3 and abs(W[1, 3] - 0.87) < 5e-3
    # seules les entrées des arêtes absentes diffèrent de S
    mask = A | np.eye(4, dtype=bool)
    assert np.allclose(W[mask], S[mask])
    assert abs(Theta[0, 2]) < 1e-10 and abs(Theta[1, 3]) < 1e-10
    assert np.allclose(W @ Theta, np.eye(4), atol=1e-8)


def test_graphe_complet_donne_S():
    S = _cov()
    A = ~np.eye(S.shape[0], dtype=bool)
    W, Theta = g.fit_known_graph(S, A)
    assert np.allclose(W, S, atol=1e-8)
    assert np.allclose(Theta, np.linalg.inv(S), atol=1e-6)


def test_lasso_cd_vs_sklearn():
    from sklearn.linear_model import Lasso
    rng = np.random.RandomState(1)
    N, q = 80, 6
    Z = rng.randn(N, q)
    y = Z @ np.array([2, -1, 0, 0, 0.5, 0]) + rng.randn(N)
    Z -= Z.mean(0)
    y -= y.mean()
    lam = 0.2
    b = g.lasso_cd(Z.T @ Z / N, Z.T @ y / N, lam)
    ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Z, y).coef_
    assert np.allclose(b, ref, atol=1e-6)
