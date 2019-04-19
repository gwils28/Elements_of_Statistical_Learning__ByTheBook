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
