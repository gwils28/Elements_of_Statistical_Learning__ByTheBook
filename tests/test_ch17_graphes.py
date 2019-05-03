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


@pytest.mark.parametrize("lam", [0.05, 0.2, 0.5])
def test_graphical_lasso_vs_sklearn(lam):
    from sklearn.covariance import graphical_lasso as sk_gl
    S = _cov(seed=2)
    W, Theta = g.graphical_lasso(S, lam, penalize_diagonal=False, tol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_sk, prec_sk = sk_gl(S, lam, tol=1e-10, max_iter=2000)
    # sklearn s'arrête sur un critère de saut de dualité moins strict : on
    # compare avec une tolérance relative de 1 %
    assert np.allclose(Theta, prec_sk, atol=1e-2 * np.abs(prec_sk).max())
    assert np.allclose(W, cov_sk, atol=1e-2)

    # notre solution vérifie au moins aussi bien l'optimalité
    def obj(T):
        return -g.gaussian_loglik(T, S) + lam * (np.abs(T).sum() - np.abs(np.diag(T)).sum())
    assert obj(Theta) <= obj(prec_sk) + 1e-8


def test_graphical_lasso_kkt_et_diagonale():
    S = _cov(seed=3)
    p = S.shape[0]
    lam = 0.15
    W, Theta = g.graphical_lasso(S, lam, tol=1e-12)
    # diagonale de W = s_jj + lam (diagonale pénalisée)
    assert np.allclose(np.diag(W), np.diag(S) + lam)
    assert np.allclose(W @ Theta, np.eye(p), atol=1e-8)
    G = W - S  # = lam * Sign(Theta) (éq. 17.22)
    off = ~np.eye(p, dtype=bool)
    nz = (np.abs(Theta) > 1e-10) & off
    assert np.allclose(G[nz], lam * np.sign(Theta[nz]), atol=1e-7)
    assert np.all(np.abs(G[off & ~nz]) <= lam + 1e-7)


def test_graphical_lasso_lam_infini_retrouve_17_1():
    S = _cov(seed=4, p=5)
    A = g.adjacency_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    Lam = np.where(A, 0.0, np.inf)
    np.fill_diagonal(Lam, 0.0)
    W1, T1 = g.fit_known_graph(S, A)
    W2, T2 = g.graphical_lasso(S, Lam, tol=1e-13)
    assert np.allclose(T1, T2, atol=1e-6)


def test_chemin_creux_quand_lam_grand():
    S = _cov(seed=5)
    lam_max = np.abs(S[~np.eye(S.shape[0], dtype=bool)]).max()
    path = g.graphical_lasso_path(S, [lam_max * 1.01, 0.5 * lam_max, 0.01])
    assert g.support_graph(path[0]).sum() == 0
    assert g.support_graph(path[1]).sum() <= g.support_graph(path[2]).sum()


def test_correlations_partielles():
    # rho_{12|3} via la régression des résidus
    rng = np.random.RandomState(6)
    X = rng.randn(4000, 3) @ np.array([[1, .5, .2], [0, 1, .4], [0, 0, 1.]])
    S = g.empirical_covariance(X)
    R = g.partial_correlations(np.linalg.inv(S))
    Z = np.c_[np.ones(len(X)), X[:, 2]]
    r0 = X[:, 0] - Z @ np.linalg.lstsq(Z, X[:, 0], rcond=None)[0]
    r1 = X[:, 1] - Z @ np.linalg.lstsq(Z, X[:, 1], rcond=None)[0]
    assert abs(R[0, 1] - np.corrcoef(r0, r1)[0, 1]) < 1e-10


def test_em_sans_manquant_egal_17_1():
    rng = np.random.RandomState(7)
    A = g.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    X = rng.randn(200, 4)
    mu, W, Theta, _ = g.em_gaussian_graph(X, A)
    W2, _ = g.fit_known_graph(g.empirical_covariance(X), A)
    assert np.allclose(W, W2, atol=1e-8)
    assert np.allclose(mu, X.mean(0))


# --- Ising -----------------------------------------------------------------

def _ising(seed=0, p=5):
    rng = np.random.RandomState(seed)
    A = g.adjacency_from_edges(p, [(j, j + 1) for j in range(p - 1)] + [(0, p - 1)])
    th0 = rng.uniform(-1, 0.5, p)
    Th = np.zeros((p, p))
    for j, k in g.edges_from_adjacency(A):
        Th[j, k] = Th[k, j] = rng.uniform(-1.5, 1.5)
    return rng, A, th0, Th


def test_ising_probas_et_partition():
    _, A, th0, Th = _ising()
    X, pr = g.ising_probs(th0, Th)
    assert abs(pr.sum() - 1) < 1e-12
    Phi = g.ising_log_partition(th0, Th)
    assert np.allclose(np.log(pr), g.ising_energy(X, th0, Th) - Phi)


def test_ising_conditionnelle_logistique():
    _, A, th0, Th = _ising(1)
    X, pr = g.ising_probs(th0, Th)
    j = 2
    for x in X[:8]:
        x1, x0 = x.copy(), x.copy()
        x1[j], x0[j] = 1, 0
        p1 = np.exp(g.ising_energy(x1, th0, Th))[0]
        p0 = np.exp(g.ising_energy(x0, th0, Th))[0]
        assert abs(g.ising_conditional(x, j, th0, Th)[0] - p1 / (p0 + p1)) < 1e-12
