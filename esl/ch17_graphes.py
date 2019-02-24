"""Chapitre 17 - Modèles graphiques non orientés.

Ce module regroupe les outils du chapitre :

* vocabulaire des graphes (matrice d'adjacence, cliques maximales,
  séparation) et dessin d'un graphe avec matplotlib ;
* modèles graphiques gaussiens : estimation à graphe connu
  (algorithme 17.1 du livre), lasso par descente de coordonnées
  (éq. 17.26), graphical lasso (algorithme 17.2), algorithme EM avec
  données manquantes (exercice 17.9) ;
* modèles de Ising (machines de Boltzmann entièrement visibles) pour de
  petites valeurs de ``p`` : énumération des ``2^p`` états, fonction de
  partition, maximum de vraisemblance par Newton, pseudo-vraisemblance,
  régressions logistiques nœud par nœud.

Conventions pour le modèle de Ising : les variables valent 0 ou 1 ; les
paramètres sont un vecteur ``theta0`` (arêtes vers le nœud constant
``X_0 = 1``) et une matrice symétrique ``Theta`` de diagonale nulle, de sorte
que ::

    log p(x) = sum_j theta0[j] x_j + sum_{j<k} Theta[j, k] x_j x_k - Phi.
"""

import itertools

import numpy as np

# ---------------------------------------------------------------------------
# 1. Graphes
# ---------------------------------------------------------------------------


def adjacency_from_edges(p, edges):
    """Matrice d'adjacence booléenne (p, p) à partir d'une liste d'arêtes.

    Parameters
    ----------
    p : int
        Nombre de sommets (numérotés 0, ..., p-1).
    edges : iterable de couples (j, k)
        Arêtes non orientées.

    Returns
    -------
    A : ndarray (p, p) de booléens, symétrique, diagonale nulle.
    """
    A = np.zeros((p, p), dtype=bool)
    for j, k in edges:
        if j != k:
            A[j, k] = A[k, j] = True
    return A


def edges_from_adjacency(A):
    """Liste des arêtes ``(j, k)`` avec ``j < k`` d'une matrice d'adjacence."""
    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    return [(j, k) for j in range(p) for k in range(j + 1, p) if A[j, k]]


def support_graph(Theta, tol=1e-8):
    """Graphe (adjacence) des coefficients hors diagonale non nuls de ``Theta``."""
    A = np.abs(np.asarray(Theta)) > tol
    np.fill_diagonal(A, False)
    return A | A.T


def maximal_cliques(A):
    """Cliques maximales d'un graphe (algorithme de Bron–Kerbosch avec pivot).

    Returns
    -------
    list de tuples triés, eux-mêmes triés par ordre lexicographique.
    """
    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    nbrs = [set(np.flatnonzero(A[j])) for j in range(p)]
    cliques = []

    def bk(R, P, X):
        if not P and not X:
            cliques.append(tuple(sorted(int(r) for r in R)))
            return
        pivot = max(P | X, key=lambda u: len(nbrs[u] & P))
        for v in list(P - nbrs[pivot]):
            bk(R | {v}, P & nbrs[v], X & nbrs[v])
            P = P - {v}
            X = X | {v}

    bk(set(), set(range(p)), set())
    return sorted(cliques)


def separates(A, a, b, c):
    """Teste si l'ensemble ``c`` sépare les ensembles ``a`` et ``b``.

    ``c`` sépare ``a`` et ``b`` si tout chemin d'un sommet de ``a`` à un
    sommet de ``b`` passe par un sommet de ``c`` ; autrement dit, après
    suppression des sommets de ``c``, aucun sommet de ``b`` n'est atteignable
    depuis ``a`` (parcours en largeur).
    """
    A = np.asarray(A, dtype=bool)
    a, b, c = set(a), set(b), set(c)
    if (a & b) - c:
        return False
    start = a - c
    seen = set(start)
    stack = list(start)
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(A[u]):
            v = int(v)
            if v in c or v in seen:
                continue
            if v in b:
                return False
            seen.add(v)
            stack.append(v)
    return True


def connected_components(A):
    """Composantes connexes (liste de listes triées de sommets)."""
    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    label = -np.ones(p, dtype=int)
    comps = []
    for s in range(p):
        if label[s] >= 0:
            continue
        stack, comp = [s], []
        label[s] = len(comps)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(A[u]):
                if label[v] < 0:
                    label[v] = len(comps)
                    stack.append(int(v))
        comps.append(sorted(comp))
    return comps
