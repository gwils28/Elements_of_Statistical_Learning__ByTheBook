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


def circle_layout(p, start=np.pi / 2):
    """Positions de ``p`` sommets régulièrement répartis sur le cercle unité
    (sens horaire, premier sommet en haut)."""
    ang = start - 2 * np.pi * np.arange(p) / p
    return np.c_[np.cos(ang), np.sin(ang)]


def draw_graph(A, labels=None, pos=None, weights=None, ax=None, title=None,
               node_color="#f0efec", edge_color="#2a78d6", node_size=0.11,
               fontsize=10, cmap=None, vmax=None):
    """Dessine un graphe non orienté avec matplotlib.

    Parameters
    ----------
    A : ndarray (p, p) booléen
        Matrice d'adjacence.
    labels : liste de str, optionnelle
        Étiquettes des sommets (par défaut ``X1, ..., Xp``).
    pos : ndarray (p, 2), optionnel
        Positions des sommets (par défaut disposition circulaire).
    weights : ndarray (p, p), optionnel
        Poids des arêtes (par ex. corrélations partielles). L'épaisseur est
        proportionnelle à ``|poids|`` ; si ``cmap`` est donné, la couleur
        code le signe et l'intensité.
    ax : axes matplotlib, optionnel.
    title : str, optionnel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    if labels is None:
        labels = ["$X_{%d}$" % (j + 1) for j in range(p)]
    if pos is None:
        pos = circle_layout(p)
    pos = np.asarray(pos, dtype=float)
    if ax is None:
        ax = plt.gca()
    if weights is not None and vmax is None:
        w = np.abs(np.asarray(weights)[A])
        vmax = w.max() if w.size else 1.0
    for j, k in edges_from_adjacency(A):
        lw, col = 1.6, edge_color
        if weights is not None:
            wjk = weights[j, k]
            lw = 0.6 + 4.0 * abs(wjk) / vmax
            if cmap is not None:
                col = cmap(0.5 + 0.5 * np.clip(wjk / vmax, -1, 1))
        ax.plot(pos[[j, k], 0], pos[[j, k], 1], color=col, lw=lw,
                solid_capstyle="round", zorder=1)
    for j in range(p):
        ax.add_patch(Circle(pos[j], node_size, facecolor=node_color,
                            edgecolor="#52514e", lw=1.0, zorder=2))
        ax.text(pos[j, 0], pos[j, 1], labels[j], ha="center", va="center",
                fontsize=fontsize, zorder=3)
    pad = node_size + 0.08
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# 2. Modèles graphiques gaussiens
# ---------------------------------------------------------------------------


def soft_threshold(x, t):
    """Opérateur de seuillage doux ``S(x, t) = sign(x) (|x| - t)_+`` (éq. 17.27)."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def partial_correlations(Theta):
    """Corrélations partielles ``rho_{jk|reste} = -theta_jk / sqrt(theta_jj theta_kk)``.

    La diagonale est fixée à 1 (convention).
    """
    Theta = np.asarray(Theta, dtype=float)
    d = 1.0 / np.sqrt(np.diag(Theta))
    R = -Theta * np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def gaussian_loglik(Theta, S):
    """Log-vraisemblance profilée (éq. 17.11) : ``log det Theta - tr(S Theta)``."""
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return -np.inf
    return logdet - np.trace(S @ Theta)
