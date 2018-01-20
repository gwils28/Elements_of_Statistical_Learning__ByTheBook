"""Méthodes à prototypes et plus proches voisins - chapitre 13 d'ESL.

Ce module regroupe des implémentations explicites (numpy) de :

* K-means (algorithme de Lloyd) et son utilisation comme classifieur
  (R prototypes par classe) ;
* LVQ1 (Learning Vector Quantization, algorithme 13.1 du livre) ;
* un classifieur par mélanges gaussiens à covariance commune, ajusté par EM
  (section 13.2.3) ;
* les k plus proches voisins (avec plusieurs valeurs de k d'un coup et
  validation croisée) ;
* la métrique adaptative DANN (éq. (13.8)-(13.9)) ;
* les vecteurs tangents d'images (rotation, translations, homothéties,
  cisaillement, épaisseur) et la distance tangente (section 13.3.3) ;
* quelques outils : rayon médian du 1-voisinage (éq. (13.7)), simulations
  du chapitre, condensation de Hart, multi-édition.
"""

import numpy as np
from scipy.special import gammaln, logsumexp


# ---------------------------------------------------------------------------
# Outils de base
# ---------------------------------------------------------------------------

def sq_dists(A, B):
    """Distances euclidiennes au carré entre les lignes de ``A`` et de ``B``.

    Parameters
    ----------
    A : ndarray (n, p)
    B : ndarray (m, p)

    Returns
    -------
    D : ndarray (n, m), ``D[i, j] = ||A_i - B_j||^2`` (tronqué à 0).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    D = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(D, 0.0)


def kfold_indices(n, n_folds, rng=None):
    """Découpe aléatoire de ``range(n)`` en ``n_folds`` blocs (liste d'indices)."""
    rng = np.random.RandomState(0) if rng is None else rng
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, n_folds)]


# ---------------------------------------------------------------------------
# K-means et classifieur K-means
# ---------------------------------------------------------------------------

def kmeans(X, R, n_init=5, max_iter=100, rng=None, init=None):
    """Algorithme de Lloyd : minimise l'inertie intra-groupes.

    Critère : :math:`\\sum_i \\min_r \\|x_i - m_r\\|^2`.

    Parameters
    ----------
    X : ndarray (N, p)
    R : int
        Nombre de centres.
    n_init : int
        Nombre de redémarrages aléatoires (on garde le meilleur).
    max_iter : int
    rng : RandomState
    init : ndarray (R, p), optionnel
        Centres initiaux (dans ce cas un seul démarrage).

    Returns
    -------
    centers : ndarray (R, p)
    labels : ndarray (N,)
    inertia : float
    history : list
        Valeurs successives du critère pour le meilleur démarrage.
    """
    X = np.asarray(X, float)
    rng = np.random.RandomState(0) if rng is None else rng
    N = X.shape[0]
    R = min(R, N)
    best = None
    starts = [np.array(init, float)] if init is not None else [None] * n_init
    for start in starts:
        C = X[rng.choice(N, R, replace=False)].copy() if start is None else start.copy()
        hist = []
        labels = None
        for _ in range(max_iter):
            D = sq_dists(X, C)
            new_labels = D.argmin(1)
            hist.append(D[np.arange(N), new_labels].sum())
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for r in range(R):
                mask = labels == r
                if mask.any():
                    C[r] = X[mask].mean(0)
                else:  # groupe vide : on le relance sur un point au hasard
                    C[r] = X[rng.randint(N)]
        inertia = hist[-1]
        if best is None or inertia < best[2]:
            best = (C, labels, inertia, hist)
    return best


class KMeansClassifier:
    """Classifieur à prototypes obtenus par K-means dans chaque classe.

    Parameters
    ----------
    R : int
        Nombre de prototypes par classe.
    n_init : int
    rng : RandomState
    """

    def __init__(self, R=5, n_init=5, rng=None):
        self.R = R
        self.n_init = n_init
        self.rng = np.random.RandomState(0) if rng is None else rng

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        protos, labs = [], []
        for k in self.classes_:
            C, _, _, _ = kmeans(X[y == k], self.R, n_init=self.n_init, rng=self.rng)
            protos.append(C)
            labs.append(np.full(len(C), k))
        self.prototypes_ = np.vstack(protos)
        self.proto_labels_ = np.concatenate(labs)
        return self

    def predict(self, X):
        return nearest_prototype(X, self.prototypes_, self.proto_labels_)


def nearest_prototype(X, prototypes, proto_labels):
    """Classe du prototype le plus proche (distance euclidienne)."""
    D = sq_dists(X, prototypes)
    return np.asarray(proto_labels)[D.argmin(1)]


# ---------------------------------------------------------------------------
# LVQ1
# ---------------------------------------------------------------------------

def lvq1(X, y, prototypes, proto_labels, n_iter=None, eps0=0.05, rng=None):
    """Algorithme LVQ1 (algorithme 13.1 du livre).

    À chaque itération on tire un point d'apprentissage ``x_i`` au hasard ;
    le prototype le plus proche ``m`` est rapproché de ``x_i``
    (``m <- m + eps (x_i - m)``) s'il est de la même classe, éloigné
    (``m <- m - eps (x_i - m)``) sinon. Le pas décroît linéairement de
    ``eps0`` à 0.

    Parameters
    ----------
    X, y : données d'apprentissage.
    prototypes : ndarray (M, p)
        Prototypes initiaux (non modifiés : on travaille sur une copie).
    proto_labels : ndarray (M,)
    n_iter : int, par défaut ``20 * N``.
    eps0 : float
        Pas initial.
    rng : RandomState

    Returns
    -------
    M : ndarray (M, p), prototypes finaux.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    rng = np.random.RandomState(0) if rng is None else rng
    M = np.array(prototypes, float, copy=True)
    proto_labels = np.asarray(proto_labels)
    N = X.shape[0]
    n_iter = 20 * N if n_iter is None else n_iter
    idx = rng.randint(N, size=n_iter)
    for t, i in enumerate(idx):
        eps = eps0 * (1.0 - t / n_iter)
        d = ((M - X[i]) ** 2).sum(1)
        j = d.argmin()
        if proto_labels[j] == y[i]:
            M[j] += eps * (X[i] - M[j])
        else:
            M[j] -= eps * (X[i] - M[j])
    return M
