"""Chapitre 11 - Réseaux de neurones et projection pursuit regression.

Ce module rassemble les outils utilisés dans les notebooks du chapitre 11 :

* un réseau de neurones à **une couche cachée** (modèle (11.5) du livre),
  écrit entièrement en numpy, avec la **rétropropagation** du gradient
  (équations (11.12)–(11.15)), le *weight decay* (11.16) et plusieurs
  algorithmes d'entraînement (descente de gradient par lots, descente de
  gradient stochastique par mini-lots, L-BFGS) ;
* une **vérification numérique du gradient** par différences finies
  centrées ;
* un **lisseur spline cubique** (smoothing spline) à nombre de degrés de
  liberté fixé, et la **projection pursuit regression** (PPR, section 11.2)
  ajustée par l'algorithme alterné « lissage / Gauss–Newton ».

Notations (celles du livre)
---------------------------
* ``X`` : matrice (N, p) des entrées ; ``Y`` : matrice (N, K) des sorties
  (codage 0/1 « one-hot » en classification) ;
* ``alpha`` : matrice (M, p+1) ; la ligne m contient
  :math:`(\\alpha_{0m}, \\alpha_m^T)` ;
* ``beta`` : matrice (K, M+1) ; la ligne k contient
  :math:`(\\beta_{0k}, \\beta_k^T)` ;
* :math:`Z_m = \\sigma(\\alpha_{0m} + \\alpha_m^T X)`,
  :math:`T_k = \\beta_{0k} + \\beta_k^T Z`, :math:`f_k(X) = g_k(T)`.

Quand ``M = 0`` il n'y a pas de couche cachée : ``alpha`` vaut ``None`` et
``beta`` est de taille (K, p+1) (modèle linéaire / logistique multinomial).
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline

try:  # dépendance de scikit-learn
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover
    _threadpool_limits = None


def limit_blas_threads(n=1):
    """Limite le nombre de fils d'exécution de BLAS/LAPACK.

    Les calculs de ce chapitre portent sur de **petites** matrices (quelques
    centaines de lignes) répétées des milliers de fois : la parallélisation
    de BLAS y coûte plus cher qu'elle ne rapporte (facteur 10 à 100 observé
    sur une machine à nombreux cœurs). Cette fonction est appelée à
    l'import du module.
    """
    if _threadpool_limits is not None:
        return _threadpool_limits(n)
    return None


limit_blas_threads(1)

# ---------------------------------------------------------------------------
# Fonctions d'activation
# ---------------------------------------------------------------------------


def sigmoid(v):
    """Sigmoïde :math:`\\sigma(v) = 1/(1+e^{-v})`.

    On utilise l'identité :math:`\\sigma(v) = (1 + \\tanh(v/2))/2`, qui est
    numériquement stable pour les grandes valeurs de :math:`|v|`.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(v, dtype=float)))


def softmax(T):
    """Softmax ligne par ligne, éq. (11.6) : :math:`e^{T_k}/\\sum_l e^{T_l}`.

    On retranche le maximum de chaque ligne avant l'exponentielle, ce qui ne
    change pas le résultat (invariance par translation) mais évite les
    dépassements numériques.
    """
    T = np.asarray(T, dtype=float)
    E = np.exp(T - T.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def one_hot(y, K=None):
    """Codage 0/1 d'un vecteur de classes ``y`` à valeurs dans {0, ..., K-1}."""
    y = np.asarray(y, dtype=int)
    if K is None:
        K = y.max() + 1
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Paramètres : initialisation, empaquetage
# ---------------------------------------------------------------------------


def init_params(p, M, K, rng, scale=0.7):
    """Poids initiaux uniformes sur [-scale, scale] (section 11.5.3).

    Parameters
    ----------
    p, M, K : int
        Nombre d'entrées, d'unités cachées (éventuellement 0) et de sorties.
    rng : numpy.random.RandomState
        Générateur aléatoire.
    scale : float
        Demi-largeur de l'intervalle des poids initiaux.

    Returns
    -------
    dict
        ``{"alpha": (M, p+1) ou None, "beta": (K, M+1)}``.
    """
    if M == 0:
        return {"alpha": None, "beta": rng.uniform(-scale, scale, (K, p + 1))}
    return {"alpha": rng.uniform(-scale, scale, (M, p + 1)),
            "beta": rng.uniform(-scale, scale, (K, M + 1))}


def pack(params):
    """Concatène ``alpha`` puis ``beta`` en un seul vecteur :math:`\\theta`."""
    parts = [params["beta"].ravel()]
    if params["alpha"] is not None:
        parts.insert(0, params["alpha"].ravel())
    return np.concatenate(parts)


def unpack(theta, p, M, K):
    """Opération inverse de :func:`pack`."""
    theta = np.asarray(theta, dtype=float)
    if M == 0:
        return {"alpha": None, "beta": theta.reshape(K, p + 1)}
    na = M * (p + 1)
    return {"alpha": theta[:na].reshape(M, p + 1),
            "beta": theta[na:].reshape(K, M + 1)}


def n_weights(p, M, K):
    """Nombre total de poids, éq. (11.8) : M(p+1) + K(M+1)."""
    if M == 0:
        return K * (p + 1)
    return M * (p + 1) + K * (M + 1)


# ---------------------------------------------------------------------------
# Propagation avant et rétropropagation
# ---------------------------------------------------------------------------


def output_function(T, output):
    """Fonction de sortie :math:`g` : ``identity``, ``sigmoid`` ou ``softmax``."""
    if output == "identity":
        return T
    if output == "sigmoid":
        return sigmoid(T)
    if output == "softmax":
        return softmax(T)
    raise ValueError("sortie inconnue : %s" % output)
