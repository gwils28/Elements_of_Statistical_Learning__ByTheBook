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


def forward(params, X, output="identity"):
    """Passe avant : calcule toutes les quantités intermédiaires.

    Returns
    -------
    dict
        ``Z`` (N, M) unités cachées (``X`` si M = 0), ``T`` (N, K)
        combinaisons linéaires de sortie et ``F`` (N, K) sorties
        :math:`f_k(x_i)`.
    """
    X = np.asarray(X, dtype=float)
    if params["alpha"] is None:
        Z = X
    else:
        alpha = params["alpha"]
        Z = sigmoid(X @ alpha[:, 1:].T + alpha[:, 0])  # alpha_0m + alpha_m^T x_i
    beta = params["beta"]
    T = Z @ beta[:, 1:].T + beta[:, 0]                 # beta_0k + beta_k^T z_i
    return {"Z": Z, "T": T, "F": output_function(T, output)}


def data_loss(F, Y, loss="squared"):
    """Erreur d'ajustement R : somme des carrés (11.9) ou entropie croisée (11.10)."""
    if loss == "squared":
        return float(np.sum((Y - F) ** 2))
    if loss == "cross_entropy":
        return float(-np.sum(Y * np.log(np.clip(F, 1e-300, None))))
    raise ValueError("perte inconnue : %s" % loss)


def penalty(params):
    """Pénalité de weight decay (11.16), **hors termes constants** (biais).

    :math:`J(\\theta) = \\sum_{k,m\\ge 1} \\beta_{km}^2 + \\sum_{m,\\ell\\ge 1} \\alpha_{m\\ell}^2`.
    """
    J = np.sum(params["beta"][:, 1:] ** 2)
    if params["alpha"] is not None:
        J += np.sum(params["alpha"][:, 1:] ** 2)
    return float(J)


def backprop(params, X, Y, output="identity", loss="squared", weight_decay=0.0):
    """Erreur pénalisée et son gradient par rétropropagation.

    Calcule :math:`R(\\theta) + \\lambda J(\\theta)` et son gradient.
    La passe arrière suit les éq. (11.14)–(11.15) du livre :

    * erreurs de sortie :math:`\\delta_{ki} = \\partial R_i / \\partial T_{ki}` ;
    * erreurs cachées :math:`s_{mi} = \\sigma'(\\alpha_{0m}+\\alpha_m^T x_i)
      \\sum_k \\beta_{km} \\delta_{ki}` ;
    * :math:`\\partial R / \\partial \\beta_{km} = \\sum_i \\delta_{ki} z_{mi}`,
      :math:`\\partial R / \\partial \\alpha_{m\\ell} = \\sum_i s_{mi} x_{i\\ell}`.

    Parameters
    ----------
    params : dict
        Poids ``alpha`` et ``beta`` (voir :func:`init_params`).
    X, Y : ndarray
        Entrées (N, p) et sorties (N, K).
    output : {"identity", "sigmoid", "softmax"}
        Fonction de sortie :math:`g`.
    loss : {"squared", "cross_entropy"}
        Erreur (11.9) ou (11.10).
    weight_decay : float
        Paramètre :math:`\\lambda \\ge 0`.

    Returns
    -------
    R : float
        Valeur de l'erreur pénalisée.
    grads : dict
        Gradients de même forme que ``params``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    fw = forward(params, X, output)
    Z, F = fw["Z"], fw["F"]
    R = data_loss(F, Y, loss) + weight_decay * penalty(params)

    # --- dérivée par rapport aux sorties T (les "delta") ---------------------
    if output == "softmax" and loss == "cross_entropy":
        delta = F - Y                                  # simplification (Ex. 11.3)
    else:
        if loss == "squared":
            dF = -2.0 * (Y - F)
        else:
            dF = -Y / np.clip(F, 1e-300, None)
        if output == "identity":
            delta = dF
        elif output == "sigmoid":
            delta = dF * F * (1.0 - F)
        else:  # softmax : jacobienne diag(f) - f f^T
            delta = F * (dF - np.sum(dF * F, axis=1, keepdims=True))

    # --- gradient de la couche de sortie -------------------------------------
    g_beta = np.empty_like(params["beta"])             # (K, M+1)
    g_beta[:, 0] = delta.sum(axis=0)
    g_beta[:, 1:] = delta.T @ Z
    g_beta[:, 1:] += 2.0 * weight_decay * params["beta"][:, 1:]
    grads = {"alpha": None, "beta": g_beta}

    # --- rétropropagation vers la couche cachée ------------------------------
    if params["alpha"] is not None:
        s = (delta @ params["beta"][:, 1:]) * Z * (1.0 - Z)   # sigma' = sigma(1-sigma)
        g_alpha = np.empty_like(params["alpha"])       # (M, p+1)
        g_alpha[:, 0] = s.sum(axis=0)
        g_alpha[:, 1:] = s.T @ X
        g_alpha[:, 1:] += 2.0 * weight_decay * params["alpha"][:, 1:]
        grads["alpha"] = g_alpha
    return R, grads


# ---------------------------------------------------------------------------
# Vérification numérique du gradient
# ---------------------------------------------------------------------------


def numerical_gradient(fun, theta, eps=1e-6):
    """Gradient de ``fun`` en ``theta`` par différences finies centrées.

    :math:`\\partial_j f(\\theta) \\approx
    [f(\\theta + \\varepsilon e_j) - f(\\theta - \\varepsilon e_j)] / (2\\varepsilon)`,
    avec une erreur en :math:`O(\\varepsilon^2)`.
    """
    theta = np.asarray(theta, dtype=float)
    g = np.zeros_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        g[j] = (fun(theta + e) - fun(theta - e)) / (2 * eps)
    return g
