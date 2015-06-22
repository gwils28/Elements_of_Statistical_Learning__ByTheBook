"""Méthodes linéaires de régression : compléments pour les exercices 3.16 à 3.30.

Ce module contient des implémentations explicites (numpy) des méthodes du
chapitre 3 d'ESL utilisées dans les exercices 3.16 à 3.30 :

* estimateurs « orthonormaux » du tableau 3.4 (seuillages dur et doux) ;
* régression ridge (forme fermée, degrés de liberté effectifs) ;
* lasso par descente de coordonnées ;
* LAR et LAR-lasso (algorithmes 3.2 et 3.2a) ;
* régression sur composantes principales (PCR) et moindres carrés partiels
  (PLS, algorithme 3.3) ;
* sélection pas à pas ascendante (forward stepwise) et forward stagewise
  incrémental (algorithme 3.4) ;
* gradient conjugué ;
* analyse des corrélations canoniques (CCA) et régression de rang réduit ;
* augmentation des données pour l'elastic net ;
* outils de validation croisée (plis, règle « un écart-type »).

Sauf mention contraire, les fonctions de régression supposent des
prédicteurs **centrés** (et en général réduits) et une réponse **centrée** :
l'ordonnée à l'origine est alors estimée par la moyenne de la réponse
(exercice 3.5 du livre).
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------

def standardize(X, X_other=None, unit_norm=False):
    """Centre et réduit les colonnes de ``X``.

    Parameters
    ----------
    X : ndarray (N, p)
        Matrice d'apprentissage.
    X_other : ndarray (M, p), optionnel
        Autre matrice (par ex. de test) transformée avec les moyennes et
        écarts-types calculés sur ``X``.
    unit_norm : bool
        Si ``True``, chaque colonne est ramenée à une norme euclidienne 1
        (convention de l'algorithme LAR) ; sinon à une variance 1.

    Returns
    -------
    Xs : ndarray (N, p)
    Xo : ndarray (M, p) ou None
    mean, scale : ndarray (p,)
    """
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    Xc = X - mean
    if unit_norm:
        scale = np.linalg.norm(Xc, axis=0)
    else:
        scale = Xc.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Xs = Xc / scale
    Xo = None if X_other is None else (np.asarray(X_other, float) - mean) / scale
    return Xs, Xo, mean, scale


# ---------------------------------------------------------------------------
# Cas orthonormal (tableau 3.4)
# ---------------------------------------------------------------------------

def hard_threshold(beta, M):
    """Meilleur sous-ensemble de taille ``M`` dans le cas orthonormal.

    Conserve les ``M`` coefficients de plus grande valeur absolue et met
    les autres à zéro : ``beta_j * 1(|beta_j| >= |beta_(M)|)``.
    """
    beta = np.asarray(beta, float)
    out = np.zeros_like(beta)
    if M <= 0:
        return out
    keep = np.argsort(-np.abs(beta))[:M]
    out[keep] = beta[keep]
    return out


def soft_threshold(z, lam):
    """Seuillage doux : ``sign(z) * (|z| - lam)_+`` (solution lasso orthonormale)."""
    z = np.asarray(z, float)
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

def ridge(X, y, lam):
    """Coefficients ridge ``(X^T X + lam I)^{-1} X^T y`` (éq. (3.44)).

    ``X`` et ``y`` sont supposés centrés (pas d'ordonnée à l'origine).
    ``y`` peut être une matrice (N, K) : chaque colonne est traitée.
    """
    X = np.asarray(X, float)
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def ridge_svd_path(X, y, lams):
    """Chemin ridge calculé via la SVD ``X = U D V^T`` (éq. (3.47)).

    Returns
    -------
    betas : ndarray (p, len(lams))
    """
    U, d, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    uy = U.T @ y
    lams = np.atleast_1d(lams)
    coef = (d[:, None] / (d[:, None] ** 2 + lams[None, :])) * uy[:, None]
    return Vt.T @ coef


def ridge_df(d, lam):
    """Degrés de liberté effectifs ``sum d_j^2 / (d_j^2 + lam)`` (éq. (3.50))."""
    d = np.asarray(d, float)
    return np.sum(d ** 2 / (d ** 2 + lam))


def ridge_lambda_for_df(d, df):
    """Valeur de ``lam`` donnant ``df`` degrés de liberté effectifs.

    ``df`` doit être dans ``[0, rang]`` ; ``df = rang`` donne ``lam = 0`` et
    ``df = 0`` donne ``lam = inf``.
    """
    d = np.asarray(d, float)
    rank = np.sum(d > 1e-12 * d.max())
    if df >= rank - 1e-12:
        return 0.0
    if df <= 1e-12:
        return np.inf
    hi = 1.0
    while ridge_df(d, hi) > df:
        hi *= 10.0
    return brentq(lambda l: ridge_df(d, l) - df, 0.0, hi, xtol=1e-12 * hi)


# ---------------------------------------------------------------------------
# Lasso par descente de coordonnées
# ---------------------------------------------------------------------------

def lasso_cd(X, y, lam, beta0=None, tol=1e-10, max_iter=10000):
    """Lasso ``min 1/2 ||y - X b||^2 + lam ||b||_1`` par descente de coordonnées.

    Chaque coordonnée est mise à jour par seuillage doux (solution exacte du
    problème à une variable), cycliquement, jusqu'à stabilisation.

    Parameters
    ----------
    X : ndarray (N, p), y : ndarray (N,)
        Données supposées centrées.
    lam : float
        Paramètre de pénalité (convention avec le facteur 1/2, éq. (3.52)).
    beta0 : ndarray (p,), optionnel
        Point de départ (« démarrage à chaud »).

    Returns
    -------
    beta : ndarray (p,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    beta = np.zeros(p) if beta0 is None else np.array(beta0, float)
    col_sq = np.sum(X ** 2, axis=0)
    r = y - X @ beta
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            if col_sq[j] == 0:
                continue
            old = beta[j]
            zj = X[:, j] @ r + col_sq[j] * old
            new = soft_threshold(zj, lam) / col_sq[j]
            if new != old:
                r -= X[:, j] * (new - old)
                beta[j] = new
                max_change = max(max_change, abs(new - old) * np.sqrt(col_sq[j]))
        if max_change < tol:
            break
    return beta
