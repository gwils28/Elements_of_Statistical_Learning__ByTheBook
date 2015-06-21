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
