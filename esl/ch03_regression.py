"""Méthodes linéaires de régression - chapitre 3 d'ESL.

Implémentations « à la main » (numpy) des méthodes du chapitre :

* moindres carrés ordinaires par décomposition QR (Gram–Schmidt) et
  statistiques associées (erreurs standard, scores Z, test F) ;
* régression par orthogonalisations successives (algorithme 3.1) ;
* sélection de sous-ensembles : meilleur sous-ensemble, pas à pas ascendant
  (« forward stepwise ») et descendant, régression « stagewise » ;
* régression ridge (via la SVD) et degrés de liberté effectifs ;
* lasso par descente de coordonnées ;
* LAR (« least angle regression ») et sa modification lasso ;
* régression sur composantes principales (PCR) et moindres carrés partiels
  (PLS, algorithme 3.3) ;
* validation croisée à K plis et règle « un écart-type ».

Convention : sauf mention contraire, les fonctions d'ajustement reçoivent la
matrice ``X`` (N, p) **sans** colonne de uns ; l'ordonnée à l'origine est
traitée en centrant ``X`` et ``y`` puis en posant
``b0 = mean(y) - mean(X) @ beta``. Elles renvoient ``(b0, beta)``.
"""

from itertools import combinations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

PROSTATE_NAMES = ["lcavol", "lweight", "age", "lbph", "svi", "lcp",
                  "gleason", "pgg45"]


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def load_prostate_standardized():
    """Données prostate standardisées comme dans le livre.

    Les 8 prédicteurs sont centrés et réduits avec la moyenne et l'écart-type
    (``ddof=1``) calculés sur les 97 observations ; c'est ce choix qui
    reproduit exactement le tableau 3.2.

    Returns
    -------
    Xtr, ytr, Xte, yte : ndarray
        Apprentissage (67 obs.) et test (30 obs.).
    names : list of str
        Noms des prédicteurs.
    """
    from esl import datasets
    df = datasets.load_prostate()
    X = df[PROSTATE_NAMES].to_numpy(float)
    y = df["lpsa"].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    tr = df["train"].to_numpy(bool)
    return X[tr], y[tr], X[~tr], y[~tr], list(PROSTATE_NAMES)


def _center(X, y):
    """Centre ``X`` (colonnes) et ``y`` ; renvoie aussi les moyennes."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    xm, ym = X.mean(0), y.mean()
    return X - xm, y - ym, xm, ym


def add_intercept(X):
    """Ajoute une première colonne de uns à ``X``."""
    X = np.asarray(X, float)
    return np.column_stack([np.ones(X.shape[0]), X])


# ---------------------------------------------------------------------------
# Moindres carrés, QR, Gram–Schmidt
# ---------------------------------------------------------------------------

def gram_schmidt_qr(X):
    """Décomposition QR « réduite » par Gram–Schmidt (version modifiée).

    Parameters
    ----------
    X : ndarray (N, q), de rang plein q.

    Returns
    -------
    Q : ndarray (N, q)
        Colonnes orthonormées.
    R : ndarray (q, q)
        Triangulaire supérieure à diagonale > 0, avec ``X = Q R``.
    """
    X = np.asarray(X, float)
    N, q = X.shape
    Q = X.copy()
    R = np.zeros((q, q))
    for j in range(q):
        for k in range(j):
            R[k, j] = Q[:, k] @ Q[:, j]
            Q[:, j] -= R[k, j] * Q[:, k]
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] /= R[j, j]
    return Q, R


def back_substitution(R, b):
    """Résout ``R x = b`` pour ``R`` triangulaire supérieure (remontée)."""
    R = np.asarray(R, float)
    b = np.asarray(b, float)
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def ols_qr(X, y):
    """Moindres carrés via QR : ``beta = R^{-1} Q^T y`` (éq. 3.32).

    ``X`` doit déjà contenir la colonne de uns si l'on veut une ordonnée à
    l'origine.
    """
    Q, R = gram_schmidt_qr(X)
    return back_substitution(R, Q.T @ y)
