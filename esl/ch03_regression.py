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


def successive_orthogonalization(X):
    """Algorithme 3.1 : régression par orthogonalisations successives.

    Parameters
    ----------
    X : ndarray (N, p+1) dont la première colonne vaut 1.

    Returns
    -------
    Z : ndarray (N, p+1)
        Résidus orthogonaux ``z_0 = 1, z_1, ..., z_p`` (non normés).
    Gamma : ndarray (p+1, p+1)
        Triangulaire supérieure à diagonale unité, ``X = Z Gamma`` (éq. 3.30),
        ``Gamma[l, j] = <z_l, x_j> / <z_l, z_l>``.
    """
    X = np.asarray(X, float)
    q = X.shape[1]
    Z = np.zeros_like(X)
    G = np.eye(q)
    for j in range(q):
        z = X[:, j].copy()
        for l in range(j):
            G[l, j] = (Z[:, l] @ X[:, j]) / (Z[:, l] @ Z[:, l])
            z -= G[l, j] * Z[:, l]
        Z[:, j] = z
    return Z, G


def ols_fit(X, y, intercept=True):
    """Ajustement par moindres carrés et inférence classique (section 3.2).

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,)
    intercept : bool
        Ajouter une colonne de uns.

    Returns
    -------
    dict avec les clés ``coef`` (p+1 ou p), ``se`` (erreurs standard),
    ``z`` (scores Z, éq. 3.12), ``sigma2`` (estimateur sans biais, éq. 3.8),
    ``rss``, ``df_resid`` = N - (nombre de paramètres), ``fitted``,
    ``XtX_inv``.
    """
    A = add_intercept(X) if intercept else np.asarray(X, float)
    y = np.asarray(y, float)
    N, q = A.shape
    Q, R = gram_schmidt_qr(A)
    coef = back_substitution(R, Q.T @ y)
    fitted = A @ coef
    rss = float(np.sum((y - fitted) ** 2))
    df = N - q
    sigma2 = rss / df
    Rinv = np.linalg.solve(R, np.eye(q))
    XtX_inv = Rinv @ Rinv.T          # (R^T R)^{-1} = R^{-1} R^{-T}
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    return {"coef": coef, "se": se, "z": coef / se, "sigma2": sigma2,
            "rss": rss, "df_resid": df, "fitted": fitted, "XtX_inv": XtX_inv}
