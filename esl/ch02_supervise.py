"""Outils du chapitre 2 : vue d'ensemble de l'apprentissage supervisé.

Ce module regroupe les implémentations « à la main » utilisées dans les
notebooks du chapitre 2 :

* moindres carrés (éq. (2.6) du livre) et classement par régression
  linéaire d'un codage 0/1 ;
* k plus proches voisins (éq. (2.8)) en régression et en classification ;
* la simulation « mixture » de la section 2.3.3 (mélanges de gaussiennes)
  et le classifieur de Bayes associé ;
* quelques formules du fléau de la dimension (section 2.5) ;
* les poids de lissage linéaires de l'exercice 2.7 et la réduction des
  ex-aequo de l'exercice 2.6.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Moindres carrés
# ---------------------------------------------------------------------------


def add_intercept(X):
    """Ajoute une colonne de 1 en tête de la matrice de design.

    Parameters
    ----------
    X : ndarray, shape (n, p)

    Returns
    -------
    ndarray, shape (n, p + 1)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(X.shape[0]), X])


def least_squares_fit(X, y, intercept=True, weights=None):
    """Estimateur des moindres carrés (éventuellement pondérés).

    Résout les équations normales :math:`X^T W X \\beta = X^T W y`
    (éq. (2.5)-(2.6) du livre lorsque :math:`W = I`).

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Matrice des entrées.
    y : ndarray, shape (n,) ou (n, K)
        Réponse(s).
    intercept : bool
        Si vrai, une constante est ajoutée au modèle (coefficient en tête).
    weights : ndarray, shape (n,), optionnel
        Poids positifs :math:`w_i` du critère :math:`\\sum_i w_i (y_i - x_i^T\\beta)^2`.

    Returns
    -------
    beta : ndarray, shape (p + intercept,) ou (p + intercept, K)
    """
    A = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=float))
        A = A * sw[:, None]
        y = y * (sw if y.ndim == 1 else sw[:, None])
    # lstsq (décomposition SVD) est numériquement plus sûr que l'inverse explicite
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta
