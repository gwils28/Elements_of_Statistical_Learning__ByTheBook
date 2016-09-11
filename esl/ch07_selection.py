"""Outils du chapitre 7 : évaluation et sélection de modèles.

Ce module regroupe les briques réutilisées dans les notebooks du chapitre 7 :

* fonctions de perte et lisseurs linéaires (moindres carrés, k plus proches
  voisins, splines de lissage) ;
* critères analytiques (Cp, AIC, BIC, GCV) et nombre effectif de paramètres ;
* validation croisée à K blocs et estimateurs bootstrap de l'erreur
  (bootstrap naïf, bootstrap « leave-one-out », .632 et .632+) ;
* sélection de sous-ensembles (meilleur sous-ensemble, pas à pas ascendant) ;
* régression logistique par Newton-Raphson et base de splines cubiques
  naturelles (exemple des phonèmes) ;
* construction explicite de la famille {I(sin(alpha x) > 0)} qui pulvérise
  les points 10^{-1}, ..., 10^{-l} (dimension VC infinie).
"""

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Pertes
# ---------------------------------------------------------------------------

def squared_loss(y, yhat):
    """Perte quadratique ``(y - yhat)^2``, élément par élément."""
    return (np.asarray(y, float) - np.asarray(yhat, float)) ** 2


def zero_one_loss(y, yhat):
    """Perte 0-1 ``I(y != yhat)``, élément par élément."""
    return (np.asarray(y) != np.asarray(yhat)).astype(float)


# ---------------------------------------------------------------------------
# Lisseurs linéaires  yhat = S y
# ---------------------------------------------------------------------------

def _design(X, intercept=True):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return X


def hat_matrix(X, intercept=True):
    """Matrice chapeau ``H = X (X^T X)^{-1} X^T`` des moindres carrés.

    Parameters
    ----------
    X : array (N, p)
        Prédicteurs.
    intercept : bool
        Ajoute une colonne de 1 si vrai.

    Returns
    -------
    H : array (N, N)
        Projecteur orthogonal sur l'espace des colonnes de X.
    """
    Z = _design(X, intercept)
    Q, _ = np.linalg.qr(Z)
    return Q @ Q.T


def ols_fit_predict(Xtr, ytr, Xte, intercept=True):
    """Ajuste les moindres carrés sur ``(Xtr, ytr)`` et prédit en ``Xte``."""
    Ztr = _design(Xtr, intercept)
    Zte = _design(Xte, intercept)
    beta, *_ = np.linalg.lstsq(Ztr, np.asarray(ytr, float), rcond=None)
    return Zte @ beta


def knn_regression(Xtr, ytr, Xte, k):
    """Régression par les k plus proches voisins (distance euclidienne).

    Le point d'apprentissage lui-même fait partie de son voisinage lorsqu'on
    prédit en un point d'apprentissage.
    """
    Xtr = _design(Xtr, False)
    Xte = _design(Xte, False)
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1, kind="stable")[:, :k]
    return np.asarray(ytr, float)[idx].mean(axis=1)
