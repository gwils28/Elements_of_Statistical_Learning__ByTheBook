"""Chapitre 12 : machines à vecteurs de support et discriminants flexibles.

Ce module regroupe les implémentations « à la main » utilisées dans les
notebooks du chapitre 12 :

* noyaux (linéaire, polynomial, radial) ;
* un solveur SMO (*Sequential Minimal Optimization*) pour le problème dual
  quadratique commun aux SVM de classification et de régression ;
* les classes :class:`SVMClassifier` et :class:`SVMRegressor` ;
* les fonctions de perte de la figure 12.4 et de la figure 12.8 ;
* l'analyse discriminante linéaire (LDA), flexible (FDA, par *optimal
  scoring*), pénalisée (PDA) et par mélanges (MDA, algorithme EM) ;
* des simulateurs (mélange gaussien des chapitres 2 et 12, *waveform*).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Noyaux
# ---------------------------------------------------------------------------


def linear_kernel(X, Z):
    """Noyau linéaire :math:`K(x, z) = \\langle x, z\\rangle`.

    Parameters
    ----------
    X : ndarray (n, p)
    Z : ndarray (m, p)

    Returns
    -------
    ndarray (n, m)
    """
    return np.asarray(X, float) @ np.asarray(Z, float).T


def polynomial_kernel(X, Z, degree=2, coef0=1.0, gamma=1.0):
    """Noyau polynomial :math:`K(x, z) = (\\text{coef0} + \\gamma\\langle x, z\\rangle)^d`.

    Avec ``coef0=1`` et ``gamma=1`` on retrouve le noyau (12.22) du livre.
    """
    return (coef0 + gamma * linear_kernel(X, Z)) ** degree


def rbf_kernel(X, Z, gamma=1.0):
    """Noyau radial (gaussien) :math:`K(x, z) = \\exp(-\\gamma\\|x - z\\|^2)`."""
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    d2 = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * X @ Z.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def make_kernel(kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
    """Renvoie une fonction ``K(X, Z)`` correspondant au noyau demandé.

    Parameters
    ----------
    kernel : {"linear", "poly", "rbf"} ou fonction
        Nom du noyau, ou directement une fonction ``K(X, Z)``.
    gamma, degree, coef0 : float
        Paramètres du noyau (voir :func:`polynomial_kernel`, :func:`rbf_kernel`).
    """
    if callable(kernel):
        return kernel
    if kernel == "linear":
        return linear_kernel
    if kernel == "poly":
        return lambda X, Z: polynomial_kernel(X, Z, degree, coef0, gamma)
    if kernel == "rbf":
        return lambda X, Z: rbf_kernel(X, Z, gamma)
    raise ValueError("noyau inconnu : %r" % kernel)
