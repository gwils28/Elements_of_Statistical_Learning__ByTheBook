"""Chapitre 8 - Inférence sur les modèles et moyennage de modèles.

Implémentations « à la main » utilisées dans les notebooks du chapitre 8 :

* bases de B-splines cubiques, matrice de pénalité :math:`\\Omega` et
  splines de lissage (sections 8.2 et 5.4 du livre) ;
* ajustement par moindres carrés / maximum de vraisemblance, bandes
  d'erreur standard, bootstrap non paramétrique et paramétrique ;
* loi a posteriori gaussienne des coefficients (éq. (8.27)-(8.28)) ;
* algorithme EM et échantillonneur de Gibbs pour un mélange de deux
  gaussiennes (algorithmes 8.1 et 8.4) ;
* bagging, souches de décision (« stumps »), AdaBoost, bumping et stacking
  (sections 8.7 à 8.9).
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import BSpline

#: Les 20 données fictives du tableau 8.1 du livre (exemple de mélange).
MIXTURE_DATA = np.array([
    -0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53,
    0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22])


# ---------------------------------------------------------------------------
# B-splines et splines de lissage
# ---------------------------------------------------------------------------

def bspline_knots(interior, boundary, degree=3):
    """Suite de nœuds complète d'une base de B-splines.

    Parameters
    ----------
    interior : array_like
        Nœuds intérieurs (triés ou non).
    boundary : tuple (a, b)
        Bornes de l'intervalle ; elles sont répétées ``degree + 1`` fois.
    degree : int
        Degré des polynômes par morceaux (3 pour des splines cubiques).

    Returns
    -------
    ndarray
        Vecteur de nœuds de longueur ``len(interior) + 2 * (degree + 1)`` ;
        la base associée a ``len(interior) + degree + 1`` fonctions.
    """
    a, b = boundary
    return np.r_[[a] * (degree + 1), np.sort(np.asarray(interior, float)),
                 [b] * (degree + 1)]


def bspline_basis(x, knots, degree=3, deriv=0):
    """Matrice des B-splines (ou de leurs dérivées) évaluées en ``x``.

    Parameters
    ----------
    x : array_like, shape (n,)
        Points d'évaluation, dans l'intervalle couvert par les nœuds.
    knots : ndarray
        Suite de nœuds complète (voir :func:`bspline_knots`).
    degree : int
        Degré des splines.
    deriv : int
        Ordre de dérivation (0 = les fonctions elles-mêmes).

    Returns
    -------
    ndarray, shape (n, p)
        Élément ``(i, j)`` : :math:`h_j^{(deriv)}(x_i)`.
    """
    x = np.asarray(x, float)
    p = len(knots) - degree - 1
    spl = BSpline(knots, np.eye(p), degree, extrapolate=False)
    if deriv:
        spl = spl.derivative(deriv)
    out = spl(x)
    return np.nan_to_num(out)


def bspline_penalty(knots, degree=3):
    """Matrice de pénalité :math:`\\Omega_{jk} = \\int h_j''(t) h_k''(t)\\,dt`.

    L'intégrale est calculée exactement par une quadrature de Gauss-Legendre
    à 3 points sur chaque intervalle entre deux nœuds consécutifs (les
    produits :math:`h_j'' h_k''` y sont des polynômes de degré au plus 2 pour
    des splines cubiques).
    """
    u = np.unique(knots)
    gx, gw = np.polynomial.legendre.leggauss(3)
    pts, wts = [], []
    for a, b in zip(u[:-1], u[1:]):
        pts.append((a + b) / 2 + (b - a) / 2 * gx)
        wts.append((b - a) / 2 * gw)
    pts, wts = np.concatenate(pts), np.concatenate(wts)
    D2 = bspline_basis(pts, knots, degree, deriv=2)
    return D2.T @ (wts[:, None] * D2)


def smoother_matrix(H, Omega, lam):
    """Matrice de lissage :math:`S_\\lambda = H (H^TH + \\lambda\\Omega)^{-1} H^T`."""
    A = H.T @ H + lam * Omega
    return H @ np.linalg.solve(A, H.T)


def loocv_linear_smoother(S, y):
    """Erreur de validation croisée « leave-one-out » d'un lisseur linéaire.

    Utilise la formule exacte
    :math:`\\frac1N\\sum_i \\left(\\frac{y_i-\\hat f(x_i)}{1-S_{ii}}\\right)^2`.
    """
    r = (y - S @ y) / (1 - np.diag(S))
    return np.mean(r ** 2)


def gcv_linear_smoother(S, y):
    """Critère de validation croisée généralisée (GCV) d'un lisseur linéaire."""
    n = len(y)
    r = y - S @ y
    return np.mean(r ** 2) / (1 - np.trace(S) / n) ** 2
