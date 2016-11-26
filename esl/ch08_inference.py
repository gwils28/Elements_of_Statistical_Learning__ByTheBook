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
