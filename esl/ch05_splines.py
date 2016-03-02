"""Bases de fonctions, splines et lissage - chapitre 5 d'ESL.

Ce module regroupe les outils réutilisés dans les notebooks du chapitre 5 :

* bases de polynômes par morceaux et base des puissances tronquées (§5.2) ;
* B-splines par la récurrence de Cox-de Boor (annexe du chapitre, éq. (5.77)-(5.78)),
  ainsi que leurs dérivées ;
* base des splines cubiques naturelles (éq. (5.4)-(5.5)) ;
* matrice de pénalité :math:`\\Omega_{jk} = \\int B_j''(t) B_k''(t)\\,dt` et spline de
  lissage (éq. (5.9)-(5.12), (5.79)), matrice de lissage, degrés de liberté effectifs
  (éq. (5.16)), validation croisée « leave-one-out » (éq. (5.27)) ;
* transformée de Haar rapide et seuillage doux (§5.9) ;
* régression logistique par IRLS (utilisée pour l'exemple des phonèmes, §5.2.3).
"""

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "truncated_power_basis", "piecewise_polynomial_basis", "augmented_knots",
    "bspline_basis", "natural_spline_basis", "bspline_penalty_matrix",
    "SmoothingSpline", "smoother_matrix", "effective_df", "lambda_for_df",
    "loocv_score", "gcv_score", "haar_transform", "inverse_haar_transform",
    "haar_matrix", "soft_threshold", "logistic_irls",
]


# ---------------------------------------------------------------------------
# Polynômes par morceaux et puissances tronquées
# ---------------------------------------------------------------------------

def piecewise_polynomial_basis(x, knots, degree=3, continuity=None):
    """Base de polynômes par morceaux de degré ``degree`` avec nœuds ``knots``.

    La base est construite à partir des fonctions :math:`1, x, \\dots, x^d` et,
    pour chaque nœud :math:`\\xi`, des puissances tronquées
    :math:`(x-\\xi)_+^j` pour :math:`j = c+1, \\dots, d`, où :math:`c` est
    l'ordre de continuité imposé (``continuity``). On convient que
    :math:`(x-\\xi)_+^0 = \\mathbf 1(x \\geq \\xi)`.

    Parameters
    ----------
    x : array_like, shape (N,)
        Points d'évaluation.
    knots : array_like, shape (K,)
        Nœuds intérieurs.
    degree : int
        Degré :math:`d` des polynômes (ordre :math:`M = d + 1`).
    continuity : int or None
        Les dérivées d'ordre :math:`0, \\dots, c` sont continues aux nœuds.
        ``-1`` : fonction discontinue ; ``None`` (défaut) : :math:`c = d-1`,
        c'est-à-dire une spline (base des puissances tronquées).

    Returns
    -------
    H : ndarray, shape (N, d + 1 + K (d - c))
    """
    x = np.asarray(x, dtype=float)
    knots = np.atleast_1d(np.asarray(knots, dtype=float))
    if continuity is None:
        continuity = degree - 1
    cols = [x ** j for j in range(degree + 1)]
    for xi in knots:
        for j in range(continuity + 1, degree + 1):
            if j == 0:
                cols.append((x >= xi).astype(float))
            else:
                cols.append(np.where(x > xi, (x - xi) ** j, 0.0))
    return np.column_stack(cols)
