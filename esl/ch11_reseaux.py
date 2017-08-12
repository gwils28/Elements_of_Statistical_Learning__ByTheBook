"""Chapitre 11 - Réseaux de neurones et projection pursuit regression.

Ce module rassemble les outils utilisés dans les notebooks du chapitre 11 :

* un réseau de neurones à **une couche cachée** (modèle (11.5) du livre),
  écrit entièrement en numpy, avec la **rétropropagation** du gradient
  (équations (11.12)–(11.15)), le *weight decay* (11.16) et plusieurs
  algorithmes d'entraînement (descente de gradient par lots, descente de
  gradient stochastique par mini-lots, L-BFGS) ;
* une **vérification numérique du gradient** par différences finies
  centrées ;
* un **lisseur spline cubique** (smoothing spline) à nombre de degrés de
  liberté fixé, et la **projection pursuit regression** (PPR, section 11.2)
  ajustée par l'algorithme alterné « lissage / Gauss–Newton ».

Notations (celles du livre)
---------------------------
* ``X`` : matrice (N, p) des entrées ; ``Y`` : matrice (N, K) des sorties
  (codage 0/1 « one-hot » en classification) ;
* ``alpha`` : matrice (M, p+1) ; la ligne m contient
  :math:`(\\alpha_{0m}, \\alpha_m^T)` ;
* ``beta`` : matrice (K, M+1) ; la ligne k contient
  :math:`(\\beta_{0k}, \\beta_k^T)` ;
* :math:`Z_m = \\sigma(\\alpha_{0m} + \\alpha_m^T X)`,
  :math:`T_k = \\beta_{0k} + \\beta_k^T Z`, :math:`f_k(X) = g_k(T)`.

Quand ``M = 0`` il n'y a pas de couche cachée : ``alpha`` vaut ``None`` et
``beta`` est de taille (K, p+1) (modèle linéaire / logistique multinomial).
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline

try:  # dépendance de scikit-learn
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover
    _threadpool_limits = None


def limit_blas_threads(n=1):
    """Limite le nombre de fils d'exécution de BLAS/LAPACK.

    Les calculs de ce chapitre portent sur de **petites** matrices (quelques
    centaines de lignes) répétées des milliers de fois : la parallélisation
    de BLAS y coûte plus cher qu'elle ne rapporte (facteur 10 à 100 observé
    sur une machine à nombreux cœurs). Cette fonction est appelée à
    l'import du module.
    """
    if _threadpool_limits is not None:
        return _threadpool_limits(n)
    return None


limit_blas_threads(1)
