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
