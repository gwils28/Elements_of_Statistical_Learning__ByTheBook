"""Outils du chapitre 18 : problèmes en grande dimension (p >> N).

Contenu
-------
- centroïdes réduits (*nearest shrunken centroids*, NSC) et LDA diagonale ;
- régression ridge via la SVD (astuce en O(N^2 p)) et degrés de liberté ;
- lasso par descente de coordonnées et approximateur de signal *fused lasso* ;
- classifieurs n'utilisant que des produits scalaires (noyaux) ;
- composantes principales supervisées ;
- tests multiples : statistiques t, permutations, Bonferroni,
  Benjamini-Hochberg, estimation « plug-in » du FDR, SAM simplifié.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Centroïdes réduits
# ---------------------------------------------------------------------------

def soft_threshold(x, delta):
    """Seuillage doux ``sign(x) (|x| - delta)_+`` (éq. (18.5) du livre).

    Parameters
    ----------
    x : array_like
        Valeurs à seuiller.
    delta : float
        Seuil (positif ou nul).

    Returns
    -------
    ndarray
        Valeurs seuillées, de même forme que ``x``.
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - delta, 0.0)
