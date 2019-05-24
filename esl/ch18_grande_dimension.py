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


def class_statistics(X, y):
    """Moyennes par classe et écart-type intra-classe poolé de chaque variable.

    Parameters
    ----------
    X : ndarray, forme (N, p)
        Données (une ligne par observation).
    y : array_like, forme (N,)
        Étiquettes de classe.

    Returns
    -------
    classes : ndarray, forme (K,)
        Étiquettes distinctes (triées).
    counts : ndarray, forme (K,)
        Effectifs ``N_k``.
    means : ndarray, forme (K, p)
        Centroïdes ``xbar_kj``.
    s : ndarray, forme (p,)
        Écart-type intra-classe poolé ``s_j`` (diviseur ``N - K``).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, inv = np.unique(y, return_inverse=True)
    K, N = len(classes), X.shape[0]
    counts = np.bincount(inv, minlength=K)
    means = np.zeros((K, X.shape[1]))
    np.add.at(means, inv, X)
    means /= counts[:, None]
    resid = X - means[inv]
    s = np.sqrt((resid ** 2).sum(axis=0) / (N - K))
    return classes, counts, means, s
