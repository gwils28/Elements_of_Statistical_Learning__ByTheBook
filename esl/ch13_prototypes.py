"""Méthodes à prototypes et plus proches voisins - chapitre 13 d'ESL.

Ce module regroupe des implémentations explicites (numpy) de :

* K-means (algorithme de Lloyd) et son utilisation comme classifieur
  (R prototypes par classe) ;
* LVQ1 (Learning Vector Quantization, algorithme 13.1 du livre) ;
* un classifieur par mélanges gaussiens à covariance commune, ajusté par EM
  (section 13.2.3) ;
* les k plus proches voisins (avec plusieurs valeurs de k d'un coup et
  validation croisée) ;
* la métrique adaptative DANN (éq. (13.8)-(13.9)) ;
* les vecteurs tangents d'images (rotation, translations, homothéties,
  cisaillement, épaisseur) et la distance tangente (section 13.3.3) ;
* quelques outils : rayon médian du 1-voisinage (éq. (13.7)), simulations
  du chapitre, condensation de Hart, multi-édition.
"""

import numpy as np
from scipy.special import gammaln, logsumexp


# ---------------------------------------------------------------------------
# Outils de base
# ---------------------------------------------------------------------------

def sq_dists(A, B):
    """Distances euclidiennes au carré entre les lignes de ``A`` et de ``B``.

    Parameters
    ----------
    A : ndarray (n, p)
    B : ndarray (m, p)

    Returns
    -------
    D : ndarray (n, m), ``D[i, j] = ||A_i - B_j||^2`` (tronqué à 0).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    D = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(D, 0.0)


def kfold_indices(n, n_folds, rng=None):
    """Découpe aléatoire de ``range(n)`` en ``n_folds`` blocs (liste d'indices)."""
    rng = np.random.RandomState(0) if rng is None else rng
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, n_folds)]
