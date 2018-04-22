"""Chapitre 14 (exercices 14.13 à 14.25) : outils complémentaires.

Courbes principales, cartes auto-organisatrices, analyse factorielle,
ACP à noyau, densités gaussiennes « inclinées » (ICA directe), algorithme
du point fixe (FastICA), laplacien de graphe, PageRank et factorisation
en matrices non négatives (NMF).

Toutes les fonctions sont écrites avec numpy/scipy uniquement, dans un but
pédagogique (lisibilité plutôt que performance).
"""

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Courbes principales (Hastie et Stuetzle) - exercice 14.13
# ---------------------------------------------------------------------------

def kernel_smoother(t, y, t_eval, bandwidth):
    """Régression linéaire locale à noyau gaussien.

    Parameters
    ----------
    t : ndarray (N,)
        Abscisses des observations.
    y : ndarray (N,) ou (N, p)
        Réponses (une ou plusieurs colonnes lissées simultanément).
    t_eval : ndarray (M,)
        Points où évaluer le lissage.
    bandwidth : float
        Écart-type du noyau gaussien.

    Returns
    -------
    ndarray (M,) ou (M, p)
        Valeurs lissées.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    one_dim = y.ndim == 1
    Y = y[:, None] if one_dim else y
    out = np.empty((len(t_eval), Y.shape[1]))
    for m, t0 in enumerate(np.asarray(t_eval, float)):
        w = np.exp(-0.5 * ((t - t0) / bandwidth) ** 2)
        B = np.column_stack([np.ones_like(t), t - t0])
        BW = B.T * w
        coef = np.linalg.solve(BW @ B + 1e-10 * np.eye(2), BW @ Y)
        out[m] = coef[0]
    return out[:, 0] if one_dim else out


def project_on_polyline(X, F):
    """Projette chaque ligne de ``X`` sur la ligne brisée de sommets ``F``.

    Parameters
    ----------
    X : ndarray (N, p)
        Points à projeter.
    F : ndarray (M, p)
        Sommets successifs de la courbe.

    Returns
    -------
    lam : ndarray (N,)
        Abscisse curviligne (longueur d'arc depuis ``F[0]``) du projeté.
    dist2 : ndarray (N,)
        Carré de la distance de chaque point à la courbe.
    """
    seg = np.diff(F, axis=0)                       # (M-1, p)
    seg_len2 = np.maximum((seg ** 2).sum(1), 1e-300)
    cum = np.concatenate([[0.0], np.cumsum(np.sqrt(seg_len2))])
    D = X[:, None, :] - F[None, :-1, :]           # (N, M-1, p)
    u = np.clip((D * seg[None]).sum(2) / seg_len2, 0.0, 1.0)
    P = F[None, :-1, :] + u[..., None] * seg[None]
    d2 = ((X[:, None, :] - P) ** 2).sum(2)
    k = d2.argmin(1)
    idx = np.arange(len(X))
    lam = cum[k] + u[idx, k] * np.sqrt(seg_len2[k])
    return lam, d2[idx, k]
