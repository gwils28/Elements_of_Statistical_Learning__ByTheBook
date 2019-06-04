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


class NearestShrunkenCentroids:
    """Classifieur des centroïdes réduits (section 18.2 du livre).

    Parameters
    ----------
    delta : float
        Quantité de rétrécissement ``Delta`` (``delta = 0`` donne la LDA diagonale
        avec l'écart-type régularisé ``s_j + s_0``).
    s0 : float ou "median"
        Constante ajoutée aux ``s_j`` ; par défaut la médiane des ``s_j``.
    mk : {"nsc", "lasso"}
        ``"nsc"`` : ``m_k^2 = 1/N_k - 1/N`` (éq. (18.4)) ;
        ``"lasso"`` : ``m_k^2 = 1/N_k`` (variante de l'exercice 18.2).
    priors : {"sample", "uniform"}
        Probabilités a priori ``pi_k`` utilisées dans le score (18.2).

    Attributes
    ----------
    classes_, counts_, centroids_, overall_mean_, s_, s0_, m_ : ndarray
        Quantités estimées à l'apprentissage.
    d_ : ndarray, forme (K, p)
        Écarts standardisés ``d_kj`` (éq. (18.4)).
    d_shrunk_ : ndarray, forme (K, p)
        Écarts seuillés ``d'_kj`` (éq. (18.5)).
    shrunken_centroids_ : ndarray, forme (K, p)
        Centroïdes réduits ``xbar'_kj`` (éq. (18.7)).
    """

    def __init__(self, delta=0.0, s0="median", mk="nsc", priors="sample"):
        self.delta = delta
        self.s0 = s0
        self.mk = mk
        self.priors = priors

    def fit(self, X, y):
        """Estime les centroïdes réduits ; retourne ``self``."""
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        classes, counts, means, s = class_statistics(X, y)
        self.classes_, self.counts_, self.centroids_, self.s_ = classes, counts, means, s
        self.overall_mean_ = X.mean(axis=0)
        self.s0_ = float(np.median(s)) if self.s0 == "median" else float(self.s0)
        if self.mk == "nsc":
            self.m_ = np.sqrt(1.0 / counts - 1.0 / N)
        elif self.mk == "lasso":
            self.m_ = np.sqrt(1.0 / counts)
        else:
            raise ValueError("mk doit valoir 'nsc' ou 'lasso'")
        scale = self.m_[:, None] * (s + self.s0_)[None, :]
        self.d_ = (means - self.overall_mean_) / scale
        self.set_delta(self.delta)
        if self.priors == "sample":
            self.log_priors_ = np.log(counts / N)
        else:
            self.log_priors_ = np.full(len(classes), -np.log(len(classes)))
        return self

    def set_delta(self, delta):
        """Change ``Delta`` sans réestimer les statistiques ; retourne ``self``."""
        self.delta = delta
        scale = self.m_[:, None] * (self.s_ + self.s0_)[None, :]
        self.d_shrunk_ = soft_threshold(self.d_, delta)
        self.shrunken_centroids_ = self.overall_mean_ + scale * self.d_shrunk_
        return self

    @property
    def active_features_(self):
        """Indices des variables ayant au moins un ``d'_kj`` non nul."""
        return np.flatnonzero(np.any(self.d_shrunk_ != 0, axis=0))

    def discriminant(self, X):
        """Scores discriminants ``delta_k(x)`` de l'éq. (18.2), forme (n, K)."""
        X = np.asarray(X, dtype=float)
        sd2 = (self.s_ + self.s0_) ** 2
        d2 = (((X[:, None, :] - self.shrunken_centroids_[None, :, :]) ** 2)
              / sd2).sum(axis=2)
        return -d2 + 2 * self.log_priors_

    def predict_proba(self, X):
        """Probabilités de classe de l'éq. (18.8)."""
        sc = 0.5 * self.discriminant(X)
        sc -= sc.max(axis=1, keepdims=True)
        e = np.exp(sc)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Classe de score maximal (règle (18.3))."""
        return self.classes_[np.argmax(self.discriminant(X), axis=1)]


def stratified_folds(y, n_folds, rng):
    """Numéro de pli (0..n_folds-1) pour chaque observation, stratifié par classe."""
    y = np.asarray(y)
    folds = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(len(idx))]
        start = rng.randint(n_folds)
        folds[idx] = (start + np.arange(len(idx))) % n_folds
    return folds


# ---------------------------------------------------------------------------
# Ridge via la SVD
# ---------------------------------------------------------------------------

def svd_reduce(X):
    """Réduction ``X = U D V^T = R V^T`` (éq. (18.12)-(18.13)).

    Parameters
    ----------
    X : ndarray, forme (N, p)

    Returns
    -------
    R : ndarray, forme (N, r)
        ``R = U D`` (lignes ``r_i``), avec ``r = min(N, p)``.
    V : ndarray, forme (p, r)
        Colonnes orthonormées.
    """
    U, d, Vt = np.linalg.svd(np.asarray(X, dtype=float), full_matrices=False)
    return U * d, Vt.T


def ridge_direct(X, y, lam):
    """Ridge sans intercept par la formule (18.14) : résout un système p x p."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def ridge_svd(X, y, lam):
    """Ridge sans intercept par la formule (18.15) : ``V (R^T R + lam I)^{-1} R^T y``.

    Coût en ``O(N^2 p)`` au lieu de ``O(p^3)``.

    Parameters
    ----------
    X : ndarray, forme (N, p)
    y : ndarray, forme (N,) ou (N, q)
    lam : float ou array_like
        Paramètre(s) de régularisation ; si c'est un tableau de longueur L,
        le résultat a une dimension supplémentaire en tête (L, p, ...).

    Returns
    -------
    ndarray
        Coefficients ``beta``.
    """
    U, d, Vt = np.linalg.svd(np.asarray(X, dtype=float), full_matrices=False)
    uty = U.T @ y
    lams = np.atleast_1d(lam).astype(float)
    out = []
    for l in lams:
        f = d / (d ** 2 + l)
        out.append(Vt.T @ (f.reshape((-1,) + (1,) * (uty.ndim - 1)) * uty))
    out = np.array(out)
    return out[0] if np.ndim(lam) == 0 else out
