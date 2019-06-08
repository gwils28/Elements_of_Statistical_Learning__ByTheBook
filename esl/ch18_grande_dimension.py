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


def ridge_df(X, lam):
    """Degrés de liberté effectifs ``sum_j d_j^2 / (d_j^2 + lam)`` (éq. (3.50))."""
    d = np.linalg.svd(np.asarray(X, dtype=float), compute_uv=False)
    lams = np.atleast_1d(lam).astype(float)
    df = np.array([(d ** 2 / (d ** 2 + l)).sum() for l in lams])
    return df[0] if np.ndim(lam) == 0 else df


# ---------------------------------------------------------------------------
# Lasso et fused lasso
# ---------------------------------------------------------------------------

def lasso_cd(X, y, lam, alpha=1.0, beta0=None, n_iter=1000, tol=1e-8):
    """Elastic net / lasso par descente de coordonnées cyclique (sans intercept).

    Minimise ``(1/2) ||y - X b||^2 + lam * sum_j (alpha |b_j| + (1-alpha)/2 b_j^2)``.

    Parameters
    ----------
    X : ndarray, forme (N, p)
    y : ndarray, forme (N,)
    lam : float
        Niveau de pénalité.
    alpha : float
        Mélange L1 / L2 (``alpha = 1`` : lasso).
    beta0 : ndarray ou None
        Point de départ (démarrage à chaud).
    n_iter : int
        Nombre maximal de balayages.
    tol : float
        Critère d'arrêt sur la variation maximale des coefficients.

    Returns
    -------
    ndarray, forme (p,)
    """
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    b = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)
    col2 = (X ** 2).sum(axis=0)
    r = y - X @ b

    def sweep(idx):
        maxch = 0.0
        for j in idx:
            if col2[j] == 0:
                continue
            old = b[j]
            rho = X[:, j] @ r + col2[j] * old
            new = soft_threshold(rho, lam * alpha) / (col2[j] + lam * (1 - alpha))
            if new != old:
                r[:] -= X[:, j] * (new - old)
                b[j] = new
                maxch = max(maxch, abs(new - old))
        return maxch

    # stratégie « ensemble actif » : on itère sur les coefficients non nuls
    # jusqu'à convergence, puis on vérifie par un balayage complet.
    for _ in range(n_iter):
        if sweep(range(p)) < tol:
            break
        active = np.flatnonzero(b)
        for _ in range(n_iter):
            if sweep(active) < tol:
                break
    return b


def lasso_path(X, y, lams, alpha=1.0, **kw):
    """Chemin de solutions de :func:`lasso_cd` (démarrages à chaud), forme (L, p)."""
    out, b = [], None
    for l in lams:
        b = lasso_cd(X, y, l, alpha=alpha, beta0=b, **kw)
        out.append(b.copy())
    return np.array(out)


def tv_denoise(y, lam, n_iter=50000, tol=1e-12):
    """Débruitage par variation totale 1D (fused lasso avec ``lambda_1 = 0``).

    Minimise ``(1/2) sum (y_i - b_i)^2 + lam sum |b_{i+1} - b_i|`` par gradient
    projeté accéléré (FISTA) sur le problème dual
    ``min_{|u|_inf <= lam} (1/2) ||y - D^T u||^2`` ; ``b = y - D^T u``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    def Dt(u):  # D^T u, avec (D b)_i = b_{i+1} - b_i
        out = np.zeros(n)
        out[:-1] -= u
        out[1:] += u
        return out

    u = np.zeros(n - 1)
    z, t = u.copy(), 1.0
    step = 0.25  # 1 / ||D D^T|| (majorant 4)
    for _ in range(n_iter):
        b = y - Dt(z)
        u_new = np.clip(z + step * np.diff(b), -lam, lam)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = u_new + (t - 1) / t_new * (u_new - u)
        if np.max(np.abs(u_new - u)) < tol:
            u = u_new
            break
        u, t = u_new, t_new
    return y - Dt(u)


def fused_lasso_signal(y, lam1, lam2, **kw):
    """Approximateur de signal fused lasso (éq. (18.24), sans ``beta_0``).

    Minimise ``(1/2) sum (y_i - b_i)^2 + lam1 sum |b_i| + lam2 sum |b_{i+1}-b_i|``.
    On utilise le fait (Friedman et al., 2007) que la solution s'obtient en
    seuillant doucement (au niveau ``lam1``) la solution à ``lam1 = 0``.

    Note : la constante 1/2 devant les moindres carrés revient à doubler les
    pénalités par rapport à l'écriture (18.24) du livre.
    """
    return soft_threshold(tv_denoise(y, lam2, **kw), lam1)


# ---------------------------------------------------------------------------
# Méthodes à noyaux (produits scalaires seulement)
# ---------------------------------------------------------------------------

def double_center(B):
    """Double centrage ``(I - M) B (I - M)`` avec ``M = 11^T / N`` (éq. (18.31))."""
    B = np.asarray(B, dtype=float)
    return B - B.mean(axis=0, keepdims=True) - B.mean(axis=1, keepdims=True) + B.mean()


def kernel_centroid_distances(K, K0, k00, y):
    """Distances carrées aux centroïdes calculées avec des produits scalaires (18.28).

    Parameters
    ----------
    K : ndarray, forme (N, N)
        Produits scalaires entre observations d'apprentissage.
    K0 : ndarray, forme (n, N)
        Produits scalaires entre points test et points d'apprentissage.
    k00 : ndarray, forme (n,)
        Normes carrées ``<x0, x0>`` des points test.
    y : array_like, forme (N,)
        Classes d'apprentissage.

    Returns
    -------
    classes : ndarray, forme (K,)
    D2 : ndarray, forme (n, K)
    """
    y = np.asarray(y)
    classes = np.unique(y)
    D2 = np.empty((K0.shape[0], len(classes)))
    for k, c in enumerate(classes):
        idx = y == c
        nk = idx.sum()
        D2[:, k] = k00 - 2 * K0[:, idx].sum(axis=1) / nk + K[np.ix_(idx, idx)].sum() / nk ** 2
    return classes, D2


def kernel_distances(K, K0, k00):
    """Distances carrées ``||x0 - x_i||^2`` à partir des produits scalaires (18.27)."""
    return k00[:, None] + np.diag(K)[None, :] - 2 * K0
