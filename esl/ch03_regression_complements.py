"""Méthodes linéaires de régression : compléments pour les exercices 3.16 à 3.30.

Ce module contient des implémentations explicites (numpy) des méthodes du
chapitre 3 d'ESL utilisées dans les exercices 3.16 à 3.30 :

* estimateurs « orthonormaux » du tableau 3.4 (seuillages dur et doux) ;
* régression ridge (forme fermée, degrés de liberté effectifs) ;
* lasso par descente de coordonnées ;
* LAR et LAR-lasso (algorithmes 3.2 et 3.2a) ;
* régression sur composantes principales (PCR) et moindres carrés partiels
  (PLS, algorithme 3.3) ;
* sélection pas à pas ascendante (forward stepwise) et forward stagewise
  incrémental (algorithme 3.4) ;
* gradient conjugué ;
* analyse des corrélations canoniques (CCA) et régression de rang réduit ;
* augmentation des données pour l'elastic net ;
* outils de validation croisée (plis, règle « un écart-type »).

Sauf mention contraire, les fonctions de régression supposent des
prédicteurs **centrés** (et en général réduits) et une réponse **centrée** :
l'ordonnée à l'origine est alors estimée par la moyenne de la réponse
(exercice 3.5 du livre).
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------

def standardize(X, X_other=None, unit_norm=False):
    """Centre et réduit les colonnes de ``X``.

    Parameters
    ----------
    X : ndarray (N, p)
        Matrice d'apprentissage.
    X_other : ndarray (M, p), optionnel
        Autre matrice (par ex. de test) transformée avec les moyennes et
        écarts-types calculés sur ``X``.
    unit_norm : bool
        Si ``True``, chaque colonne est ramenée à une norme euclidienne 1
        (convention de l'algorithme LAR) ; sinon à une variance 1.

    Returns
    -------
    Xs : ndarray (N, p)
    Xo : ndarray (M, p) ou None
    mean, scale : ndarray (p,)
    """
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    Xc = X - mean
    if unit_norm:
        scale = np.linalg.norm(Xc, axis=0)
    else:
        scale = Xc.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Xs = Xc / scale
    Xo = None if X_other is None else (np.asarray(X_other, float) - mean) / scale
    return Xs, Xo, mean, scale


# ---------------------------------------------------------------------------
# Cas orthonormal (tableau 3.4)
# ---------------------------------------------------------------------------

def hard_threshold(beta, M):
    """Meilleur sous-ensemble de taille ``M`` dans le cas orthonormal.

    Conserve les ``M`` coefficients de plus grande valeur absolue et met
    les autres à zéro : ``beta_j * 1(|beta_j| >= |beta_(M)|)``.
    """
    beta = np.asarray(beta, float)
    out = np.zeros_like(beta)
    if M <= 0:
        return out
    keep = np.argsort(-np.abs(beta))[:M]
    out[keep] = beta[keep]
    return out


def soft_threshold(z, lam):
    """Seuillage doux : ``sign(z) * (|z| - lam)_+`` (solution lasso orthonormale)."""
    z = np.asarray(z, float)
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

def ridge(X, y, lam):
    """Coefficients ridge ``(X^T X + lam I)^{-1} X^T y`` (éq. (3.44)).

    ``X`` et ``y`` sont supposés centrés (pas d'ordonnée à l'origine).
    ``y`` peut être une matrice (N, K) : chaque colonne est traitée.
    """
    X = np.asarray(X, float)
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def ridge_svd_path(X, y, lams):
    """Chemin ridge calculé via la SVD ``X = U D V^T`` (éq. (3.47)).

    Returns
    -------
    betas : ndarray (p, len(lams))
    """
    U, d, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    uy = U.T @ y
    lams = np.atleast_1d(lams)
    coef = (d[:, None] / (d[:, None] ** 2 + lams[None, :])) * uy[:, None]
    return Vt.T @ coef


def ridge_df(d, lam):
    """Degrés de liberté effectifs ``sum d_j^2 / (d_j^2 + lam)`` (éq. (3.50))."""
    d = np.asarray(d, float)
    return np.sum(d ** 2 / (d ** 2 + lam))


def ridge_lambda_for_df(d, df):
    """Valeur de ``lam`` donnant ``df`` degrés de liberté effectifs.

    ``df`` doit être dans ``[0, rang]`` ; ``df = rang`` donne ``lam = 0`` et
    ``df = 0`` donne ``lam = inf``.
    """
    d = np.asarray(d, float)
    rank = np.sum(d > 1e-12 * d.max())
    if df >= rank - 1e-12:
        return 0.0
    if df <= 1e-12:
        return np.inf
    hi = 1.0
    while ridge_df(d, hi) > df:
        hi *= 10.0
    return brentq(lambda l: ridge_df(d, l) - df, 0.0, hi, xtol=1e-12 * hi)


# ---------------------------------------------------------------------------
# Lasso par descente de coordonnées
# ---------------------------------------------------------------------------

def lasso_cd(X, y, lam, beta0=None, tol=1e-10, max_iter=10000):
    """Lasso ``min 1/2 ||y - X b||^2 + lam ||b||_1`` par descente de coordonnées.

    Chaque coordonnée est mise à jour par seuillage doux (solution exacte du
    problème à une variable), cycliquement, jusqu'à stabilisation.

    Parameters
    ----------
    X : ndarray (N, p), y : ndarray (N,)
        Données supposées centrées.
    lam : float
        Paramètre de pénalité (convention avec le facteur 1/2, éq. (3.52)).
    beta0 : ndarray (p,), optionnel
        Point de départ (« démarrage à chaud »).

    Returns
    -------
    beta : ndarray (p,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    beta = np.zeros(p) if beta0 is None else np.array(beta0, float)
    col_sq = np.sum(X ** 2, axis=0)
    r = y - X @ beta
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            if col_sq[j] == 0:
                continue
            old = beta[j]
            zj = X[:, j] @ r + col_sq[j] * old
            new = soft_threshold(zj, lam) / col_sq[j]
            if new != old:
                r -= X[:, j] * (new - old)
                beta[j] = new
                max_change = max(max_change, abs(new - old) * np.sqrt(col_sq[j]))
        if max_change < tol:
            break
    return beta


# ---------------------------------------------------------------------------
# LAR et LAR-lasso
# ---------------------------------------------------------------------------

def lars(X, y, method="lar", max_steps=None, tol=1e-10):
    """Least Angle Regression (algorithme 3.2) et sa variante lasso (3.2a).

    À chaque étape, la direction est le coefficient des moindres carrés du
    résidu courant sur les variables actives (éq. (3.55)) ; le pas ``alpha``
    est le plus petit dans ``]0, 1]`` faisant entrer une nouvelle variable
    (« look-ahead », exercice 3.25) ou, pour ``method="lasso"``, faisant
    s'annuler un coefficient actif.

    Parameters
    ----------
    X : ndarray (N, p)
        Prédicteurs centrés (de préférence de norme 1).
    y : ndarray (N,)
        Réponse centrée.
    method : {"lar", "lasso"}
    max_steps : int, optionnel
        Nombre maximal d'étapes.

    Returns
    -------
    dict avec les clés
        ``coefs`` (p, K) : coefficients aux nœuds du chemin (K nœuds) ;
        ``lambdas`` (K,) : corrélation (produit scalaire) maximale absolue
        ``max_j |x_j^T r|`` à chaque nœud ;
        ``actives`` : liste des ensembles actifs après chaque étape ;
        ``alphas`` : pas ``alpha`` utilisés ;
        ``events`` : liste de couples (« entrée »/« sortie », indice).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, p = X.shape
    if max_steps is None:
        max_steps = 8 * p
    beta = np.zeros(p)
    r = y.copy()
    c = X.T @ r
    active = [int(np.argmax(np.abs(c)))]
    coefs = [beta.copy()]
    lambdas = [np.max(np.abs(c))]
    actives, alphas, events = [], [], [("entrée", active[0])]
    max_active = min(N - 1, p)
    for _ in range(max_steps):
        A = np.array(active)
        C = np.max(np.abs(c[A]))
        XA = X[:, A]
        delta = np.linalg.lstsq(XA, r, rcond=None)[0]
        u = XA @ delta
        a = X.T @ u
        alpha, enter, drop = 1.0, None, None
        if len(active) < max_active:
            inactive = np.setdiff1d(np.arange(p), A)
            with np.errstate(divide="ignore", invalid="ignore"):
                g1 = (C - c[inactive]) / (C - a[inactive])
                g2 = (C + c[inactive]) / (C + a[inactive])
            for k, v1, v2 in zip(inactive, g1, g2):
                for v in (v1, v2):
                    if np.isfinite(v) and tol < v < alpha:
                        alpha, enter = v, int(k)
        if method == "lasso":
            for idx, k in enumerate(A):
                if delta[idx] != 0:
                    t = -beta[k] / delta[idx]
                    if tol < t < alpha:
                        alpha, enter, drop = t, None, int(k)
        beta[A] += alpha * delta
        r -= alpha * u
        c = X.T @ r
        if drop is not None:
            beta[drop] = 0.0
            active.remove(drop)
            events.append(("sortie", drop))
        elif enter is not None:
            active.append(enter)
            events.append(("entrée", enter))
        coefs.append(beta.copy())
        lambdas.append(np.max(np.abs(c)))
        actives.append(list(active))
        alphas.append(alpha)
        if enter is None and drop is None:
            break
    return {"coefs": np.array(coefs).T, "lambdas": np.array(lambdas),
            "actives": actives, "alphas": np.array(alphas), "events": events}


def interpolate_path(coefs, s, norm_ref=None):
    """Interpole un chemin linéaire par morceaux en fraction de norme L1.

    Parameters
    ----------
    coefs : ndarray (p, K)
        Coefficients aux nœuds (norme L1 croissante, cas du lasso).
    s : array_like
        Fractions ``||beta||_1 / norm_ref`` voulues, dans ``[0, 1]``.
    norm_ref : float, optionnel
        Norme de référence (par défaut celle du dernier nœud, i.e. des
        moindres carrés).

    Returns
    -------
    ndarray (p, len(s))
    """
    l1 = np.abs(coefs).sum(axis=0)
    if norm_ref is None:
        norm_ref = l1[-1]
    l1 = np.maximum.accumulate(l1)
    target = np.atleast_1d(s) * norm_ref
    out = np.empty((coefs.shape[0], len(target)))
    for i, t in enumerate(target):
        k = np.searchsorted(l1, t)
        if k == 0:
            out[:, i] = coefs[:, 0]
        elif k >= len(l1):
            out[:, i] = coefs[:, -1]
        else:
            w = (t - l1[k - 1]) / max(l1[k] - l1[k - 1], 1e-300)
            out[:, i] = (1 - w) * coefs[:, k - 1] + w * coefs[:, k]
    return out


# ---------------------------------------------------------------------------
# Directions dérivées : PCR et PLS
# ---------------------------------------------------------------------------

def pcr(X, y, M=None):
    """Régression sur composantes principales (section 3.5.1).

    Returns
    -------
    betas : ndarray (p, M+1)
        Colonne ``m`` : coefficients (sur les colonnes de ``X``) de la PCR
        à ``m`` composantes ; la colonne 0 est nulle.
    """
    X = np.asarray(X, float)
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    p = X.shape[1]
    M = len(d) if M is None else M
    betas = np.zeros((p, M + 1))
    for m in range(1, M + 1):
        betas[:, m] = betas[:, m - 1] + Vt[m - 1] * (U[:, m - 1] @ y) / d[m - 1]
    return betas


def pls(X, y, M=None):
    """Moindres carrés partiels (algorithme 3.3).

    Parameters
    ----------
    X : ndarray (N, p)
        Prédicteurs centrés-réduits.
    y : ndarray (N,)
        Réponse (centrée ou non : seul ``y - ȳ`` intervient).
    M : int, optionnel
        Nombre de directions (par défaut ``p``).

    Returns
    -------
    dict avec ``betas`` (p, M+1) les coefficients sur les colonnes de ``X``
    après ``m = 0..M`` directions, ``Z`` (N, M) les directions ``z_m`` et
    ``theta`` (M,) les coefficients ``theta_m``.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float) - np.mean(y)
    N, p = X.shape
    M = p if M is None else M
    Xm = X.copy()
    R = np.eye(p)            # X^{(m)} = X R
    betas = np.zeros((p, M + 1))
    Z = np.zeros((N, M))
    theta = np.zeros(M)
    for m in range(M):
        phi = Xm.T @ y
        z = Xm @ phi
        zz = z @ z
        if zz < 1e-20:       # plus rien à expliquer
            betas[:, m + 1:] = betas[:, [m]]
            Z, theta = Z[:, :m], theta[:m]
            break
        w = R @ phi          # z = X w
        theta[m] = (z @ y) / zz
        betas[:, m + 1] = betas[:, m] + theta[m] * w
        Z[:, m] = z
        proj = (z @ Xm) / zz
        Xm = Xm - np.outer(z, proj)
        R = R - np.outer(w, proj)
    return {"betas": betas, "Z": Z, "theta": theta}


# ---------------------------------------------------------------------------
# Sélection ascendante et forward stagewise
# ---------------------------------------------------------------------------

def forward_stepwise(X, y, max_size=None):
    """Sélection pas à pas ascendante (section 3.3.2).

    À chaque étape on ajoute la variable qui diminue le plus la somme des
    carrés résiduelle, i.e. qui maximise ``<x_{j.A}, r>^2 / ||x_{j.A}||^2``
    où ``x_{j.A}`` est ``x_j`` ajusté (orthogonalisé) sur les variables
    actives (exercice 3.26).

    Returns
    -------
    order : list
        Variables dans l'ordre d'entrée.
    betas : ndarray (p, max_size+1)
        Coefficients des moindres carrés de chaque modèle emboîté.
    rss : ndarray (max_size+1,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, p = X.shape
    max_size = min(p, N - 1) if max_size is None else max_size
    Xr = X.copy()
    r = y.copy()
    order = []
    betas = np.zeros((p, max_size + 1))
    rss = [r @ r]
    for k in range(max_size):
        norms = np.sum(Xr ** 2, axis=0)
        score = np.where(norms > 1e-10 * np.max(norms), (Xr.T @ r) ** 2 / np.maximum(norms, 1e-300), -np.inf)
        score[order] = -np.inf
        j = int(np.argmax(score))
        order.append(j)
        q = Xr[:, j] / np.sqrt(norms[j])
        r = r - q * (q @ r)
        Xr = Xr - np.outer(q, q @ Xr)
        coef = np.linalg.lstsq(X[:, order], y, rcond=None)[0]
        betas[order, k + 1] = coef
        rss.append(r @ r)
    return order, betas, np.array(rss)
