"""Méthodes linéaires de régression - chapitre 3 d'ESL.

Implémentations « à la main » (numpy) des méthodes du chapitre :

* moindres carrés ordinaires par décomposition QR (Gram–Schmidt) et
  statistiques associées (erreurs standard, scores Z, test F) ;
* régression par orthogonalisations successives (algorithme 3.1) ;
* sélection de sous-ensembles : meilleur sous-ensemble, pas à pas ascendant
  (« forward stepwise ») et descendant, régression « stagewise » ;
* régression ridge (via la SVD) et degrés de liberté effectifs ;
* lasso par descente de coordonnées ;
* LAR (« least angle regression ») et sa modification lasso ;
* régression sur composantes principales (PCR) et moindres carrés partiels
  (PLS, algorithme 3.3) ;
* validation croisée à K plis et règle « un écart-type ».

Convention : sauf mention contraire, les fonctions d'ajustement reçoivent la
matrice ``X`` (N, p) **sans** colonne de uns ; l'ordonnée à l'origine est
traitée en centrant ``X`` et ``y`` puis en posant
``b0 = mean(y) - mean(X) @ beta``. Elles renvoient ``(b0, beta)``.
"""

from itertools import combinations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

PROSTATE_NAMES = ["lcavol", "lweight", "age", "lbph", "svi", "lcp",
                  "gleason", "pgg45"]


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def load_prostate_standardized():
    """Données prostate standardisées comme dans le livre.

    Les 8 prédicteurs sont centrés et réduits avec la moyenne et l'écart-type
    (``ddof=1``) calculés sur les 97 observations ; c'est ce choix qui
    reproduit exactement le tableau 3.2.

    Returns
    -------
    Xtr, ytr, Xte, yte : ndarray
        Apprentissage (67 obs.) et test (30 obs.).
    names : list of str
        Noms des prédicteurs.
    """
    from esl import datasets
    df = datasets.load_prostate()
    X = df[PROSTATE_NAMES].to_numpy(float)
    y = df["lpsa"].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    tr = df["train"].to_numpy(bool)
    return X[tr], y[tr], X[~tr], y[~tr], list(PROSTATE_NAMES)


def _center(X, y):
    """Centre ``X`` (colonnes) et ``y`` ; renvoie aussi les moyennes."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    xm, ym = X.mean(0), y.mean()
    return X - xm, y - ym, xm, ym


def add_intercept(X):
    """Ajoute une première colonne de uns à ``X``."""
    X = np.asarray(X, float)
    return np.column_stack([np.ones(X.shape[0]), X])


# ---------------------------------------------------------------------------
# Moindres carrés, QR, Gram–Schmidt
# ---------------------------------------------------------------------------

def gram_schmidt_qr(X):
    """Décomposition QR « réduite » par Gram–Schmidt (version modifiée).

    Parameters
    ----------
    X : ndarray (N, q), de rang plein q.

    Returns
    -------
    Q : ndarray (N, q)
        Colonnes orthonormées.
    R : ndarray (q, q)
        Triangulaire supérieure à diagonale > 0, avec ``X = Q R``.
    """
    X = np.asarray(X, float)
    N, q = X.shape
    Q = X.copy()
    R = np.zeros((q, q))
    for j in range(q):
        for k in range(j):
            R[k, j] = Q[:, k] @ Q[:, j]
            Q[:, j] -= R[k, j] * Q[:, k]
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] /= R[j, j]
    return Q, R


def back_substitution(R, b):
    """Résout ``R x = b`` pour ``R`` triangulaire supérieure (remontée)."""
    R = np.asarray(R, float)
    b = np.asarray(b, float)
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def ols_qr(X, y):
    """Moindres carrés via QR : ``beta = R^{-1} Q^T y`` (éq. 3.32).

    ``X`` doit déjà contenir la colonne de uns si l'on veut une ordonnée à
    l'origine.
    """
    Q, R = gram_schmidt_qr(X)
    return back_substitution(R, Q.T @ y)


def successive_orthogonalization(X):
    """Algorithme 3.1 : régression par orthogonalisations successives.

    Parameters
    ----------
    X : ndarray (N, p+1) dont la première colonne vaut 1.

    Returns
    -------
    Z : ndarray (N, p+1)
        Résidus orthogonaux ``z_0 = 1, z_1, ..., z_p`` (non normés).
    Gamma : ndarray (p+1, p+1)
        Triangulaire supérieure à diagonale unité, ``X = Z Gamma`` (éq. 3.30),
        ``Gamma[l, j] = <z_l, x_j> / <z_l, z_l>``.
    """
    X = np.asarray(X, float)
    q = X.shape[1]
    Z = np.zeros_like(X)
    G = np.eye(q)
    for j in range(q):
        z = X[:, j].copy()
        for l in range(j):
            G[l, j] = (Z[:, l] @ X[:, j]) / (Z[:, l] @ Z[:, l])
            z -= G[l, j] * Z[:, l]
        Z[:, j] = z
    return Z, G


def ols_fit(X, y, intercept=True):
    """Ajustement par moindres carrés et inférence classique (section 3.2).

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,)
    intercept : bool
        Ajouter une colonne de uns.

    Returns
    -------
    dict avec les clés ``coef`` (p+1 ou p), ``se`` (erreurs standard),
    ``z`` (scores Z, éq. 3.12), ``sigma2`` (estimateur sans biais, éq. 3.8),
    ``rss``, ``df_resid`` = N - (nombre de paramètres), ``fitted``,
    ``XtX_inv``.
    """
    A = add_intercept(X) if intercept else np.asarray(X, float)
    y = np.asarray(y, float)
    N, q = A.shape
    Q, R = gram_schmidt_qr(A)
    coef = back_substitution(R, Q.T @ y)
    fitted = A @ coef
    rss = float(np.sum((y - fitted) ** 2))
    df = N - q
    sigma2 = rss / df
    Rinv = np.linalg.solve(R, np.eye(q))
    XtX_inv = Rinv @ Rinv.T          # (R^T R)^{-1} = R^{-1} R^{-T}
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    return {"coef": coef, "se": se, "z": coef / se, "sigma2": sigma2,
            "rss": rss, "df_resid": df, "fitted": fitted, "XtX_inv": XtX_inv}


def f_test(rss0, rss1, p0, p1, N):
    """Statistique F (éq. 3.13) pour comparer deux modèles emboîtés.

    Parameters
    ----------
    rss0, rss1 : float
        RSS du petit modèle (p0 + 1 paramètres) et du gros (p1 + 1).
    p0, p1 : int
        Nombres de prédicteurs (hors constante), ``p1 > p0``.
    N : int

    Returns
    -------
    F, pvalue : float
    """
    F = ((rss0 - rss1) / (p1 - p0)) / (rss1 / (N - p1 - 1))
    return F, stats.f.sf(F, p1 - p0, N - p1 - 1)


def hat_matrix(X):
    """Matrice chapeau ``H = X (X^T X)^{-1} X^T`` (éq. 3.7), via QR."""
    Q, _ = gram_schmidt_qr(X)
    return Q @ Q.T


# ---------------------------------------------------------------------------
# Sélection de sous-ensembles
# ---------------------------------------------------------------------------

def _rss_subset(Xc, yc, subset):
    """RSS de la régression (avec constante) de y sur les colonnes ``subset``
    (``Xc``, ``yc`` centrés)."""
    if len(subset) == 0:
        return float(yc @ yc)
    A = Xc[:, list(subset)]
    b = np.linalg.lstsq(A, yc, rcond=None)[0]
    r = yc - A @ b
    return float(r @ r)


def all_subsets_rss(X, y):
    """RSS de tous les sous-ensembles de prédicteurs (figure 3.5).

    Returns
    -------
    list of (subset, rss) pour les 2^p sous-ensembles (constante toujours
    incluse).
    """
    Xc, yc, _, _ = _center(X, y)
    p = Xc.shape[1]
    out = []
    for k in range(p + 1):
        for S in combinations(range(p), k):
            out.append((S, _rss_subset(Xc, yc, S)))
    return out


def best_subset(X, y):
    """Meilleur sous-ensemble de chaque taille k = 0..p (recherche exhaustive).

    Returns
    -------
    subsets : list of tuple
        ``subsets[k]`` = meilleur sous-ensemble de taille k.
    rss : ndarray (p+1,)
    """
    p = np.asarray(X).shape[1]
    best = [None] * (p + 1)
    rss = np.full(p + 1, np.inf)
    for S, r in all_subsets_rss(X, y):
        k = len(S)
        if r < rss[k]:
            rss[k], best[k] = r, S
    return best, rss


def forward_stepwise(X, y, max_steps=None):
    """Sélection pas à pas ascendante, mise à jour efficace par QR (ex. 3.9).

    À chaque étape, on a une base orthonormée ``Q`` des variables retenues
    (constante comprise) et le résidu courant ``r``. Pour une candidate
    ``x_j``, sa partie orthogonale ``e_j = x_j - Q Q^T x_j`` fait baisser le
    RSS de ``(e_j^T r)^2 / ||e_j||^2`` : on retient la plus forte baisse.

    Returns
    -------
    order : list of int
        Ordre d'entrée des variables.
    rss : ndarray
        RSS après 0, 1, 2, ... variables.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, p = X.shape
    max_steps = p if max_steps is None else max_steps
    Q = np.ones((N, 1)) / np.sqrt(N)
    r = y - Q @ (Q.T @ y)
    order, rss = [], [float(r @ r)]
    remaining = list(range(p))
    for _ in range(max_steps):
        E = X[:, remaining] - Q @ (Q.T @ X[:, remaining])
        nrm2 = np.sum(E ** 2, axis=0)
        gain = (E.T @ r) ** 2 / nrm2
        i = int(np.argmax(gain))
        j = remaining.pop(i)
        q = E[:, i] / np.sqrt(nrm2[i])
        Q = np.column_stack([Q, q])
        r = r - q * (q @ r)
        order.append(j)
        rss.append(float(r @ r))
    return order, np.array(rss)


def backward_stepwise(X, y):
    """Sélection pas à pas descendante (ex. 3.10).

    À chaque étape on retire la variable de plus petit |Z| : la hausse du RSS
    causée par le retrait de la variable j vaut ``z_j^2 * sigma2_hat``.

    Returns
    -------
    removed : list of int
        Ordre de retrait.
    rss : ndarray
        RSS avec p, p-1, ..., 0 variables.
    """
    X = np.asarray(X, float)
    active = list(range(X.shape[1]))
    removed = []
    fit = ols_fit(X[:, active], y)
    rss = [fit["rss"]]
    while active:
        z = fit["z"][1:]
        i = int(np.argmin(np.abs(z)))
        removed.append(active.pop(i))
        if active:
            fit = ols_fit(X[:, active], y)
            rss.append(fit["rss"])
        else:
            yc = np.asarray(y) - np.mean(y)
            rss.append(float(yc @ yc))
    return removed, np.array(rss)


def subset_fit(X, y, subset):
    """Moindres carrés sur un sous-ensemble ; renvoie ``(b0, beta)`` complet."""
    Xc, yc, xm, ym = _center(X, y)
    beta = np.zeros(Xc.shape[1])
    if len(subset):
        S = list(subset)
        beta[S] = np.linalg.lstsq(Xc[:, S], yc, rcond=None)[0]
    return ym - xm @ beta, beta


def forward_stagewise(X, y, eps=None, n_steps=1000, tol=1e-10):
    """Régression « stagewise » ascendante (sections 3.3.3 et 3.8.1).

    Les prédicteurs sont centrés (et supposés de même échelle). À chaque
    étape on choisit la variable la plus corrélée au résidu ``r`` puis :

    * ``eps=None`` (FS « complet », §3.3.3) : on ajoute à son coefficient le
      coefficient de la régression simple de ``r`` sur ``x_j`` ;
    * ``eps>0`` (FS_eps, algorithme 3.4) : on ajoute ``eps * sign(<x_j, r>)``.

    Returns
    -------
    path : ndarray (n_steps+1, p)
        Coefficients après chaque étape (la première ligne est nulle).
    """
    Xc, yc, _, _ = _center(X, y)
    N, p = Xc.shape
    nrm2 = np.sum(Xc ** 2, axis=0)
    beta = np.zeros(p)
    r = yc.copy()
    path = [beta.copy()]
    for _ in range(n_steps):
        c = Xc.T @ r
        corr = c / np.sqrt(nrm2)
        j = int(np.argmax(np.abs(corr)))
        if abs(corr[j]) < tol:
            break
        delta = c[j] / nrm2[j] if eps is None else eps * np.sign(c[j])
        beta[j] += delta
        r -= delta * Xc[:, j]
        path.append(beta.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

def ridge_fit(X, y, lam):
    """Régression ridge (éq. 3.41–3.44) calculée par la SVD de X centrée.

    ``beta = V diag(d_j / (d_j^2 + lam)) U^T y`` (cf. éq. 3.47).

    Returns
    -------
    b0, beta
    """
    Xc, yc, xm, ym = _center(X, y)
    U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
    beta = Vt.T @ (d / (d ** 2 + lam) * (U.T @ yc))
    return ym - xm @ beta, beta


def ridge_df(d, lam):
    """Degrés de liberté effectifs ``sum d_j^2 / (d_j^2 + lam)`` (éq. 3.50)."""
    d = np.asarray(d, float)
    return float(np.sum(d ** 2 / (d ** 2 + lam)))


def ridge_lambda_for_df(X, df):
    """Valeur de lambda telle que df(lambda) = ``df`` (0 < df < p)."""
    Xc = np.asarray(X, float) - np.mean(X, axis=0)
    d = np.linalg.svd(Xc, compute_uv=False)
    p = len(d)
    if df >= p:
        return 0.0
    if df <= 0:
        return np.inf
    return brentq(lambda l: ridge_df(d, l) - df, 1e-12, 1e12, xtol=1e-12)


# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------

def soft_threshold(z, gamma):
    """Seuillage doux ``sign(z) (|z| - gamma)_+`` (tableau 3.4, éq. 3.84)."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def lasso_cd(X, y, lam, beta0=None, max_iter=10000, tol=1e-10):
    """Lasso par descente de coordonnées (section 3.8.6).

    Minimise ``1/2 ||y - b0 - X beta||^2 + lam * ||beta||_1`` (éq. 3.52 à un
    facteur 1/2 près). La mise à jour cyclique de la coordonnée j est
    ``beta_j <- S(<x_j, r_(j)>, lam) / ||x_j||^2`` où ``r_(j)`` est le résidu
    partiel sans la variable j.

    Parameters
    ----------
    beta0 : ndarray, optionnel
        Point de départ (« démarrage à chaud »).

    Returns
    -------
    b0, beta
    """
    Xc, yc, xm, ym = _center(X, y)
    p = Xc.shape[1]
    beta = np.zeros(p) if beta0 is None else np.array(beta0, float)
    nrm2 = np.sum(Xc ** 2, axis=0)
    r = yc - Xc @ beta
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            old = beta[j]
            rho = Xc[:, j] @ r + nrm2[j] * old
            new = soft_threshold(rho, lam) / nrm2[j]
            if new != old:
                r -= (new - old) * Xc[:, j]
                beta[j] = new
                max_change = max(max_change, abs(new - old))
        if max_change < tol:
            break
    return ym - xm @ beta, beta
