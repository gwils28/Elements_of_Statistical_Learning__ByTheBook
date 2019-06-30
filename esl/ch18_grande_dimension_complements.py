"""Chapitre 18 (p >> N) - outils pour les exercices 18.11 à 18.20.

Contenu :

* régression logistique (avec ou sans pénalité quadratique) et données séparables ;
* réduction « SVD » X = R V^T et validation croisée de la régression ridge ;
* régression logistique ridge à partir d'une matrice de noyau K ;
* 1-plus-proche-voisin pondéré par les distances (score continu) et courbe ROC ;
* ACP à noyau et projection d'un nouveau point ;
* tests multiples : t-statistiques, permutations, Bonferroni, Benjamini-Hochberg,
  estimation « plug-in » du FDR, estimation de pi0.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Régression logistique
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Fonction logistique 1 / (1 + exp(-z)), calculée de façon stable."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logistic_loglik(beta0, beta, X, y):
    """Log-vraisemblance du modèle logit Pr(Y=1|x) = beta0 + x^T beta.

    Parameters
    ----------
    beta0 : float
        Ordonnée à l'origine.
    beta : ndarray (p,)
        Coefficients.
    X : ndarray (N, p)
    y : ndarray (N,)  valeurs 0/1.

    Returns
    -------
    float
        sum_i [ y_i eta_i - log(1 + exp(eta_i)) ]  (toujours < 0).
    """
    eta = beta0 + X @ np.atleast_1d(beta)
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def logistic_gradient_ascent(X, y, n_iter=10000, lr=0.5, record=None):
    """Montée de gradient sur la log-vraisemblance logistique (sans pénalité).

    Sert à visualiser la divergence des coefficients quand les classes sont
    séparables (exercice 18.11).

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,) 0/1
    n_iter : int
    lr : float
        Pas (appliqué au gradient moyen).
    record : array-like d'entiers, optionnel
        Itérations auxquelles mémoriser (beta0, beta, loglik).

    Returns
    -------
    iters : ndarray
    path : ndarray (len(iters), p + 1)  - colonnes (beta0, beta_1..beta_p)
    loglik : ndarray (len(iters),)
    """
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    if record is None:
        record = np.unique(np.logspace(0, np.log10(n_iter), 60).astype(int))
    record = set(int(r) for r in record)
    b0, b = 0.0, np.zeros(p)
    iters, path, ll = [], [], []
    for t in range(1, n_iter + 1):
        r = y - sigmoid(b0 + X @ b)
        b0 += lr * r.mean()
        b += lr * (X.T @ r) / N
        if t in record:
            iters.append(t)
            path.append(np.r_[b0, b])
            ll.append(logistic_loglik(b0, b, X, y))
    return np.array(iters), np.array(path), np.array(ll)


def logistic_ridge(X, y, lam, max_iter=100, tol=1e-10):
    """Régression logistique avec pénalité quadratique (Newton-Raphson).

    Minimise  - sum_i log-vraisemblance_i + lam * ||beta||^2  ; l'ordonnée à
    l'origine beta0 n'est pas pénalisée (forme (18.16) du livre).

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,) 0/1
    lam : float > 0

    Returns
    -------
    beta0 : float
    beta : ndarray (p,)
    """
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    Xa = np.c_[np.ones(N), X]
    P = 2.0 * lam * np.eye(p + 1)
    P[0, 0] = 0.0
    w = np.zeros(p + 1)
    for _ in range(max_iter):
        pr = sigmoid(Xa @ w)
        grad = -Xa.T @ (y - pr) + P @ w
        H = Xa.T @ (Xa * (pr * (1 - pr))[:, None]) + P
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w[0], w[1:]


# ---------------------------------------------------------------------------
# Réduction SVD et régression ridge
# ---------------------------------------------------------------------------

def svd_reduce(X):
    """Écrit X = R V^T avec R = U D (N x N) et V (p x N) à colonnes orthonormées.

    Returns
    -------
    R : ndarray (N, min(N,p))
    V : ndarray (p, min(N,p))
    """
    U, d, Vt = np.linalg.svd(np.asarray(X, dtype=float), full_matrices=False)
    return U * d, Vt.T


def ridge_fit(X, y, lam):
    """Régression ridge avec ordonnée à l'origine non pénalisée.

    Minimise ||y - beta0 - X beta||^2 + lam ||beta||^2. Le calcul passe par la
    forme « duale » beta = Xc^T (Xc Xc^T + lam I)^{-1} yc (Xc, yc centrés), qui ne
    coûte que O(N^2 p).

    Returns
    -------
    beta0 : float
    beta : ndarray (p,)
    """
    X = np.asarray(X, dtype=float)
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    N = X.shape[0]
    beta = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(N), yc)
    return ym - xm @ beta, beta


def kfold_indices(N, n_folds, rng):
    """Attribue aléatoirement chacune des N observations à l'un des n_folds blocs."""
    return rng.permutation(np.arange(N) % n_folds)


def ridge_cv_errors(X, y, lams, folds):
    """Erreur quadratique de validation croisée de la ridge pour chaque lambda.

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,)
    lams : array-like
    folds : ndarray (N,) - numéro de bloc de chaque observation.

    Returns
    -------
    ndarray (len(lams),) - erreur quadratique moyenne de validation croisée.
    """
    lams = np.asarray(lams, dtype=float)
    err = np.zeros(len(lams))
    for f in np.unique(folds):
        te = folds == f
        for i, lam in enumerate(lams):
            b0, b = ridge_fit(X[~te], y[~te], lam)
            err[i] += np.sum((y[te] - b0 - X[te] @ b) ** 2)
    return err / len(y)


# ---------------------------------------------------------------------------
# Noyaux
# ---------------------------------------------------------------------------

def rbf_kernel(A, B, gamma):
    """Noyau gaussien K(a, b) = exp(-gamma ||a - b||^2)."""
    sq = (np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :]
          - 2 * A @ B.T)
    return np.exp(-gamma * np.maximum(sq, 0.0))


def kernel_logistic_ridge(K, y, lam, tol_eig=1e-10):
    """Régression logistique ridge à partir de la seule matrice K (exercice 18.13).

    On diagonalise K = U D^2 U^T, on ajuste la logistique ridge sur R = U D, puis
    alpha = U D^{-1} beta_R. Les valeurs propres numériquement nulles sont
    écartées (pseudo-inverse).

    Returns
    -------
    theta0 : float
    alpha : ndarray (N,) - la prédiction en x0 vaut theta0 + k0^T alpha.
    """
    ev, U = np.linalg.eigh(K)
    keep = ev > tol_eig * ev.max()
    U, d = U[:, keep], np.sqrt(ev[keep])
    R = U * d
    theta0, beta_R = logistic_ridge(R, y, lam)
    return theta0, U @ (beta_R / d)


# ---------------------------------------------------------------------------
# Plus proches voisins pondérés par les distances
# ---------------------------------------------------------------------------

def pairwise_dist(A, B):
    """Distances euclidiennes entre les lignes de A et celles de B."""
    sq = (np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :]
          - 2 * A @ B.T)
    return np.sqrt(np.maximum(sq, 0.0))


def dw_knn_score(Xtr, ytr, X0, k=1, priors=None, dim=1):
    """Score discriminant des k-ppv pondérés par les distances (exercice 18.14).

    delta(x0) = dim * log( d_-^{(k)}(x0) / d_+^{(k)}(x0) )
                + log( pi_+ N_- / (pi_- N_+) )   (si des a priori sont donnés),

    où d_+^{(k)} est la distance au k-ième plus proche voisin de la classe +1.
    Avec k = 1, dim = 1 et priors=None on retrouve (18.57).

    Parameters
    ----------
    Xtr : ndarray (N, p)
    ytr : ndarray (N,) à valeurs dans {-1, +1}
    X0 : ndarray (n0, p)
    k : int
    priors : tuple (pi_plus, pi_minus) ou None (a priori empiriques)
    dim : float
        Exposant (dimension effective) devant le log du rapport des distances.

    Returns
    -------
    ndarray (n0,) - score > 0 : classe +1.
    """
    D = pairwise_dist(X0, Xtr)
    dp = np.sort(D[:, ytr == 1], axis=1)[:, k - 1]
    dm = np.sort(D[:, ytr == -1], axis=1)[:, k - 1]
    s = dim * (np.log(dm) - np.log(dp))
    if priors is not None:
        Np, Nm = np.sum(ytr == 1), np.sum(ytr == -1)
        s = s + np.log(priors[0] * Nm / (priors[1] * Np))
    return s


def roc_curve(scores, labels):
    """Courbe ROC d'un score (grand score = classe positive).

    Parameters
    ----------
    scores : ndarray (n,)
    labels : ndarray (n,) booléen ou 0/1 (1 = positif).

    Returns
    -------
    fpr, tpr : ndarray - taux de faux positifs / vrais positifs (de 0 à 1).
    auc : float - aire sous la courbe (règle des trapèzes).
    """
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    tpr = np.r_[0.0, np.cumsum(lab) / lab.sum()]
    fpr = np.r_[0.0, np.cumsum(~lab) / (~lab).sum()]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# ACP à noyau
# ---------------------------------------------------------------------------

def kernel_pca(K, n_components=None, tol_eig=1e-10):
    """ACP à partir d'une matrice de produits scalaires non centrée K.

    On diagonalise (I - M) K (I - M) = U D^2 U^T avec M = 11^T / N.

    Returns
    -------
    U : ndarray (N, q)  vecteurs propres (valeurs propres décroissantes > 0)
    d : ndarray (q,)    racines des valeurs propres
    Z : ndarray (N, q)  variables principales Z = U D
    """
    N = K.shape[0]
    C = np.eye(N) - np.ones((N, N)) / N
    ev, U = np.linalg.eigh(C @ K @ C)
    order = np.argsort(ev)[::-1]
    ev, U = ev[order], U[:, order]
    keep = ev > tol_eig * ev[0]
    ev, U = ev[keep], U[:, keep]
    if n_components is not None:
        ev, U = ev[:n_components], U[:, :n_components]
    d = np.sqrt(ev)
    return U, d, U * d


def kernel_pca_project(K, U, d, k0):
    """Projection centrée de nouveaux points sur les axes principaux (18.58).

    z0 = D^{-1} U^T (I - M) [k0 - K 1 / N].

    Parameters
    ----------
    K : ndarray (N, N) - produits scalaires (non centrés) d'apprentissage
    U, d : sortie de :func:`kernel_pca`
    k0 : ndarray (N,) ou (N, n0) - produits scalaires <x0, x_i>

    Returns
    -------
    ndarray (q,) ou (q, n0)
    """
    N = K.shape[0]
    k0 = np.asarray(k0, dtype=float)
    kbar = K.mean(axis=1)
    v = k0 - (kbar if k0.ndim == 1 else kbar[:, None])
    v = v - v.mean(axis=0)            # application de (I - M)
    return (U.T @ v) / (d if k0.ndim == 1 else d[:, None])


# ---------------------------------------------------------------------------
# Tests multiples
# ---------------------------------------------------------------------------

def two_sample_t(X, group):
    """t-statistiques à deux échantillons (18.38)-(18.39), gène par gène.

    Parameters
    ----------
    X : ndarray (M, N) - un gène par ligne, un échantillon par colonne.
    group : ndarray (N,) booléen - True pour le groupe 2.
        Peut aussi être une matrice (K, N) de K étiquetages (permutations).

    Returns
    -------
    ndarray (M,) ou (M, K) - t_j = (xbar_2j - xbar_1j) / se_j.
    """
    X = np.asarray(X, dtype=float)
    G = np.atleast_2d(np.asarray(group, dtype=float))      # (K, N)
    N = X.shape[1]
    n2 = G.sum(axis=1)
    n1 = N - n2
    S = X @ G.T                                            # sommes groupe 2
    S2 = (X ** 2) @ G.T
    T, T2 = X.sum(axis=1)[:, None], (X ** 2).sum(axis=1)[:, None]
    m2, m1 = S / n2, (T - S) / n1
    ss = (S2 - n2 * m2 ** 2) + (T2 - S2 - n1 * m1 ** 2)
    sigma = np.sqrt(np.maximum(ss, 0) / (N - 2))
    t = (m2 - m1) / (sigma * np.sqrt(1 / n1 + 1 / n2))
    return t[:, 0] if np.ndim(group) == 1 else t


def permutation_t(X, group, K, rng):
    """t-statistiques sous K permutations aléatoires des étiquettes.

    Returns
    -------
    ndarray (M, K)
    """
    group = np.asarray(group, dtype=bool)
    G = np.array([rng.permutation(group) for _ in range(K)])
    return two_sample_t(X, G)


def pooled_pvalues(t, tperm):
    """p-valeurs de permutation « mutualisées » (18.41).

    p_j = (1 / (M K)) #{(j', k) : |t_kj'| >= |t_j|}  (convention >= ; sans ex
    aequo elle coïncide avec (18.41)).
    """
    null = np.sort(np.abs(np.ravel(tperm)))
    n = null.size
    return (n - np.searchsorted(null, np.abs(t), side="left")) / n


def bonferroni_reject(p, alpha):
    """Rejets de la méthode de Bonferroni : p_j < alpha / M."""
    p = np.asarray(p)
    return p < alpha / p.size


def benjamini_hochberg(p, alpha):
    """Procédure de Benjamini-Hochberg (algorithme 18.2).

    Returns
    -------
    L : int - L = max{ j : p_(j) < alpha j / M } (0 si aucun).
    reject : ndarray booléen - p_j <= p_(L).
    """
    p = np.asarray(p, dtype=float)
    M = p.size
    ps = np.sort(p)
    ok = np.nonzero(ps < alpha * np.arange(1, M + 1) / M)[0]
    if ok.size == 0:
        return 0, np.zeros(M, dtype=bool)
    L = int(ok[-1] + 1)
    return L, p <= ps[L - 1]


def plugin_fdr(t, tperm, cuts):
    """Estimation plug-in du FDR (algorithme 18.3), convention |t| >= C.

    Parameters
    ----------
    t : ndarray (M,)
    tperm : ndarray (M, K)
    cuts : array-like de seuils C

    Returns
    -------
    fdr : ndarray - E(V)_hat / R_obs (inf si R_obs = 0)
    R : ndarray - nombre de rejets observés
    EV : ndarray - nombre moyen de rejets par permutation
    """
    cuts = np.atleast_1d(np.asarray(cuts, dtype=float))
    a = np.sort(np.abs(t))
    null = np.sort(np.abs(np.ravel(tperm)))
    K = tperm.shape[1]
    R = a.size - np.searchsorted(a, cuts, side="left")
    EV = (null.size - np.searchsorted(null, cuts, side="left")) / K
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(R > 0, EV / np.maximum(R, 1), np.inf)
    return fdr, R, EV


def plugin_reject(t, tperm, alpha):
    """Rejets « plug-in » : on prend le plus petit seuil C = |t|_(l) observé tel
    que FDR_hat(C) < alpha, et on rejette |t_j| >= C."""
    a = np.sort(np.abs(t))[::-1]
    fdr, _, _ = plugin_fdr(t, tperm, a)
    ok = np.nonzero(fdr < alpha)[0]
    if ok.size == 0:
        return np.zeros(len(t), dtype=bool)
    return np.abs(t) >= a[ok[-1]]
