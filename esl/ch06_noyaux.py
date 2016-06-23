"""Méthodes de lissage par noyaux - chapitre 6 d'ESL.

Ce module regroupe les outils implémentés « à la main » dans les notebooks du
chapitre 6 :

* noyaux de lissage (Epanechnikov, tri-cube, gaussien) et poids
  :math:`K_\\lambda(x_0, x_i)` à fenêtre métrique ou à k plus proches voisins ;
* moyenne mobile des k-ppv, estimateur de Nadaraya-Watson ;
* régression polynomiale locale en dimension quelconque (monômes au choix),
  noyaux équivalents, matrice de lissage, critères LOO / GCV / :math:`C_p` ;
* régression logistique (globale et locale) par IRLS ;
* estimation de densité par noyau, classifieur par densités, Bayes naïf ;
* fonctions de base radiales, mélanges gaussiens (algorithme EM) ;
* analyse discriminante linéaire locale (exercice 6.12).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Noyaux
# ---------------------------------------------------------------------------


def epanechnikov(t):
    """Noyau d'Epanechnikov :math:`D(t) = \\tfrac34 (1 - t^2)` sur [-1, 1] (éq. 6.4).

    Parameters
    ----------
    t : array_like
        Distances réduites :math:`|x - x_0| / \\lambda`.

    Returns
    -------
    ndarray
        Valeurs de :math:`D(t)` (nulles hors de [-1, 1]). Intègre à 1.
    """
    t = np.asarray(t, dtype=float)
    return np.where(np.abs(t) <= 1, 0.75 * (1 - t ** 2), 0.0)


def tricube(t):
    """Noyau tri-cube :math:`D(t) = (1 - |t|^3)^3` sur [-1, 1] (éq. 6.6).

    Non normalisé : son intégrale vaut 81/70.
    """
    a = np.abs(np.asarray(t, dtype=float))
    return np.where(a <= 1, (1 - a ** 3) ** 3, 0.0)


def gaussian(t):
    """Noyau gaussien :math:`D(t) = \\varphi(t)` (densité de la loi N(0, 1))."""
    t = np.asarray(t, dtype=float)
    return np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)


KERNELS = {"epanechnikov": epanechnikov, "tricube": tricube,
           "gaussian": gaussian}


def _kernel(kernel):
    """Renvoie la fonction noyau associée à un nom (ou la fonction elle-même)."""
    if callable(kernel):
        return kernel
    try:
        return KERNELS[kernel]
    except KeyError:
        raise ValueError("Noyau inconnu : %r" % (kernel,))


def _as_2d(x):
    """Met un tableau de points sous la forme (n, p)."""
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def distances(x0, x, A=None):
    """Distances entre points cibles et points d'apprentissage.

    Parameters
    ----------
    x0 : array_like, shape (m,) ou (m, p)
        Points cibles.
    x : array_like, shape (n,) ou (n, p)
        Points d'apprentissage.
    A : ndarray (p, p), optional
        Matrice semi-définie positive de la métrique
        :math:`\\sqrt{(x - x_0)^T A (x - x_0)}` (cf. éq. 6.14). Par défaut,
        l'identité (distance euclidienne).

    Returns
    -------
    ndarray, shape (m, n)
    """
    x0, x = _as_2d(x0), _as_2d(x)
    diff = x[None, :, :] - x0[:, None, :]
    if A is None:
        d2 = np.einsum("mnp,mnp->mn", diff, diff)
    else:
        d2 = np.einsum("mnp,pq,mnq->mn", diff, np.asarray(A, float), diff)
    return np.sqrt(np.maximum(d2, 0.0))


def kernel_weights(x0, x, lam=None, k=None, kernel="epanechnikov", A=None):
    """Poids :math:`K_\\lambda(x_0, x_i) = D(\\|x_i - x_0\\| / h(x_0))` (éq. 6.5).

    Parameters
    ----------
    x0, x : array_like
        Points cibles (m,) / (m, p) et points d'apprentissage (n,) / (n, p).
    lam : float, optional
        Largeur métrique constante :math:`h(x_0) = \\lambda`.
    k : int, optional
        Fenêtre adaptative : :math:`h(x_0)` est la distance au k-ième plus
        proche voisin de :math:`x_0` (exclusif avec ``lam``).
    kernel : str ou callable
        ``"epanechnikov"``, ``"tricube"``, ``"gaussian"`` ou une fonction D.
    A : ndarray, optional
        Métrique (voir :func:`distances`).

    Returns
    -------
    ndarray, shape (m, n)
    """
    if (lam is None) == (k is None):
        raise ValueError("Préciser exactement un des deux paramètres lam ou k.")
    d = distances(x0, x, A)
    if k is not None:
        h = np.partition(d, k - 1, axis=1)[:, k - 1]
        h = np.where(h > 0, h, 1e-12)
    else:
        h = np.full(d.shape[0], float(lam))
    return _kernel(kernel)(d / h[:, None])


# ---------------------------------------------------------------------------
# Moyennes locales
# ---------------------------------------------------------------------------


def knn_mean(x0, x, y, k):
    """Moyenne des k plus proches voisins (éq. 6.1)."""
    d = distances(x0, x)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.asarray(y, float)[idx].mean(axis=1)


def nadaraya_watson(x0, x, y, lam=None, k=None, kernel="epanechnikov", A=None):
    """Moyenne pondérée de Nadaraya-Watson (éq. 6.2).

    Renvoie ``nan`` aux points cibles dont la fenêtre ne contient aucun point.
    """
    W = kernel_weights(x0, x, lam=lam, k=k, kernel=kernel, A=A)
    s = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, W @ np.asarray(y, float) / np.where(s > 0, s, 1), np.nan)


# ---------------------------------------------------------------------------
# Régression polynomiale locale
# ---------------------------------------------------------------------------


def monomials_of_degree(p, degree):
    """Liste des exposants (tuples de longueur p) de degré total <= ``degree``.

    Le monôme constant ``(0, ..., 0)`` est toujours en tête.
    """
    out = [(0,) * p]
    if degree >= 1:
        for j in range(p):
            e = [0] * p
            e[j] = 1
            out.append(tuple(e))
    if degree >= 2:
        for j in range(p):
            for l in range(j, p):
                e = [0] * p
                e[j] += 1
                e[l] += 1
                out.append(tuple(e))
    if degree >= 3:
        if p != 1:
            raise ValueError("Degré > 2 seulement disponible en dimension 1.")
        out += [(d,) for d in range(3, degree + 1)]
    return out


def _design(xc, monomials):
    """Matrice des monômes évalués en des coordonnées centrées ``xc`` (n, p)."""
    cols = [np.prod(xc ** np.asarray(e)[None, :], axis=1) for e in monomials]
    return np.column_stack(cols)


def local_weights_matrix(x0, x, lam=None, k=None, degree=1, kernel="tricube",
                         A=None, monomials=None, W=None):
    """Noyau équivalent de la régression polynomiale locale (éq. 6.8-6.9).

    Pour chaque point cible :math:`x_0`, on résout le problème de moindres
    carrés pondérés (6.12) avec des monômes centrés en :math:`x_0` ; la
    prédiction :math:`\\hat f(x_0)` est alors le coefficient du monôme
    constant, ce qui s'écrit :math:`\\hat f(x_0) = \\sum_i l_i(x_0) y_i`.

    Parameters
    ----------
    x0, x : array_like
        Points cibles et points d'apprentissage.
    lam, k, kernel, A :
        Voir :func:`kernel_weights`.
    degree : int
        Degré du polynôme local (0 : Nadaraya-Watson, 1 : linéaire local...).
    monomials : list of tuple, optional
        Liste explicite d'exposants (doit contenir le monôme constant) ; elle
        remplace ``degree`` (utile pour les modèles à coefficients variables).
    W : ndarray (m, n), optional
        Poids déjà calculés (remplacent ``lam``/``k``/``kernel``).

    Returns
    -------
    L : ndarray, shape (m, n)
        ``L[a, i]`` = :math:`l_i(x_{0,a})`. Lignes à ``nan`` si le système
        local est singulier (trop peu de points dans la fenêtre).
    """
    x0, x = _as_2d(x0), _as_2d(x)
    p = x.shape[1]
    if monomials is None:
        monomials = monomials_of_degree(p, degree)
    monomials = [tuple(e) for e in monomials]
    const = (0,) * p
    if const not in monomials:
        raise ValueError("Le monôme constant doit figurer dans la liste.")
    j0 = monomials.index(const)
    if W is None:
        W = kernel_weights(x0, x, lam=lam, k=k, kernel=kernel, A=A)
    L = np.full((x0.shape[0], x.shape[0]), np.nan)
    for a in range(x0.shape[0]):
        w = W[a]
        B = _design(x - x0[a], monomials)
        BtW = B.T * w
        M = BtW @ B
        if np.linalg.matrix_rank(M, tol=1e-10 * max(1.0, np.abs(M).max())) < M.shape[0]:
            continue
        L[a] = np.linalg.solve(M, BtW)[j0]
    return L


def local_regression(x0, x, y, lam=None, k=None, degree=1, kernel="tricube",
                     A=None, monomials=None):
    """Prédictions :math:`\\hat f(x_0)` de la régression polynomiale locale."""
    L = local_weights_matrix(x0, x, lam=lam, k=k, degree=degree, kernel=kernel,
                             A=A, monomials=monomials)
    return L @ np.asarray(y, float)


def smoother_matrix(x, lam=None, k=None, degree=1, kernel="tricube", A=None):
    """Matrice de lissage :math:`S_\\lambda` telle que :math:`\\hat{\\mathbf f} = S_\\lambda y`.

    ``S[i, j]`` = :math:`l_j(x_i)` (poids de :math:`y_j` dans l'ajustement en
    :math:`x_i`).
    """
    return local_weights_matrix(x, x, lam=lam, k=k, degree=degree,
                                kernel=kernel, A=A)


def effective_df(S):
    """Degrés de liberté effectifs :math:`\\mathrm{trace}(S_\\lambda)`."""
    return float(np.trace(S))


def loocv_error(S, y):
    """Erreur de validation croisée « leave-one-out » d'un lisseur linéaire.

    Utilise la formule :math:`y_i - \\hat f^{(-i)}(x_i) =
    (y_i - \\hat f(x_i)) / (1 - S_{ii})` (exercice 6.7).
    """
    y = np.asarray(y, float)
    r = (y - S @ y) / (1 - np.diag(S))
    return float(np.mean(r ** 2))


def gcv_error(S, y):
    """Validation croisée généralisée :math:`\\mathrm{ASR}/(1 - \\mathrm{tr}(S)/N)^2`."""
    y = np.asarray(y, float)
    n = len(y)
    return float(np.mean((y - S @ y) ** 2) / (1 - np.trace(S) / n) ** 2)


def cp_criterion(S, y, sigma2):
    """Critère :math:`C_\\lambda = \\mathrm{ASR} + 2 \\sigma^2 \\mathrm{tr}(S)/N` (éq. 6.35)."""
    y = np.asarray(y, float)
    n = len(y)
    return float(np.mean((y - S @ y) ** 2) + 2 * sigma2 * np.trace(S) / n)


# ---------------------------------------------------------------------------
# Régression logistique globale et locale
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def logistic_irls(X, y, w=None, n_iter=100, tol=1e-10, ridge=1e-10):
    """Régression logistique binaire pondérée par Newton-Raphson (IRLS).

    Maximise :math:`\\sum_i w_i [y_i \\eta_i - \\log(1 + e^{\\eta_i})]` avec
    :math:`\\eta_i = x_i^T \\beta` (la colonne de 1 doit être fournie).

    Parameters
    ----------
    X : ndarray (n, q)
    y : ndarray (n,) à valeurs dans {0, 1}
    w : ndarray (n,), optional
        Poids des observations (par ex. poids de noyau).
    ridge : float
        Petite pénalité quadratique qui stabilise les cas séparables.

    Returns
    -------
    beta : ndarray (q,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = _sigmoid(X @ beta)
        g = X.T @ (w * (y - p)) - ridge * beta
        H = (X.T * (w * p * (1 - p))) @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def local_logistic(x0, x, y, lam=None, k=None, degree=1, kernel="tricube",
                   n_iter=50, ridge=1e-8):
    """Régression logistique polynomiale locale (vraisemblance locale, §6.5).

    Pour chaque :math:`x_0`, on maximise la log-vraisemblance pondérée par
    :math:`K_\\lambda(x_0, x_i)` d'un modèle logistique dont le logit est un
    polynôme en :math:`x - x_0`.

    Returns
    -------
    prob : ndarray (m,)
        Probabilités estimées :math:`\\hat P(Y = 1 | x_0)`.
    se_logit : ndarray (m,)
        Écart-type « sandwich » estimé du logit en :math:`x_0`.
    """
    x0, x = _as_2d(x0), _as_2d(x)
    y = np.asarray(y, float)
    W = kernel_weights(x0, x, lam=lam, k=k, kernel=kernel)
    monos = monomials_of_degree(x.shape[1], degree)
    prob = np.full(x0.shape[0], np.nan)
    se = np.full(x0.shape[0], np.nan)
    for a in range(x0.shape[0]):
        w = W[a]
        keep = w > 0
        B = _design(x[keep] - x0[a], monos)
        beta = logistic_irls(B, y[keep], w[keep], n_iter=n_iter, ridge=ridge)
        p = _sigmoid(B @ beta)
        v = p * (1 - p)
        H = (B.T * (w[keep] * v)) @ B + ridge * np.eye(B.shape[1])
        G = (B.T * (w[keep] ** 2 * v)) @ B
        Hi = np.linalg.inv(H)
        cov = Hi @ G @ Hi
        prob[a] = _sigmoid(beta[0])
        se[a] = np.sqrt(max(cov[0, 0], 0.0))
    return prob, se
