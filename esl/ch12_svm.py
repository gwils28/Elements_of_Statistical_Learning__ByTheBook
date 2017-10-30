"""Chapitre 12 : machines à vecteurs de support et discriminants flexibles.

Ce module regroupe les implémentations « à la main » utilisées dans les
notebooks du chapitre 12 :

* noyaux (linéaire, polynomial, radial) ;
* un solveur SMO (*Sequential Minimal Optimization*) pour le problème dual
  quadratique commun aux SVM de classification et de régression ;
* les classes :class:`SVMClassifier` et :class:`SVMRegressor` ;
* les fonctions de perte de la figure 12.4 et de la figure 12.8 ;
* l'analyse discriminante linéaire (LDA), flexible (FDA, par *optimal
  scoring*), pénalisée (PDA) et par mélanges (MDA, algorithme EM) ;
* des simulateurs (mélange gaussien des chapitres 2 et 12, *waveform*).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Noyaux
# ---------------------------------------------------------------------------


def linear_kernel(X, Z):
    """Noyau linéaire :math:`K(x, z) = \\langle x, z\\rangle`.

    Parameters
    ----------
    X : ndarray (n, p)
    Z : ndarray (m, p)

    Returns
    -------
    ndarray (n, m)
    """
    return np.asarray(X, float) @ np.asarray(Z, float).T


def polynomial_kernel(X, Z, degree=2, coef0=1.0, gamma=1.0):
    """Noyau polynomial :math:`K(x, z) = (\\text{coef0} + \\gamma\\langle x, z\\rangle)^d`.

    Avec ``coef0=1`` et ``gamma=1`` on retrouve le noyau (12.22) du livre.
    """
    return (coef0 + gamma * linear_kernel(X, Z)) ** degree


def rbf_kernel(X, Z, gamma=1.0):
    """Noyau radial (gaussien) :math:`K(x, z) = \\exp(-\\gamma\\|x - z\\|^2)`."""
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    d2 = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * X @ Z.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def make_kernel(kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
    """Renvoie une fonction ``K(X, Z)`` correspondant au noyau demandé.

    Parameters
    ----------
    kernel : {"linear", "poly", "rbf"} ou fonction
        Nom du noyau, ou directement une fonction ``K(X, Z)``.
    gamma, degree, coef0 : float
        Paramètres du noyau (voir :func:`polynomial_kernel`, :func:`rbf_kernel`).
    """
    if callable(kernel):
        return kernel
    if kernel == "linear":
        return linear_kernel
    if kernel == "poly":
        return lambda X, Z: polynomial_kernel(X, Z, degree, coef0, gamma)
    if kernel == "rbf":
        return lambda X, Z: rbf_kernel(X, Z, gamma)
    raise ValueError("noyau inconnu : %r" % kernel)


def poly2_features(X):
    """Plongement explicite associé au noyau :math:`(1 + \\langle x, z\\rangle)^2`.

    Pour :math:`x \\in \\mathbb{R}^p`, renvoie le vecteur
    :math:`(1, \\sqrt2 x_j, x_j^2, \\sqrt2 x_j x_k \\ (j<k))` de sorte que
    :math:`\\langle h(x), h(z)\\rangle = (1 + \\langle x, z\\rangle)^2`
    (éq. (12.23) du livre pour :math:`p = 2`).
    """
    X = np.asarray(X, float)
    n, p = X.shape
    cols = [np.ones(n)]
    cols += [np.sqrt(2) * X[:, j] for j in range(p)]
    cols += [X[:, j] ** 2 for j in range(p)]
    cols += [np.sqrt(2) * X[:, j] * X[:, k] for j in range(p) for k in range(j + 1, p)]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Solveur SMO
# ---------------------------------------------------------------------------


def smo_solve(Q, p, s, C, tol=1e-3, max_iter=200000):
    """Résout le problème quadratique dual « générique » des SVM par SMO.

    Le problème est

    .. math::
        \\min_a \\tfrac12 a^T Q a + p^T a
        \\quad\\text{s.c.}\\quad s^T a = 0,\\quad 0 \\le a_t \\le C,

    avec :math:`s_t \\in \\{-1, +1\\}`. À chaque itération on choisit une
    paire :math:`(i, j)` : :math:`i` viole le plus les conditions KKT, puis
    :math:`j` est la variable qui, associée à :math:`i`, fait le plus
    décroître l'objectif (règle « du second ordre » de Fan, Chen et Lin,
    utilisée par LIBSVM). On optimise alors exactement l'objectif selon ces
    deux coordonnées, en respectant la contrainte d'égalité et les bornes.

    Parameters
    ----------
    Q : ndarray (n, n)
        Matrice symétrique semi-définie positive.
    p : ndarray (n,)
    s : ndarray (n,)
        Signes :math:`\\pm 1`.
    C : float ou ndarray (n,)
        Borne supérieure des variables.
    tol : float
        Tolérance sur l'écart KKT :math:`m(a) - M(a)`.
    max_iter : int
        Nombre maximal d'itérations.

    Returns
    -------
    a : ndarray (n,)
        Solution duale.
    b : float
        Constante :math:`\\beta_0` telle que, pour toute variable « libre »
        (:math:`0 < a_t < C`), :math:`(Qa + p)_t + s_t b = 0`.
    n_iter : int
        Nombre d'itérations effectuées.
    """
    Q = np.asarray(Q, float)
    p = np.asarray(p, float)
    s = np.asarray(s, float)
    n = len(p)
    Cv = np.broadcast_to(np.asarray(C, float), (n,)).copy()
    a = np.zeros(n)
    G = p.copy()  # gradient Q a + p (a = 0 au départ)
    diag = np.diag(Q).copy()
    pos = s > 0
    eps_b = 1e-12
    it = 0
    for it in range(1, max_iter + 1):
        up = (pos & (a < Cv - eps_b)) | (~pos & (a > eps_b))
        low = (pos & (a > eps_b)) | (~pos & (a < Cv - eps_b))
        v = -s * G
        vu = np.where(up, v, -np.inf)
        vl = np.where(low, v, np.inf)
        i = int(np.argmax(vu))
        if vu[i] - vl.min() < tol:
            break
        # choix de j au second ordre (Fan, Chen et Lin, 2005) : on prend le j
        # qui fait le plus décroître l'objectif parmi les candidats violants
        bij = vu[i] - vl
        aij = np.maximum(diag[i] + diag - 2 * s[i] * s * Q[i], 1e-12)
        gain = np.where(low & (bij > 0), bij ** 2 / aij, -np.inf)
        j = int(np.argmax(gain))
        eta = aij[j]
        t = bij[j] / eta
        # bornes : a_i <- a_i + s_i t, a_j <- a_j - s_j t, avec t > 0
        ti = Cv[i] - a[i] if s[i] > 0 else a[i]
        tj = a[j] if s[j] > 0 else Cv[j] - a[j]
        t = min(t, ti, tj)
        a[i] += s[i] * t
        a[j] -= s[j] * t
        G += t * (s[i] * Q[:, i] - s[j] * Q[:, j])
    a = np.clip(a, 0, Cv)
    v = -s * G
    thr = 1e-5 * np.minimum(Cv, max(a.max(), 1e-300))
    free = (a > thr) & (a < Cv * (1 - 1e-5))
    if free.any():
        b = float(np.mean(v[free]))
    else:
        up = (pos & (a < Cv)) | (~pos & (a > 0))
        low = (pos & (a > 0)) | (~pos & (a < Cv))
        b = 0.5 * (v[up].max() + v[low].min())
    return a, b, it


def qp_interior_point(Q, p, s, C, tol=1e-10, max_iter=100):
    """Même problème que :func:`smo_solve`, résolu par points intérieurs.

    On applique la méthode de Newton aux conditions KKT « perturbées »
    (:math:`a_t z_t = \\mu`, :math:`(C - a_t) w_t = \\mu`) en faisant
    décroître :math:`\\mu` vers 0. Chaque itération résout un système
    linéaire de taille :math:`n + 1` : la méthode est donc réservée aux
    problèmes de taille modérée (quelques centaines de points), mais elle est
    très précise et insensible au mauvais conditionnement (grands :math:`C`).

    Returns
    -------
    a : ndarray (n,)
    b : float
        Multiplicateur de la contrainte d'égalité (égal à :math:`\\beta_0`).
    n_iter : int
    """
    Q = np.asarray(Q, float)
    p = np.asarray(p, float)
    s = np.asarray(s, float)
    n = len(p)
    Cv = np.broadcast_to(np.asarray(C, float), (n,)).copy()
    a = Cv / 2
    z = np.ones(n)
    w = np.ones(n)
    nu = 0.0
    scale = 1 + np.abs(Q).max() * Cv.max() + np.abs(p).max()
    it = 0
    for it in range(1, max_iter + 1):
        u = Cv - a
        mu = (a @ z + u @ w) / (2 * n)
        rd = Q @ a + p + s * nu - z + w
        re = s @ a
        if mu < tol and np.abs(rd).max() < 1e-8 * scale and abs(re) < 1e-8 * Cv.max() * n:
            break
        sig = 0.1
        D = z / a + w / u
        rhs = -rd + (sig * mu / a - z) - (sig * mu / u - w)
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = Q + np.diag(D)
        M[:n, n] = s
        M[n, :n] = s
        sol = np.linalg.solve(M, np.r_[rhs, -re])
        da, dnu = sol[:n], sol[n]
        dz = (sig * mu - a * z - z * da) / a
        dw = (sig * mu - u * w + w * da) / u
        # pas maximal gardant a dans ]0, C[ et z, w > 0
        steps = [1.0]
        for v, dv in ((a, da), (u, -da), (z, dz), (w, dw)):
            neg = dv < 0
            if neg.any():
                steps.append(np.min(-v[neg] / dv[neg]))
        t = min(1.0, 0.99 * min(steps))
        a, z, w, nu = a + t * da, z + t * dz, w + t * dw, nu + t * dnu
    return np.clip(a, 0, Cv), float(nu), it
