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


def principal_curve(X, n_iter=10, bandwidth_frac=0.08, n_grid=200):
    """Courbe principale par l'algorithme alterné (14.62) du livre.

    On part de la première composante principale, puis on alterne :
    (a) lissage de chaque coordonnée :math:`X_j` en fonction de
    l'abscisse curviligne :math:`\\hat\\lambda` (espérance conditionnelle) ;
    (b) projection de chaque point sur la courbe obtenue.

    Parameters
    ----------
    X : ndarray (N, p)
        Données.
    n_iter : int
        Nombre d'itérations (a)-(b).
    bandwidth_frac : float
        Largeur du noyau en fraction de la longueur totale de la courbe.
    n_grid : int
        Nombre de sommets utilisés pour représenter la courbe.

    Returns
    -------
    dict
        ``lam`` (abscisses des points), ``curve`` (sommets, (n_grid, p)),
        ``grid`` (abscisses des sommets), ``d2`` (distances au carré),
        ``history`` (somme des distances au carré à chaque itération).
    """
    X = np.asarray(X, float)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    lam = (X - mu) @ Vt[0]
    lam = lam - lam.min()
    history = []
    for _ in range(n_iter):
        grid = np.linspace(lam.min(), lam.max(), n_grid)
        h = bandwidth_frac * (lam.max() - lam.min())
        F = kernel_smoother(lam, X, grid, h)          # étape (a)
        lam, d2 = project_on_polyline(X, F)            # étape (b)
        history.append(d2.sum())
    grid = np.linspace(lam.min(), lam.max(), n_grid)
    F = kernel_smoother(lam, X, grid, bandwidth_frac * (lam.max() - lam.min()))
    lam, d2 = project_on_polyline(X, F)
    seg = np.sqrt((np.diff(F, axis=0) ** 2).sum(1))
    grid = np.concatenate([[0.0], np.cumsum(seg)])
    return {"lam": lam, "curve": F, "grid": grid, "d2": d2,
            "history": np.array(history)}


# ---------------------------------------------------------------------------
# Cartes auto-organisatrices (SOM) - exercice 14.13
# ---------------------------------------------------------------------------

def som_fit(X, grid_shape, n_epochs=30, alpha=(1.0, 0.0), radius=None,
            random_state=0):
    """Carte auto-organisatrice « en ligne » (mise à jour (14.46) du livre).

    Parameters
    ----------
    X : ndarray (N, p)
        Données.
    grid_shape : tuple
        ``(q1,)`` pour une chaîne de prototypes, ``(q1, q2)`` pour une grille.
    n_epochs : int
        Nombre de passages sur les données (une itération par observation).
    alpha : tuple
        Taux d'apprentissage initial et final (décroissance linéaire).
    radius : tuple ou None
        Rayon de voisinage initial et final (décroissance linéaire) ; par
        défaut ``(max(grid_shape)/2, 1)``.
    random_state : int
        Graine.

    Returns
    -------
    M : ndarray (K, p)
        Prototypes.
    coords : ndarray (K, d)
        Coordonnées entières des prototypes sur la grille.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, float)
    coords = np.array(np.meshgrid(*[np.arange(q) for q in grid_shape],
                                  indexing="ij")).reshape(len(grid_shape), -1).T
    coords = coords.astype(float)
    if radius is None:
        radius = (max(grid_shape) / 2.0, 1.0)
    # initialisation sur le plan (ou la droite) des composantes principales
    mu = X.mean(0)
    U, d, Vt = np.linalg.svd(X - mu, full_matrices=False)
    sd = d / np.sqrt(len(X))
    M = mu.copy()[None, :].repeat(len(coords), 0)
    for a in range(len(grid_shape)):
        c = coords[:, a] / max(grid_shape[a] - 1, 1) * 2 - 1     # dans [-1, 1]
        M = M + 2 * sd[a] * c[:, None] * Vt[a]
    n_total = n_epochs * len(X)
    step = 0
    for _ in range(n_epochs):
        for i in rng.permutation(len(X)):
            frac = step / max(n_total - 1, 1)
            a_t = alpha[0] + (alpha[1] - alpha[0]) * frac
            r_t = radius[0] + (radius[1] - radius[0]) * frac
            j = ((M - X[i]) ** 2).sum(1).argmin()
            nb = np.sqrt(((coords - coords[j]) ** 2).sum(1)) <= r_t
            M[nb] += a_t * (X[i] - M[nb])
            step += 1
    return M, coords


# ---------------------------------------------------------------------------
# Analyse factorielle - exercices 14.14 et 14.15
# ---------------------------------------------------------------------------

def factor_analysis_em(X, q, n_iter=500, tol=1e-10, psi_min=1e-8,
                       random_state=0):
    """Analyse factorielle gaussienne par l'algorithme EM.

    Modèle : :math:`X = \\mu + A S + \\varepsilon`, :math:`S\\sim N(0, I_q)`,
    :math:`\\varepsilon \\sim N(0, D_\\varepsilon)` avec
    :math:`D_\\varepsilon` diagonale, de sorte que
    :math:`\\Sigma = AA^T + D_\\varepsilon` (éq. (14.81)).

    Parameters
    ----------
    X : ndarray (N, p)
        Données.
    q : int
        Nombre de facteurs.
    n_iter : int
        Nombre maximal d'itérations.
    tol : float
        Arrêt quand la log-vraisemblance progresse de moins de ``tol``.
    psi_min : float
        Plancher pour les variances spécifiques.
    random_state : int
        Graine de l'initialisation.

    Returns
    -------
    A : ndarray (p, q)
        Matrice des saturations (loadings).
    psi : ndarray (p,)
        Variances spécifiques (diagonale de :math:`D_\\varepsilon`).
    loglik : ndarray
        Log-vraisemblance (moyenne par observation) au fil des itérations.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, float)
    N, p = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / N
    A = rng.randn(p, q) * np.sqrt(np.diag(S))[:, None] * 0.5
    psi = np.diag(S).copy()
    hist = []
    for _ in range(n_iter):
        Sigma = A @ A.T + np.diag(psi)
        sign, logdet = np.linalg.slogdet(Sigma)
        Si = np.linalg.inv(Sigma)
        ll = -0.5 * (p * np.log(2 * np.pi) + logdet + np.trace(Si @ S))
        hist.append(ll)
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol:
            break
        # étape E : moments de S | X
        B = A.T @ Si                         # (q, p) : E[S|x] = B x
        Ess = np.eye(q) - B @ A + B @ S @ B.T
        # étape M
        A = S @ B.T @ np.linalg.inv(Ess)
        psi = np.maximum(np.diag(S - A @ B @ S), psi_min)
    return A, psi, np.array(hist)


def factor_loglik(S, A, psi):
    """Log-vraisemblance gaussienne moyenne pour :math:`\\Sigma = AA^T + \\mathrm{diag}(\\psi)`.

    ``S`` est la covariance empirique (divisée par N).
    """
    p = len(S)
    Sigma = A @ A.T + np.diag(psi)
    logdet = np.linalg.slogdet(Sigma)[1]
    return -0.5 * (p * np.log(2 * np.pi) + logdet + np.trace(np.linalg.solve(Sigma, S)))


def loadings_given_psi(S, psi, q):
    """Saturations optimales à :math:`\\psi` fixé (solution de Lawley–Jöreskog).

    Si :math:`\\Psi^{-1/2}S\\Psi^{-1/2} = U\\Lambda U^T`, alors
    :math:`\\hat A = \\Psi^{1/2}U_q(\\Lambda_q - I)_+^{1/2}`.
    """
    r = 1.0 / np.sqrt(psi)
    ev, U = np.linalg.eigh(S * r[:, None] * r[None, :])
    ev, U = ev[::-1][:q], U[:, ::-1][:, :q]
    return (np.sqrt(psi)[:, None] * U) * np.sqrt(np.maximum(ev - 1.0, 0.0))


def factor_analysis_ml(X, q, n_starts=5, psi_min_frac=1e-8, random_state=0):
    """Analyse factorielle par maximum de vraisemblance « profilée ».

    Pour :math:`\\psi` fixé, les saturations optimales sont explicites
    (:func:`loadings_given_psi`) ; on maximise ensuite la vraisemblance en
    :math:`\\log\\psi` (L-BFGS-B, plusieurs départs). Plus fiable que l'EM
    lorsque certaines variances spécifiques tendent vers 0 (cas de Heywood).

    Returns
    -------
    A : ndarray (p, q)
    psi : ndarray (p,)
    loglik : float
    """
    from scipy.optimize import minimize
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / len(X)
    v = np.diag(S)
    lo = np.log(psi_min_frac * v)

    def nll(t):
        psi = np.exp(t)
        return -factor_loglik(S, loadings_given_psi(S, psi, q), psi)

    best = None
    for k in range(n_starts):
        t0 = np.log(v * (0.5 if k == 0 else rng.uniform(0.05, 1.0, len(v))))
        res = minimize(nll, t0, method="L-BFGS-B",
                       bounds=[(l, np.log(vi)) for l, vi in zip(lo, v)])
        if best is None or res.fun < best.fun:
            best = res
    psi = np.exp(best.x)
    return loadings_given_psi(S, psi, q), psi, -best.fun


def standardize_factor_model(A, psi):
    """Décomposition équivalente pour la matrice de corrélation (ex. 14.14).

    Si :math:`\\Sigma = AA^T + D_\\varepsilon` et
    :math:`\\Delta = \\mathrm{diag}(\\Sigma)^{-1/2}`, alors
    :math:`\\Delta\\Sigma\\Delta = (\\Delta A)(\\Delta A)^T + \\Delta D_\\varepsilon\\Delta`.

    Returns
    -------
    A_std : ndarray (p, q)
        Saturations redimensionnées :math:`\\Delta A`.
    psi_std : ndarray (p,)
        Variances spécifiques redimensionnées (« unicités »).
    """
    sigma2 = (A ** 2).sum(1) + psi
    delta = 1.0 / np.sqrt(sigma2)
    return A * delta[:, None], psi * delta ** 2


# ---------------------------------------------------------------------------
# ACP à noyau - exercices 14.16 et 14.17
# ---------------------------------------------------------------------------

def rbf_kernel(X, Y, c):
    """Noyau radial :math:`K(x, x') = \\exp(-\\|x-x'\\|^2/c)` (éq. (14.67))."""
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(2)
    return np.exp(-d2 / c)


def center_gram(K):
    """Double centrage :math:`(I-M)K(I-M)` avec :math:`M = 11^T/N` (éq. (14.65))."""
    return K - K.mean(0)[None, :] - K.mean(1)[:, None] + K.mean()


def kernel_pca(K, tol=1e-10):
    """ACP à noyau à partir d'une matrice de Gram ``K`` (non centrée).

    Returns
    -------
    dict
        ``U`` (vecteurs propres de :math:`\\tilde K` associés aux valeurs
        propres non nulles), ``d`` (les :math:`d_m`, racines des valeurs
        propres), ``Z = U D`` (composantes), ``alpha = U D^{-1}``.
    """
    Kt = center_gram(K)
    ev, U = np.linalg.eigh(Kt)
    order = np.argsort(ev)[::-1]
    ev, U = ev[order], U[:, order]
    keep = ev > tol * max(ev[0], 1.0)
    d = np.sqrt(ev[keep])
    U = U[:, keep]
    return {"U": U, "d": d, "Z": U * d, "alpha": U / d, "K": K}


def kernel_pca_transform(model, K_new):
    """Projette de nouvelles observations : :math:`z_{0m} = \\sum_j \\alpha_{jm}\\tilde K(x_0, x_j)`.

    Parameters
    ----------
    model : dict
        Sortie de :func:`kernel_pca`.
    K_new : ndarray (n0, N)
        :math:`K(x_0, x_j)` pour les nouvelles observations.
    """
    K = model["K"]
    Kc = (K_new - K_new.mean(1)[:, None] - K.mean(0)[None, :] + K.mean())
    return Kc @ model["alpha"]


# ---------------------------------------------------------------------------
# Densités gaussiennes inclinées pénalisées - exercice 14.18
# ---------------------------------------------------------------------------

def difference_matrix(L, order):
    """Matrice des différences finies d'ordre ``order`` (taille (L-order, L))."""
    D = sparse.eye(L, format="csr")
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D


def tilted_gaussian_fit(s, lam=1.0, order=3, L=150, center_penalty=False,
                        n_iter=100):
    """Estimation pénalisée :math:`f = \\varphi e^{g}` (forme discrétisée (14.94)).

    On discrétise l'axe en ``L`` cases de largeur :math:`\\Delta`, on compte
    les :math:`y^*_\\ell` (proportions d'observations par case) et on maximise
    par la méthode de Newton

    .. math:: \\sum_\\ell y^*_\\ell [\\log\\varphi(s^*_\\ell) + g_\\ell]
              - \\Delta\\sum_\\ell \\varphi(s^*_\\ell)e^{g_\\ell}
              - \\lambda\\, g^T P g,

    avec :math:`P = D_k^T D_k` (différences d'ordre ``order`` : version
    discrète de :math:`\\int (g^{(k)})^2`, à un facteur d'échelle près absorbé
    dans :math:`\\lambda`). Si ``center_penalty`` est vrai, on pénalise
    seulement l'écart des différences à leur moyenne
    (:math:`P = D_k^T(I - 11^T/n)D_k`), ce qui ajoute les quadratiques au
    noyau de la pénalité lorsque ``order=2``.

    Returns
    -------
    dict
        ``grid`` (centres des cases), ``delta``, ``y`` (proportions),
        ``g`` (fonction estimée), ``f`` (densité estimée sur la grille),
        ``grad`` (norme infinie du gradient final).
    """
    s = np.asarray(s, float)
    lo, hi = s.min() - 1.0, s.max() + 1.0
    edges = np.linspace(lo, hi, L + 1)
    grid = 0.5 * (edges[1:] + edges[:-1])
    delta = edges[1] - edges[0]
    y = np.histogram(s, edges)[0] / len(s)
    logphi = -0.5 * grid ** 2 - 0.5 * np.log(2 * np.pi)
    D = difference_matrix(L, order).toarray()
    if center_penalty:
        D = D - D.mean(0)[None, :]
    P = D.T @ D
    g = np.zeros(L)
    for _ in range(n_iter):
        mu = delta * np.exp(logphi + g)
        grad = y - mu - 2 * lam * (P @ g)
        step = np.linalg.solve(np.diag(mu) + 2 * lam * P, grad)
        g = g + step
        if np.max(np.abs(step)) < 1e-13:
            break
    mu = delta * np.exp(logphi + g)
    grad = y - mu - 2 * lam * (P @ g)
    f = np.exp(logphi + g)
    return {"grid": grid, "delta": delta, "y": y, "g": g, "f": f,
            "grad": np.max(np.abs(grad))}


# ---------------------------------------------------------------------------
# ICA : terme gaussien et point fixe - exercices 14.19 et 14.20
# ---------------------------------------------------------------------------

def gaussian_loglik_term(A, X):
    """:math:`\\sum_j\\sum_i \\log\\varphi(a_j^T x_i)` (premier terme de (14.92))."""
    S = X @ A
    return (-0.5 * S ** 2 - 0.5 * np.log(2 * np.pi)).sum()
