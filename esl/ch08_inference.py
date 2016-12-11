"""Chapitre 8 - Inférence sur les modèles et moyennage de modèles.

Implémentations « à la main » utilisées dans les notebooks du chapitre 8 :

* bases de B-splines cubiques, matrice de pénalité :math:`\\Omega` et
  splines de lissage (sections 8.2 et 5.4 du livre) ;
* ajustement par moindres carrés / maximum de vraisemblance, bandes
  d'erreur standard, bootstrap non paramétrique et paramétrique ;
* loi a posteriori gaussienne des coefficients (éq. (8.27)-(8.28)) ;
* algorithme EM et échantillonneur de Gibbs pour un mélange de deux
  gaussiennes (algorithmes 8.1 et 8.4) ;
* bagging, souches de décision (« stumps »), AdaBoost, bumping et stacking
  (sections 8.7 à 8.9).
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import BSpline

#: Les 20 données fictives du tableau 8.1 du livre (exemple de mélange).
MIXTURE_DATA = np.array([
    -0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53,
    0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22])


# ---------------------------------------------------------------------------
# B-splines et splines de lissage
# ---------------------------------------------------------------------------

def bspline_knots(interior, boundary, degree=3):
    """Suite de nœuds complète d'une base de B-splines.

    Parameters
    ----------
    interior : array_like
        Nœuds intérieurs (triés ou non).
    boundary : tuple (a, b)
        Bornes de l'intervalle ; elles sont répétées ``degree + 1`` fois.
    degree : int
        Degré des polynômes par morceaux (3 pour des splines cubiques).

    Returns
    -------
    ndarray
        Vecteur de nœuds de longueur ``len(interior) + 2 * (degree + 1)`` ;
        la base associée a ``len(interior) + degree + 1`` fonctions.
    """
    a, b = boundary
    return np.r_[[a] * (degree + 1), np.sort(np.asarray(interior, float)),
                 [b] * (degree + 1)]


def bspline_basis(x, knots, degree=3, deriv=0):
    """Matrice des B-splines (ou de leurs dérivées) évaluées en ``x``.

    Parameters
    ----------
    x : array_like, shape (n,)
        Points d'évaluation, dans l'intervalle couvert par les nœuds.
    knots : ndarray
        Suite de nœuds complète (voir :func:`bspline_knots`).
    degree : int
        Degré des splines.
    deriv : int
        Ordre de dérivation (0 = les fonctions elles-mêmes).

    Returns
    -------
    ndarray, shape (n, p)
        Élément ``(i, j)`` : :math:`h_j^{(deriv)}(x_i)`.
    """
    x = np.asarray(x, float)
    p = len(knots) - degree - 1
    spl = BSpline(knots, np.eye(p), degree, extrapolate=False)
    if deriv:
        spl = spl.derivative(deriv)
    out = spl(x)
    return np.nan_to_num(out)


def bspline_penalty(knots, degree=3):
    """Matrice de pénalité :math:`\\Omega_{jk} = \\int h_j''(t) h_k''(t)\\,dt`.

    L'intégrale est calculée exactement par une quadrature de Gauss-Legendre
    à 3 points sur chaque intervalle entre deux nœuds consécutifs (les
    produits :math:`h_j'' h_k''` y sont des polynômes de degré au plus 2 pour
    des splines cubiques).
    """
    u = np.unique(knots)
    gx, gw = np.polynomial.legendre.leggauss(3)
    pts, wts = [], []
    for a, b in zip(u[:-1], u[1:]):
        pts.append((a + b) / 2 + (b - a) / 2 * gx)
        wts.append((b - a) / 2 * gw)
    pts, wts = np.concatenate(pts), np.concatenate(wts)
    D2 = bspline_basis(pts, knots, degree, deriv=2)
    return D2.T @ (wts[:, None] * D2)


def smoother_matrix(H, Omega, lam):
    """Matrice de lissage :math:`S_\\lambda = H (H^TH + \\lambda\\Omega)^{-1} H^T`."""
    A = H.T @ H + lam * Omega
    return H @ np.linalg.solve(A, H.T)


def loocv_linear_smoother(S, y):
    """Erreur de validation croisée « leave-one-out » d'un lisseur linéaire.

    Utilise la formule exacte
    :math:`\\frac1N\\sum_i \\left(\\frac{y_i-\\hat f(x_i)}{1-S_{ii}}\\right)^2`.
    """
    r = (y - S @ y) / (1 - np.diag(S))
    return np.mean(r ** 2)


def gcv_linear_smoother(S, y):
    """Critère de validation croisée généralisée (GCV) d'un lisseur linéaire."""
    n = len(y)
    r = y - S @ y
    return np.mean(r ** 2) / (1 - np.trace(S) / n) ** 2


# ---------------------------------------------------------------------------
# Moindres carrés, maximum de vraisemblance, bootstrap, Bayes
# ---------------------------------------------------------------------------

def ls_fit(H, y):
    """Ajustement par moindres carrés = maximum de vraisemblance gaussien.

    Parameters
    ----------
    H : ndarray, shape (N, p)
        Matrice de la base évaluée aux données.
    y : ndarray, shape (N,)
        Réponses.

    Returns
    -------
    beta : ndarray, shape (p,)
        :math:`\\hat\\beta = (H^TH)^{-1}H^Ty` (éq. (8.2)).
    sigma2 : float
        :math:`\\hat\\sigma^2 = \\frac1N\\sum_i (y_i-\\hat\\mu(x_i))^2`
        (estimateur du maximum de vraisemblance, éq. (8.21)).
    cov : ndarray, shape (p, p)
        :math:`(H^TH)^{-1}\\hat\\sigma^2` (éq. (8.3)).
    """
    HtH_inv = np.linalg.inv(H.T @ H)
    beta = HtH_inv @ H.T @ y
    sigma2 = np.mean((y - H @ beta) ** 2)
    return beta, sigma2, HtH_inv * sigma2


def pointwise_se(Hx, cov):
    """Erreurs standard ponctuelles :math:`\\sqrt{h(x)^T C\\, h(x)}` (éq. (8.4))."""
    return np.sqrt(np.einsum("ij,jk,ik->i", Hx, cov, Hx))


def bootstrap_indices(n, B, rng):
    """``B`` tirages avec remise de ``n`` indices parmi ``0..n-1``.

    Returns
    -------
    ndarray, shape (B, n)
    """
    return rng.randint(0, n, size=(B, n))


def bootstrap_curves(x, y, fit_predict, x_eval, B, rng):
    """Bootstrap non paramétrique d'un lisseur.

    Parameters
    ----------
    x, y : ndarray, shape (N,)
        Données d'apprentissage ; l'unité de rééchantillonnage est le couple.
    fit_predict : callable
        ``fit_predict(x_b, y_b, x_eval)`` ajuste sur l'échantillon bootstrap
        et renvoie les prédictions en ``x_eval``.
    x_eval : ndarray
        Points où l'on évalue les courbes.
    B : int
        Nombre d'échantillons bootstrap.
    rng : numpy.random.RandomState

    Returns
    -------
    ndarray, shape (B, len(x_eval))
    """
    idx = bootstrap_indices(len(y), B, rng)
    return np.array([fit_predict(x[i], y[i], x_eval) for i in idx])


def parametric_bootstrap_curves(H, y, H_eval, B, rng):
    """Bootstrap paramétrique (éq. (8.6)) du lisseur par moindres carrés.

    On simule :math:`y^*_i = \\hat\\mu(x_i) + \\varepsilon^*_i`,
    :math:`\\varepsilon^*_i \\sim N(0, \\hat\\sigma^2)` puis on réajuste.

    Returns
    -------
    ndarray, shape (B, len(H_eval))
    """
    beta, sigma2, _ = ls_fit(H, y)
    fitted = H @ beta
    P = np.linalg.solve(H.T @ H, H.T)          # (p, N)
    eps = rng.randn(B, len(y)) * np.sqrt(sigma2)
    betas = (fitted + eps) @ P.T                # (B, p)
    return betas @ H_eval.T


def gaussian_posterior(H, y, sigma2, tau, Sigma=None):
    """Loi a posteriori de :math:`\\beta` sous l'a priori :math:`N(0, \\tau\\Sigma)`.

    Implémente l'éq. (8.27) :
    :math:`E(\\beta|Z) = (H^TH + \\frac{\\sigma^2}{\\tau}\\Sigma^{-1})^{-1}H^Ty`,
    :math:`\\mathrm{cov}(\\beta|Z) = (H^TH + \\frac{\\sigma^2}{\\tau}\\Sigma^{-1})^{-1}\\sigma^2`.

    Returns
    -------
    mean : ndarray, shape (p,)
    cov : ndarray, shape (p, p)
    """
    p = H.shape[1]
    Sigma_inv = np.eye(p) if Sigma is None else np.linalg.inv(Sigma)
    A = H.T @ H + sigma2 / tau * Sigma_inv
    A_inv = np.linalg.inv(A)
    return A_inv @ H.T @ y, A_inv * sigma2


# ---------------------------------------------------------------------------
# Mélange de deux gaussiennes : EM et Gibbs
# ---------------------------------------------------------------------------

def normal_pdf(y, mu, var):
    """Densité de la loi :math:`N(\\mu, \\sigma^2)` (``var`` = :math:`\\sigma^2`)."""
    return np.exp(-0.5 * (y - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def mixture_loglik(y, theta):
    """Log-vraisemblance observée (8.39) du mélange à 2 composantes.

    ``theta = (mu1, var1, mu2, var2, pi)`` où ``pi`` = Pr(Δ = 1).
    """
    mu1, v1, mu2, v2, pi = theta
    return np.sum(np.log((1 - pi) * normal_pdf(y, mu1, v1)
                         + pi * normal_pdf(y, mu2, v2)))


def responsibilities(y, theta):
    """Responsabilités :math:`\\hat\\gamma_i` de la composante 2 (éq. (8.42))."""
    mu1, v1, mu2, v2, pi = theta
    a = (1 - pi) * normal_pdf(y, mu1, v1)
    b = pi * normal_pdf(y, mu2, v2)
    return b / (a + b)


def mixture_Q(y, theta_new, theta_old):
    """Fonction :math:`Q(\\theta', \\theta)` de l'étape E (éq. (8.43)).

    Espérance de la log-vraisemblance complète (8.40) sous la loi des
    variables latentes :math:`\\Delta_i` sachant les données et ``theta_old``.
    """
    g = responsibilities(y, theta_old)
    mu1, v1, mu2, v2, pi = theta_new
    return np.sum((1 - g) * (np.log(1 - pi) + np.log(normal_pdf(y, mu1, v1)))
                  + g * (np.log(pi) + np.log(normal_pdf(y, mu2, v2))))


def em_mixture(y, theta0, n_iter=100, tol=None):
    """Algorithme EM (algorithme 8.1) pour un mélange de deux gaussiennes.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Données.
    theta0 : tuple
        Valeurs initiales ``(mu1, var1, mu2, var2, pi)``.
    n_iter : int
        Nombre maximal d'itérations.
    tol : float or None
        Arrêt si la log-vraisemblance augmente de moins de ``tol``.

    Returns
    -------
    dict
        ``theta`` (estimation finale), ``thetas`` (trajectoire, la première
        ligne étant ``theta0``), ``loglik`` (log-vraisemblance observée le long
        de la trajectoire), ``gamma`` (responsabilités finales).
    """
    y = np.asarray(y, float)
    theta = tuple(float(t) for t in theta0)
    thetas, logliks = [theta], [mixture_loglik(y, theta)]
    for _ in range(n_iter):
        g = responsibilities(y, theta)                   # étape E
        w1, w2 = 1 - g, g                                # étape M
        mu1 = np.sum(w1 * y) / np.sum(w1)
        mu2 = np.sum(w2 * y) / np.sum(w2)
        v1 = np.sum(w1 * (y - mu1) ** 2) / np.sum(w1)
        v2 = np.sum(w2 * (y - mu2) ** 2) / np.sum(w2)
        theta = (mu1, v1, mu2, v2, np.mean(g))
        thetas.append(theta)
        logliks.append(mixture_loglik(y, theta))
        if tol is not None and logliks[-1] - logliks[-2] < tol:
            break
    return {"theta": theta, "thetas": np.array(thetas),
            "loglik": np.array(logliks), "gamma": responsibilities(y, theta)}


def gibbs_mixture(y, var1, var2, pi, mu_init, n_iter, rng, book_variance=False):
    """Échantillonneur de Gibbs pour les moyennes d'un mélange (algorithme 8.4).

    Les variances et la proportion sont fixées ; on alterne
    (a) le tirage des :math:`\\Delta_i` selon les responsabilités et
    (b) le tirage des moyennes sachant les :math:`\\Delta_i`.

    Parameters
    ----------
    book_variance : bool
        Si ``False`` (défaut), on tire :math:`\\mu_k \\sim N(\\bar y_k,
        \\sigma_k^2/n_k)`, loi a posteriori exacte sous un a priori plat ;
        si ``True``, on utilise :math:`N(\\bar y_k, \\sigma_k^2)` comme écrit
        dans l'algorithme 8.4 du livre.

    Returns
    -------
    mus : ndarray, shape (n_iter, 2)
    props : ndarray, shape (n_iter,)
        Proportion :math:`\\sum_i \\Delta_i / N` à chaque itération.
    """
    y = np.asarray(y, float)
    mu1, mu2 = mu_init
    mus, props = np.empty((n_iter, 2)), np.empty(n_iter)
    for t in range(n_iter):
        g = responsibilities(y, (mu1, var1, mu2, var2, pi))
        d = rng.rand(len(y)) < g
        n2 = d.sum()
        n1 = len(y) - n2
        m1 = y[~d].mean() if n1 > 0 else mu1
        m2 = y[d].mean() if n2 > 0 else mu2
        s1 = var1 if book_variance else var1 / max(n1, 1)
        s2 = var2 if book_variance else var2 / max(n2, 1)
        mu1 = m1 + np.sqrt(s1) * rng.randn()
        mu2 = m2 + np.sqrt(s2) * rng.randn()
        mus[t] = mu1, mu2
        props[t] = n2 / len(y)
    return mus, props
