"""Bases de fonctions, splines et lissage - chapitre 5 d'ESL.

Ce module regroupe les outils réutilisés dans les notebooks du chapitre 5 :

* bases de polynômes par morceaux et base des puissances tronquées (§5.2) ;
* B-splines par la récurrence de Cox-de Boor (annexe du chapitre, éq. (5.77)-(5.78)),
  ainsi que leurs dérivées ;
* base des splines cubiques naturelles (éq. (5.4)-(5.5)) ;
* matrice de pénalité :math:`\\Omega_{jk} = \\int B_j''(t) B_k''(t)\\,dt` et spline de
  lissage (éq. (5.9)-(5.12), (5.79)), matrice de lissage, degrés de liberté effectifs
  (éq. (5.16)), validation croisée « leave-one-out » (éq. (5.27)) ;
* transformée de Haar rapide et seuillage doux (§5.9) ;
* régression logistique par IRLS (utilisée pour l'exemple des phonèmes, §5.2.3).
"""

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "truncated_power_basis", "piecewise_polynomial_basis", "augmented_knots",
    "bspline_basis", "natural_spline_basis", "bspline_penalty_matrix",
    "SmoothingSpline", "smoother_matrix", "effective_df", "lambda_for_df",
    "loocv_score", "gcv_score", "haar_transform", "inverse_haar_transform",
    "haar_matrix", "soft_threshold", "logistic_irls",
]


# ---------------------------------------------------------------------------
# Polynômes par morceaux et puissances tronquées
# ---------------------------------------------------------------------------

def piecewise_polynomial_basis(x, knots, degree=3, continuity=None):
    """Base de polynômes par morceaux de degré ``degree`` avec nœuds ``knots``.

    La base est construite à partir des fonctions :math:`1, x, \\dots, x^d` et,
    pour chaque nœud :math:`\\xi`, des puissances tronquées
    :math:`(x-\\xi)_+^j` pour :math:`j = c+1, \\dots, d`, où :math:`c` est
    l'ordre de continuité imposé (``continuity``). On convient que
    :math:`(x-\\xi)_+^0 = \\mathbf 1(x \\geq \\xi)`.

    Parameters
    ----------
    x : array_like, shape (N,)
        Points d'évaluation.
    knots : array_like, shape (K,)
        Nœuds intérieurs.
    degree : int
        Degré :math:`d` des polynômes (ordre :math:`M = d + 1`).
    continuity : int or None
        Les dérivées d'ordre :math:`0, \\dots, c` sont continues aux nœuds.
        ``-1`` : fonction discontinue ; ``None`` (défaut) : :math:`c = d-1`,
        c'est-à-dire une spline (base des puissances tronquées).

    Returns
    -------
    H : ndarray, shape (N, d + 1 + K (d - c))
    """
    x = np.asarray(x, dtype=float)
    knots = np.atleast_1d(np.asarray(knots, dtype=float))
    if continuity is None:
        continuity = degree - 1
    cols = [x ** j for j in range(degree + 1)]
    for xi in knots:
        for j in range(continuity + 1, degree + 1):
            if j == 0:
                cols.append((x >= xi).astype(float))
            else:
                cols.append(np.where(x > xi, (x - xi) ** j, 0.0))
    return np.column_stack(cols)


def truncated_power_basis(x, knots, degree=3):
    """Base des puissances tronquées d'une spline d'ordre ``degree + 1``.

    Colonnes : :math:`h_j(x) = x^{j-1}` (:math:`j = 1..M`) puis
    :math:`h_{M+\\ell}(x) = (x - \\xi_\\ell)_+^{M-1}` (:math:`\\ell = 1..K`),
    cf. éq. (5.3) du livre et le paragraphe qui la suit.

    Parameters
    ----------
    x : array_like, shape (N,)
    knots : array_like, shape (K,)
    degree : int
        Degré :math:`M - 1` (3 pour les splines cubiques).

    Returns
    -------
    H : ndarray, shape (N, M + K)
    """
    return piecewise_polynomial_basis(x, knots, degree=degree,
                                      continuity=degree - 1)


# ---------------------------------------------------------------------------
# B-splines (Cox-de Boor)
# ---------------------------------------------------------------------------

def augmented_knots(inner_knots, lo, hi, order=4):
    """Suite de nœuds augmentée :math:`\\tau` de l'annexe du chapitre.

    Les nœuds de bord :math:`\\xi_0 = lo` et :math:`\\xi_{K+1} = hi` sont
    répétés ``order`` fois : :math:`\\tau = (lo, \\dots, lo, \\xi_1, \\dots,
    \\xi_K, hi, \\dots, hi)`, de longueur :math:`K + 2M`.
    """
    inner = np.sort(np.atleast_1d(np.asarray(inner_knots, dtype=float)))
    return np.concatenate([np.repeat(float(lo), order), inner,
                           np.repeat(float(hi), order)])


def _safe_div(num, den):
    """Division avec la convention 0/0 = 0 (nœuds répétés)."""
    out = np.zeros_like(num)
    ok = den != 0
    out[:, ok] = num[:, ok] / den[ok]
    return out


def _bspline_from_tau(x, tau, order, deriv=0):
    """B-splines d'ordre ``order`` (et dérivée ``deriv``) pour la suite ``tau``."""
    x = np.asarray(x, dtype=float)
    if deriv > 0:
        # B'_{i,m} = (m-1) [B_{i,m-1}/(tau_{i+m-1}-tau_i) - B_{i+1,m-1}/(tau_{i+m}-tau_{i+1})]
        lower = _bspline_from_tau(x, tau, order - 1, deriv - 1)
        m = order
        n = len(tau) - m
        d1 = tau[m - 1:m - 1 + n] - tau[:n]
        d2 = tau[m:m + n] - tau[1:1 + n]
        return (m - 1) * (_safe_div(lower[:, :n], d1)
                          - _safe_div(lower[:, 1:n + 1], d2))
    # ordre 1 : indicatrices des intervalles [tau_i, tau_{i+1}) (éq. 5.77)
    n1 = len(tau) - 1
    B = ((tau[:n1] <= x[:, None]) & (x[:, None] < tau[1:])).astype(float)
    # le point hi (bord droit) est rattaché au dernier intervalle non vide
    nonempty = np.nonzero(tau[1:] > tau[:-1])[0]
    if len(nonempty):
        last = nonempty[-1]
        B[x == tau[last + 1], last] = 1.0
    # récurrence de Cox-de Boor (éq. 5.78)
    for m in range(2, order + 1):
        n = len(tau) - m
        w1 = _safe_div((x[:, None] - tau[:n]), tau[m - 1:m - 1 + n] - tau[:n])
        w2 = _safe_div((tau[m:m + n] - x[:, None]), tau[m:m + n] - tau[1:1 + n])
        B = w1 * B[:, :n] + w2 * B[:, 1:n + 1]
    return B


def bspline_basis(x, inner_knots, lo=None, hi=None, order=4, deriv=0):
    """Base de B-splines d'ordre ``order`` (récurrence de Cox-de Boor).

    Parameters
    ----------
    x : array_like, shape (N,)
        Points d'évaluation (les B-splines sont nulles hors de ``[lo, hi]``).
    inner_knots : array_like, shape (K,)
        Nœuds intérieurs :math:`\\xi_1 < \\dots < \\xi_K`.
    lo, hi : float, optional
        Nœuds de bord :math:`\\xi_0, \\xi_{K+1}` (par défaut min et max de ``x``).
    order : int
        Ordre :math:`M` (4 pour les splines cubiques).
    deriv : int
        Ordre de dérivation (0 : les fonctions elles-mêmes).

    Returns
    -------
    B : ndarray, shape (N, K + M)
    """
    x = np.asarray(x, dtype=float)
    lo = x.min() if lo is None else lo
    hi = x.max() if hi is None else hi
    tau = augmented_knots(inner_knots, lo, hi, order)
    return _bspline_from_tau(x, tau, order, deriv)


def bspline_penalty_matrix(inner_knots, lo, hi, order=4, deriv=2):
    """Matrice :math:`\\Omega_{jk} = \\int_{lo}^{hi} B_j^{(m)}(t) B_k^{(m)}(t)\\,dt`.

    Le calcul est exact : sur chaque intervalle entre nœuds consécutifs, le
    produit des dérivées est un polynôme de degré :math:`2(M-1-m)`, intégré
    exactement par une quadrature de Gauss-Legendre à :math:`M - m` points.

    Parameters
    ----------
    inner_knots : array_like
    lo, hi : float
    order : int
        Ordre :math:`M` des B-splines.
    deriv : int
        Ordre :math:`m` de la dérivée pénalisée (2 pour les splines de lissage).

    Returns
    -------
    Omega : ndarray, shape (K + M, K + M), symétrique semi-définie positive.
    """
    br = np.unique(np.concatenate([[lo], np.atleast_1d(inner_knots), [hi]]))
    npts = max(order - deriv, 1)
    g, w = np.polynomial.legendre.leggauss(npts)
    a, b = br[:-1], br[1:]
    t = ((a + b)[:, None] + (b - a)[:, None] * g[None, :]) / 2
    wt = ((b - a)[:, None] / 2) * w[None, :]
    D = bspline_basis(t.ravel(), inner_knots, lo, hi, order, deriv)
    return D.T @ (wt.ravel()[:, None] * D)


# ---------------------------------------------------------------------------
# Splines cubiques naturelles
# ---------------------------------------------------------------------------

def natural_spline_basis(x, knots):
    """Base des splines cubiques naturelles de l'éq. (5.4)-(5.5) du livre.

    :math:`N_1(x) = 1`, :math:`N_2(x) = x`, :math:`N_{k+2}(x) = d_k(x) - d_{K-1}(x)`
    avec :math:`d_k(x) = \\frac{(x-\\xi_k)_+^3 - (x-\\xi_K)_+^3}{\\xi_K - \\xi_k}`.

    Parameters
    ----------
    x : array_like, shape (N,)
    knots : array_like, shape (K,), K >= 2
        Tous les nœuds, y compris les nœuds de bord :math:`\\xi_1` et :math:`\\xi_K`.

    Returns
    -------
    N : ndarray, shape (N, K)
        Fonctions linéaires à gauche de :math:`\\xi_1` et à droite de :math:`\\xi_K`.
    """
    x = np.asarray(x, dtype=float)
    xi = np.sort(np.asarray(knots, dtype=float))
    K = len(xi)
    cube = lambda u: np.where(u > 0, u, 0.0) ** 3

    def d(k):
        return (cube(x - xi[k]) - cube(x - xi[K - 1])) / (xi[K - 1] - xi[k])

    cols = [np.ones_like(x), x]
    dK1 = d(K - 2)
    for k in range(K - 2):
        cols.append(d(k) - dK1)
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Splines de lissage
# ---------------------------------------------------------------------------

class SmoothingSpline:
    """Spline cubique de lissage (éq. (5.9) du livre), calculée en base de B-splines.

    On minimise :math:`\\sum_i w_i (y_i - f(x_i))^2 + \\lambda \\int f''(t)^2 dt`.
    La solution est une spline cubique naturelle à nœuds aux valeurs distinctes
    :math:`x_{(1)} < \\dots < x_{(n)}` ; on la représente dans la base des
    :math:`n + 2` B-splines cubiques ayant pour nœuds intérieurs
    :math:`x_{(2)}, \\dots, x_{(n-1)}` (exercice 5.8) :
    :math:`\\hat\\gamma = (B^T W B + \\lambda\\Omega_B)^{-1} B^T W y` (éq. (5.79)).

    Parameters
    ----------
    lam : float, optional
        Paramètre de lissage :math:`\\lambda` (échelle de ``x`` d'origine).
    df : float, optional
        Degrés de liberté effectifs visés (si ``lam`` n'est pas donné).
    """

    def __init__(self, lam=None, df=None):
        if lam is None and df is None:
            raise ValueError("Préciser lam ou df.")
        self.lam = lam
        self.df = df

    def _design(self, x):
        self.ux_ = np.unique(x)
        if len(self.ux_) < 3:
            raise ValueError("Il faut au moins 3 valeurs distinctes de x.")
        self.lo_, self.hi_ = self.ux_[0], self.ux_[-1]
        self.inner_ = self.ux_[1:-1]
        B = bspline_basis(x, self.inner_, self.lo_, self.hi_)
        Om = bspline_penalty_matrix(self.inner_, self.lo_, self.hi_)
        return B, Om

    def fit(self, x, y, w=None):
        """Ajuste la spline aux données ``(x, y)`` avec poids ``w`` (défaut 1).

        Pour la stabilité numérique (et pour obtenir :math:`df_\\lambda` en
        :math:`O(p)` pour tout :math:`\\lambda`), on diagonalise simultanément
        :math:`B^TWB` et :math:`\\Omega` : avec :math:`A_0 = B^TWB + c\\,\\Omega = LL^T`
        et :math:`L^{-1}\\Omega L^{-T} = U\\,\\mathrm{diag}(d)\\,U^T`, on a
        :math:`B^TWB + \\lambda\\Omega = LU\\,\\mathrm{diag}(1 + (\\lambda - c) d)\\,U^TL^T`.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
        B, Om = self._design(x)
        BtWB = B.T @ (w[:, None] * B)
        c = np.trace(BtWB) / np.trace(Om)
        L = np.linalg.cholesky(BtWB + c * Om)
        Li = np.linalg.solve(L, np.eye(len(L)))
        G = Li @ Om @ Li.T
        d, U = np.linalg.eigh((G + G.T) / 2)
        # le noyau de Omega (fonctions linéaires) : valeurs propres exactement nulles
        d = np.where(d < 1e-10 * d.max(), 0.0, d)
        self.c_, self.d_, self.T_ = c, d, Li.T @ U
        # e_k = 1 - c d_k >= 0 : valeurs propres de B^TWB dans la même base
        self.e_ = np.clip(1.0 - c * d, 0.0, None)
        self.BtWB_, self.Omega_, self.B_, self.w_ = BtWB, Om, B, w
        self.x_, self.y_ = x, y
        if self.lam is None:
            self.lam_ = self._lambda_for_df(self.df)
        else:
            self.lam_ = float(self.lam)
        self._proj_ = self.T_.T @ (B.T @ (w * y))
        return self.set_lambda(self.lam_)

    def set_lambda(self, lam):
        """Change :math:`\\lambda` sans refaire la diagonalisation (coût :math:`O(p^2)`)."""
        self.lam_ = float(lam)
        self.coef_ = self.T_ @ (self._shrink(self.lam_) * self._proj_)
        self.df_ = self._df(self.lam_)
        return self

    def _shrink(self, lam):
        return 1.0 / (self.e_ + lam * self.d_)

    def _df(self, lam):
        return float(np.sum(self.e_ * self._shrink(lam)))

    def _lambda_for_df(self, df):
        n = len(self.ux_)
        if not 2 < df < n:
            raise ValueError("df doit être dans ]2, %d[." % n)
        g = lambda t: self._df(np.exp(t)) - df
        lc = np.log(self.c_)
        return float(np.exp(brentq(g, lc - 40, lc + 40, xtol=1e-12)))

    def smoother_matrix(self):
        """Matrice de lissage :math:`S_\\lambda` aux points d'apprentissage (N x N)."""
        BT = self.B_ @ self.T_
        return (BT * self._shrink(self.lam_)) @ BT.T * self.w_[None, :]

    def linear_operator(self, x0):
        """Matrice :math:`L` (len(x0) x N) telle que ``predict(x0) == L @ y``.

        La ligne :math:`\\ell(x_0)^T` donne les poids du lisseur linéaire au point
        :math:`x_0` : :math:`\\hat f(x_0) = \\sum_i \\ell_i(x_0) y_i`.
        """
        x0 = np.asarray(x0, dtype=float)
        xc = np.clip(x0, self.lo_, self.hi_)
        P = bspline_basis(xc, self.inner_, self.lo_, self.hi_)
        P = P + (x0 - xc)[:, None] * bspline_basis(xc, self.inner_, self.lo_, self.hi_, deriv=1)
        PT = P @ self.T_
        BT = self.B_ @ self.T_
        return (PT * self._shrink(self.lam_)) @ BT.T * self.w_[None, :]

    def predict(self, x0, deriv=0):
        """Évalue :math:`\\hat f` (ou sa dérivée) ; extrapolation linéaire hors de [lo, hi]."""
        x0 = np.asarray(x0, dtype=float)
        xc = np.clip(x0, self.lo_, self.hi_)
        f = bspline_basis(xc, self.inner_, self.lo_, self.hi_, deriv=deriv) @ self.coef_
        if deriv == 0:
            slope = bspline_basis(xc, self.inner_, self.lo_, self.hi_, deriv=1) @ self.coef_
            f = f + slope * (x0 - xc)
        elif deriv == 1:
            pass
        else:
            f = np.where(x0 == xc, f, 0.0)
        return f


def smoother_matrix(x, lam, w=None):
    """Matrice de lissage :math:`S_\\lambda` d'une spline de lissage (éq. (5.14))."""
    s = SmoothingSpline(lam=lam).fit(x, np.zeros(len(x)), w)
    return s.smoother_matrix()


def effective_df(x, lam, w=None):
    """Degrés de liberté effectifs :math:`df_\\lambda = \\mathrm{tr}(S_\\lambda)` (éq. (5.16))."""
    return SmoothingSpline(lam=lam).fit(x, np.zeros(len(x)), w).df_


def lambda_for_df(x, df, w=None):
    """Valeur de :math:`\\lambda` telle que :math:`\\mathrm{tr}(S_\\lambda) = df`.

    :math:`df_\\lambda` décroît strictement de :math:`n` (nombre de valeurs
    distinctes) à 2 quand :math:`\\lambda` croît de 0 à l'infini : on résout
    l'équation par la méthode de Brent sur :math:`\\log\\lambda`.
    """
    return SmoothingSpline(df=df).fit(x, np.zeros(len(x)), w).lam_


def loocv_score(x, y, lam):
    """Validation croisée « leave-one-out » par la formule (5.27) du livre."""
    s = SmoothingSpline(lam=lam).fit(x, y)
    S = s.smoother_matrix()
    r = (y - S @ y) / (1 - np.diag(S))
    return np.mean(r ** 2)


def gcv_score(x, y, lam):
    """Validation croisée généralisée : moyenne de :math:`((y_i-\\hat f_i)/(1-df/N))^2`."""
    s = SmoothingSpline(lam=lam).fit(x, y)
    S = s.smoother_matrix()
    N = len(y)
    return np.mean(((y - S @ y) / (1 - np.trace(S) / N)) ** 2)


# ---------------------------------------------------------------------------
# Ondelettes de Haar
# ---------------------------------------------------------------------------

def haar_transform(y):
    """Transformée de Haar orthonormée rapide (algorithme pyramidal en O(N)).

    Parameters
    ----------
    y : array_like, shape (N,), N = 2**J

    Returns
    -------
    c : ndarray, shape (N,)
        ``c[0]`` : coefficient d'échelle (espace :math:`V_0`) ; puis les
        coefficients d'ondelettes des niveaux :math:`j = 0, 1, \\dots, J-1`,
        le niveau :math:`j` occupant ``c[2**j : 2**(j+1)]``.
    """
    a = np.asarray(y, dtype=float)
    N = len(a)
    J = int(round(np.log2(N)))
    if 2 ** J != N:
        raise ValueError("La longueur doit être une puissance de 2.")
    details = []
    for _ in range(J):
        s = (a[0::2] + a[1::2]) / np.sqrt(2)
        d = (a[0::2] - a[1::2]) / np.sqrt(2)
        details.append(d)
        a = s
    return np.concatenate([a] + details[::-1])


def inverse_haar_transform(c):
    """Inverse de :func:`haar_transform` (également en O(N))."""
    c = np.asarray(c, dtype=float)
    N = len(c)
    J = int(round(np.log2(N)))
    a = c[:1].copy()
    for j in range(J):
        d = c[2 ** j:2 ** (j + 1)]
        out = np.empty(2 * len(a))
        out[0::2] = (a + d) / np.sqrt(2)
        out[1::2] = (a - d) / np.sqrt(2)
        a = out
    return a


def haar_matrix(N):
    """Matrice orthogonale :math:`W` (N x N) dont les colonnes sont les ondelettes
    de Haar discrètes, de sorte que ``haar_transform(y) == W.T @ y``."""
    return np.column_stack([inverse_haar_transform(e) for e in np.eye(N)])
