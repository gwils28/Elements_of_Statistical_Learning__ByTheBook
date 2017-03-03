"""Outils du chapitre 9 : modèles additifs, arbres et méthodes apparentées.

Ce module regroupe les implémentations « à la main » utilisées dans les
notebooks du chapitre 9 :

* lisseurs linéaires : spline de lissage cubique (base de B-splines pénalisée,
  section 5.4 du livre) et régression linéaire locale (section 6.1.1) ;
* algorithme de *backfitting* (algorithme 9.1) et modèle additif généralisé
  logistique par « local scoring » (algorithme 9.2) ;
* arbres CART de régression et de classification (section 9.2) avec élagage
  coût-complexité (éq. (9.16)) et validation croisée ;
* PRIM, la recherche de « bosses » (algorithme 9.3) ;
* une version simplifiée de MARS (section 9.4) ;
* quelques outils d'évaluation (courbe ROC, aire sous la courbe).
"""

import heapq

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq

#: Noms des 57 prédicteurs des données spam (dans l'ordre des colonnes A.1 ... A.57).
SPAM_NAMES = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference",
    "ch;", "ch(", "ch[", "ch!", "ch$", "ch#", "CAPAVE", "CAPMAX", "CAPTOT",
]


def _wmean(v, w):
    """Moyenne pondérée (poids ``None`` = poids égaux)."""
    if w is None:
        return float(np.mean(v))
    return float(np.sum(w * v) / np.sum(w))


# ---------------------------------------------------------------------------
# Lisseurs linéaires
# ---------------------------------------------------------------------------


def _penalty_matrix(t, k=3):
    """Matrice de pénalité :math:`\\Omega_{ab} = \\int B_a''(x) B_b''(x)\\,dx`.

    Les dérivées secondes de B-splines cubiques sont affines par morceaux :
    une quadrature de Gauss-Legendre à 2 points par intervalle est exacte.
    """
    K = len(t) - k - 1
    bs = BSpline(t, np.eye(K), k).derivative(2)
    nodes = np.unique(t)
    g, gw = np.polynomial.legendre.leggauss(2)
    a, b = nodes[:-1], nodes[1:]
    pts = (0.5 * (b - a)[:, None] * g[None, :] + 0.5 * (a + b)[:, None]).ravel()
    wts = (0.5 * (b - a)[:, None] * gw[None, :]).ravel()
    D = bs(pts)  # (npts, K)
    return D.T @ (wts[:, None] * D)


class SplineLisseur:
    """Spline de lissage cubique pénalisée (lisseur linéaire de nuage de points).

    On minimise :math:`\\sum_i w_i (y_i - f(x_i))^2 + \\lambda \\int f''(t)^2 dt`
    sur l'espace des splines cubiques dont les nœuds sont des valeurs observées
    de :math:`x`. Lorsque tous les nœuds sont utilisés (``n_knots=None``), on
    obtient exactement la spline de lissage du livre (section 5.4) ; sinon on
    obtient une spline de régression pénalisée, beaucoup plus rapide.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Valeurs du prédicteur (fixées une fois pour toutes).
    df : float, optional
        Degrés de liberté effectifs visés, :math:`\\mathrm{tr}(S_\\lambda)`
        (calculés avec des poids unitaires). Ignoré si ``lam`` est donné.
    lam : float, optional
        Paramètre de lissage :math:`\\lambda`.
    n_knots : int or None
        Nombre de nœuds intérieurs (quantiles des valeurs distinctes) ; ``None``
        pour placer un nœud en chaque valeur distincte.
    ridge : float
        Terme :math:`\\varepsilon \\|c\\|^2` ajouté à la pénalité ; avec
        ``ridge > 0`` toutes les valeurs propres de :math:`S` sont < 1.
    """

    def __init__(self, x, df=None, lam=None, n_knots=None, ridge=0.0):
        x = np.asarray(x, dtype=float)
        self.x = x
        self.lo, self.hi = float(x.min()), float(x.max())
        u = np.unique(x)
        if len(u) < 2:
            raise ValueError("il faut au moins deux valeurs distinctes de x")
        inner = u[1:-1]
        if n_knots is not None and len(inner) > n_knots:
            inner = np.unique(np.quantile(u, np.linspace(0, 1, n_knots + 2)[1:-1]))
            inner = inner[(inner > self.lo) & (inner < self.hi)]
        self.t = np.r_[[self.lo] * 4, inner, [self.hi] * 4]
        self.B = self._basis(x)
        self.K = self.B.shape[1]
        self.Omega = _penalty_matrix(self.t)
        self.ridge = ridge
        BtB = self.B.T @ self.B
        # échelle naturelle pour rendre lambda sans dimension dans la recherche
        self._scale = np.trace(BtB) / max(np.trace(self.Omega), 1e-300)
        self.n_unique = len(u)
        if lam is not None:
            self.lam = float(lam)
        elif df is not None:
            self.lam = self.lambda_for_df(df)
        else:
            raise ValueError("donner df ou lam")
        self.coef_ = None

    def _basis(self, x):
        x = np.clip(np.asarray(x, dtype=float), self.lo, self.hi)
        return BSpline.design_matrix(x, self.t, 3).toarray()

    def _pen(self, lam):
        eps = 1e-10 * np.trace(self.B.T @ self.B) / self.K
        return lam * self.Omega + (self.ridge + eps) * np.eye(self.K)

    def df_for_lambda(self, lam, w=None):
        """Trace de la matrice de lissage :math:`S_\\lambda` (poids ``w``)."""
        Bw = self.B if w is None else self.B * w[:, None]
        G = self.B.T @ Bw
        return float(np.trace(np.linalg.solve(G + self._pen(lam), G)))

    def lambda_for_df(self, df):
        """Valeur de :math:`\\lambda` telle que :math:`\\mathrm{tr}(S_\\lambda)=` ``df``."""
        dmax = self.df_for_lambda(1e-8 * self._scale)
        if df >= dmax - 1e-6:
            return 1e-8 * self._scale
        if df <= 2.0 + 1e-6:
            raise ValueError("df doit être > 2 (la spline contient les droites)")
        f = lambda s: self.df_for_lambda(np.exp(s) * self._scale) - df
        lo, hi = -18.0, 18.0
        while f(hi) > 0:
            hi += 6.0
        return float(np.exp(brentq(f, lo, hi, xtol=1e-6)) * self._scale)

    def fit(self, y, w=None):
        """Ajuste la spline aux réponses ``y`` (poids ``w``) ; renvoie les valeurs ajustées."""
        Bw = self.B if w is None else self.B * w[:, None]
        G = self.B.T @ Bw
        self.coef_ = np.linalg.solve(G + self._pen(self.lam), Bw.T @ y)
        return self.B @ self.coef_

    def predict(self, x):
        """Évalue la dernière spline ajustée (extrapolation constante hors de l'étendue)."""
        return self._basis(x) @ self.coef_

    def matrix(self, w=None):
        """Matrice de lissage :math:`S = B(B^TWB + \\lambda\\Omega)^{-1}B^TW` (N x N)."""
        Bw = self.B if w is None else self.B * w[:, None]
        G = self.B.T @ Bw
        return self.B @ np.linalg.solve(G + self._pen(self.lam), Bw.T)


class MatriceLisseur:
    """Lisseur linéaire défini directement par une matrice :math:`S` (N x N)."""

    def __init__(self, S):
        self.S = np.asarray(S, dtype=float)

    def fit(self, y, w=None):
        return self.S @ y
