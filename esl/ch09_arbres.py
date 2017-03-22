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


def local_linear_matrix(x, bandwidth):
    """Matrice de lissage de la régression linéaire locale (noyau gaussien).

    La ligne :math:`i` contient les poids :math:`\\ell_k(x_i)` tels que
    :math:`\\hat f(x_i) = \\sum_k \\ell_k(x_i) y_k` (éq. (6.8)-(6.9) du livre).

    Parameters
    ----------
    x : ndarray, shape (N,)
    bandwidth : float
        Écart-type du noyau gaussien.

    Returns
    -------
    ndarray, shape (N, N)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    S = np.empty((N, N))
    B = np.column_stack([np.ones(N), x])
    for i, x0 in enumerate(x):
        k = np.exp(-0.5 * ((x - x0) / bandwidth) ** 2)
        BtW = B.T * k
        S[i] = np.array([1.0, x0]) @ np.linalg.solve(BtW @ B, BtW)
    return S


# ---------------------------------------------------------------------------
# Backfitting et modèles additifs
# ---------------------------------------------------------------------------


def backfitting(y, smoothers, w=None, intercept=True, tol=1e-6, max_iter=200,
                F0=None, history=False):
    """Algorithme de backfitting (algorithme 9.1 du livre).

    Parameters
    ----------
    y : ndarray, shape (N,)
    smoothers : list
        Objets possédant une méthode ``fit(r, w)`` qui renvoie les valeurs
        ajustées du lissage des résidus partiels ``r`` (un par terme).
    w : ndarray, optional
        Poids des observations (backfitting pondéré).
    intercept : bool
        Si vrai, :math:`\\hat\\alpha` = moyenne (pondérée) de ``y`` et chaque
        :math:`\\hat f_j` est recentré (seconde ligne de l'étape 2).
    tol : float
        Arrêt lorsque la variation relative des fonctions est inférieure à ``tol``.
    F0 : ndarray, shape (N, p), optional
        Valeurs initiales des :math:`\\hat f_j` (zéro par défaut).
    history : bool
        Si vrai, renvoie aussi la liste des RSS après chaque cycle.

    Returns
    -------
    alpha : float
    F : ndarray, shape (N, p)
        Colonne :math:`j` = :math:`\\hat f_j(x_{ij})`.
    n_iter : int
    hist : list (si ``history``)
    """
    y = np.asarray(y, dtype=float)
    N, p = len(y), len(smoothers)
    alpha = _wmean(y, w) if intercept else 0.0
    F = np.zeros((N, p)) if F0 is None else np.array(F0, dtype=float)
    total = F.sum(axis=1)
    hist = []
    it = 0
    for it in range(1, max_iter + 1):
        delta, norm = 0.0, 0.0
        for j, sm in enumerate(smoothers):
            r = y - alpha - total + F[:, j]
            fj = sm.fit(r, w)
            if intercept:
                fj = fj - _wmean(fj, w)
            delta += np.sum((fj - F[:, j]) ** 2)
            norm += np.sum(fj ** 2)
            total += fj - F[:, j]
            F[:, j] = fj
        if history:
            hist.append(float(np.sum((y - alpha - total) ** 2)))
        if delta <= tol ** 2 * max(norm, 1e-300):
            break
    if history:
        return alpha, F, it, hist
    return alpha, F, it


def logistic_irls(X, y, w=None, max_iter=50, tol=1e-8):
    """Régression logistique linéaire par moindres carrés repondérés (section 4.4.1).

    Parameters
    ----------
    X : ndarray, shape (N, p)
        Sans colonne de 1 (l'intercept est ajouté).
    y : ndarray, shape (N,), valeurs 0/1
    w : ndarray, optional
        Poids des observations.

    Returns
    -------
    beta : ndarray, shape (p + 1,)
        Intercept puis coefficients.
    """
    X1 = np.column_stack([np.ones(len(y)), X])
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    beta = np.zeros(X1.shape[1])
    for _ in range(max_iter):
        eta = X1 @ beta
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None) * w
        z = eta + (y - p) / np.clip(p * (1 - p), 1e-10, None)
        A = X1.T @ (W[:, None] * X1) + 1e-8 * np.eye(X1.shape[1])
        new = np.linalg.solve(A, X1.T @ (W * z))
        if np.max(np.abs(new - beta)) < tol * (1 + np.max(np.abs(beta))):
            beta = new
            break
        beta = new
    return beta


class GAM:
    """Modèle additif généralisé à splines de lissage (gaussien ou logistique).

    * ``family="gaussian"`` : backfitting (algorithme 9.1) ;
    * ``family="binomial"`` : « local scoring » (algorithme 9.2), c'est-à-dire
      Newton-Raphson dont chaque étape est un backfitting pondéré.

    Parameters
    ----------
    df : float or sequence
        Degrés de liberté *nominaux* de chaque terme, au sens du livre :
        :math:`\\mathrm{df}_j = \\mathrm{tr}(S_j) - 1`.
    n_knots : int or None
        Nombre de nœuds intérieurs de chaque spline (``None`` = tous).
    """

    def __init__(self, df=4.0, n_knots=12, family="gaussian", tol=1e-6,
                 max_iter=30, backfit_tol=1e-4, backfit_iter=50):
        self.df, self.n_knots, self.family = df, n_knots, family
        self.tol, self.max_iter = tol, max_iter
        self.backfit_tol, self.backfit_iter = backfit_tol, backfit_iter

    def fit(self, X, y, w=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        N, p = X.shape
        dfs = np.broadcast_to(np.asarray(self.df, float), (p,))
        self.smoothers_ = []
        for j in range(p):
            sm = SplineLisseur(X[:, j], lam=1.0, n_knots=self.n_knots)
            dmax = sm.df_for_lambda(1e-8 * sm._scale)
            sm.lam = sm.lambda_for_df(min(dfs[j] + 1.0, dmax - 1e-3))
            self.smoothers_.append(sm)
        w0 = np.ones(N) if w is None else np.asarray(w, float)
        if self.family == "gaussian":
            self.alpha_, self.F_, self.n_iter_ = backfitting(
                y, self.smoothers_, w=w0, tol=self.backfit_tol,
                max_iter=self.backfit_iter)
            z_fin, w_fin = y, w0
        elif self.family == "binomial":
            ybar = np.clip(_wmean(y, w0), 1e-6, 1 - 1e-6)
            alpha = np.log(ybar / (1 - ybar))
            F = np.zeros((N, p))
            dev_old = np.inf
            for it in range(1, self.max_iter + 1):
                eta = alpha + F.sum(axis=1)
                pr = np.clip(1 / (1 + np.exp(-eta)), 1e-6, 1 - 1e-6)
                z = eta + (y - pr) / (pr * (1 - pr))
                ww = w0 * pr * (1 - pr)
                ww = ww / ww.mean()  # normalisation : lambda garde son sens
                alpha, F, _ = backfitting(z, self.smoothers_, w=ww, F0=F,
                                          tol=self.backfit_tol,
                                          max_iter=self.backfit_iter)
                z_fin, w_fin = z, ww
                eta = alpha + F.sum(axis=1)
                pr = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
                dev = -2 * np.sum(w0 * (y * np.log(pr) + (1 - y) * np.log(1 - pr)))
                if abs(dev_old - dev) < self.tol * (abs(dev) + 1):
                    break
                dev_old = dev
            self.alpha_, self.F_, self.n_iter_ = alpha, F, it
            self.deviance_ = dev
        else:
            raise ValueError("family inconnue")
        # coefficients de chaque spline : on relisse une dernière fois les résidus
        # partiels ; les B-splines sommant à 1, le recentrage revient à retrancher
        # une constante aux coefficients.
        self.coefs_ = []
        tot = self.F_.sum(axis=1)
        for j, sm in enumerate(self.smoothers_):
            r = z_fin - self.alpha_ - tot + self.F_[:, j]
            fj = sm.fit(r, w_fin)
            self.coefs_.append(sm.coef_ - _wmean(fj, w_fin))
        return self

    def partial(self, X):
        """Matrice des contributions :math:`\\hat f_j(x_{ij})` pour de nouvelles données."""
        X = np.asarray(X, dtype=float)
        return np.column_stack([sm._basis(X[:, j]) @ self.coefs_[j]
                                for j, sm in enumerate(self.smoothers_)])

    def decision_function(self, X):
        """Prédicteur additif :math:`\\hat\\alpha + \\sum_j \\hat f_j(x_j)`."""
        return self.alpha_ + self.partial(X).sum(axis=1)

    def predict_proba(self, X):
        """Probabilité :math:`\\Pr(Y=1\\mid x)` (famille binomiale)."""
        return 1 / (1 + np.exp(-self.decision_function(X)))

    def predict(self, X):
        if self.family == "binomial":
            return (self.decision_function(X) > 0).astype(int)
        return self.decision_function(X)


# ---------------------------------------------------------------------------
# Arbres CART
# ---------------------------------------------------------------------------


def impurity(p, criterion="gini"):
    """Impureté d'un nœud à partir des proportions de classes (éq. (9.17)).

    Parameters
    ----------
    p : ndarray, shape (..., K)
        Proportions :math:`\\hat p_{mk}` (sommant à 1 sur le dernier axe).
    criterion : {"gini", "entropy", "misclass"}
    """
    p = np.asarray(p, dtype=float)
    if criterion == "gini":
        return np.sum(p * (1 - p), axis=-1)
    if criterion == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)
    if criterion == "misclass":
        return 1 - np.max(p, axis=-1)
    raise ValueError(criterion)


class ArbreCART:
    """Arbre CART de régression ou de classification (section 9.2 du livre).

    La croissance est gloutonne et « meilleur d'abord » : on découpe à chaque
    étape la feuille dont la meilleure coupure fait le plus baisser le critère.
    Sans ``max_leaves``, on obtient l'arbre maximal :math:`T_0` (on ne découpe plus
    un nœud de moins de ``min_samples_split`` observations).

    Parameters
    ----------
    task : {"classification", "regression"}
    criterion : {"gini", "entropy", "mse"}
        Critère de croissance (``"mse"`` pour la régression).
    min_samples_split : int
        Effectif minimal d'un nœud pour qu'on tente de le couper.
    min_samples_leaf : int
        Effectif minimal de chaque fils.
    max_leaves : int, optional
        Nombre maximal de feuilles (arbre de taille fixée).
    max_depth : int, optional
    """

    def __init__(self, task="classification", criterion=None, min_samples_split=5,
                 min_samples_leaf=1, max_leaves=None, max_depth=None, n_classes=None):
        self.task = task
        if criterion is None:
            criterion = "gini" if task == "classification" else "mse"
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.n_classes = n_classes

    # -- recherche de la meilleure coupure d'un nœud -------------------------
    def _node_cost(self, stats):
        """Coût (impureté totale) d'un nœud à partir de ses statistiques suffisantes."""
        if self.task == "regression":
            W, S1, S2 = stats
            return S2 - S1 ** 2 / W
        c = stats
        n = c.sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.criterion == "gini":
                return n - np.sum(c ** 2, axis=-1) / n
            if self.criterion == "entropy":
                q = c / n[..., None]
                return -np.sum(np.where(c > 0, c * np.log(np.where(q > 0, q, 1)), 0.0), axis=-1)
        raise ValueError(self.criterion)

    def _best_split(self, idx):
        n = len(idx)
        msl = self.min_samples_leaf
        if n < self.min_samples_split or n < 2 * msl:
            return None
        Xn = self.X_[idx]
        order = np.argsort(Xn, axis=0, kind="stable")
        xs = np.take_along_axis(Xn, order, axis=0)
        w = self.w_[idx]
        if self.task == "regression":
            y = self.y_[idx]
            W = np.cumsum(w[order], axis=0)[:-1]
            S1 = np.cumsum((w * y)[order], axis=0)[:-1]
            S2 = np.cumsum((w * y * y)[order], axis=0)[:-1]
            Wt, S1t, S2t = W[-1] + w[order[-1]], S1[-1] + (w * y)[order[-1]], \
                S2[-1] + (w * y * y)[order[-1]]
            with np.errstate(divide="ignore", invalid="ignore"):
                cost = (S2 - S1 ** 2 / W) + ((S2t - S2) - (S1t - S1) ** 2 / (Wt - W))
            parent = S2t[0] - S1t[0] ** 2 / Wt[0]
        else:
            Yk = self.Y1_[idx] * w[:, None]  # (n, K)
            C = np.cumsum(Yk[order], axis=0)  # (n, p, K)
            tot = C[-1, 0]
            L = C[:-1]
            R = tot - L
            cost = self._node_cost(L) + self._node_cost(R)
            parent = float(self._node_cost(tot))
        valid = xs[1:] > xs[:-1]
        if msl > 1:
            pos = np.arange(1, n)[:, None]
            valid &= (pos >= msl) & (n - pos >= msl)
        cost = np.where(valid & np.isfinite(cost), cost, np.inf)
        k = np.argmin(cost)
        i, j = np.unravel_index(k, cost.shape)
        if not np.isfinite(cost[i, j]):
            return None
        gain = parent - cost[i, j]
        thr = 0.5 * (xs[i, j] + xs[i + 1, j])
        left = idx[Xn[:, j] <= thr]
        right = idx[Xn[:, j] > thr]
        return gain, j, thr, left, right

    def _new_node(self, idx, depth, parent):
        w = self.w_[idx]
        node = len(self.feature)
        self.feature.append(-1)
        self.threshold.append(np.nan)
        self.left.append(-1)
        self.right.append(-1)
        self.parent.append(parent)
        self.depth.append(depth)
        self.n_node.append(len(idx))
        self.w_node.append(float(w.sum()))
        if self.task == "regression":
            y = self.y_[idx]
            m = _wmean(y, w)
            self.value.append(m)
            self.risk.append(float(np.sum(w * (y - m) ** 2)))
        else:
            c = (self.Y1_[idx] * w[:, None]).sum(0)
            self.value.append(c / c.sum())
            self.counts.append((self.Y1_[idx]).sum(0))
            self.risk.append(float(c.sum() - c.max()))
        return node

    def fit(self, X, y, sample_weight=None):
        """Fait croître l'arbre sur ``(X, y)``."""
        self.X_ = np.asarray(X, dtype=float)
        N, self.p_ = self.X_.shape
        self.w_ = np.ones(N) if sample_weight is None else np.asarray(sample_weight, float)
        if self.task == "classification":
            y = np.asarray(y).astype(int)
            K = self.n_classes or int(y.max()) + 1
            self.n_classes_ = K
            self.Y1_ = np.eye(K)[y]
        self.y_ = np.asarray(y, dtype=float)
        self.feature, self.threshold, self.left, self.right = [], [], [], []
        self.parent, self.depth, self.n_node, self.w_node = [], [], [], []
        self.value, self.risk, self.counts = [], [], []
        root = self._new_node(np.arange(N), 0, -1)
        heap, cnt = [], 0

        def push(node, idx):
            nonlocal cnt
            if self.max_depth is not None and self.depth[node] >= self.max_depth:
                return
            if self.risk[node] <= 1e-12:
                return
            s = self._best_split(idx)
            if s is None or s[0] <= 1e-12:
                return
            heapq.heappush(heap, (-s[0], cnt, node, s))
            cnt += 1

        push(root, np.arange(N))
        n_leaves = 1
        while heap and (self.max_leaves is None or n_leaves < self.max_leaves):
            _, _, node, (gain, j, thr, li, ri) = heapq.heappop(heap)
            self.feature[node], self.threshold[node] = j, thr
            d = self.depth[node] + 1
            a = self._new_node(li, d, node)
            b = self._new_node(ri, d, node)
            self.left[node], self.right[node] = a, b
            n_leaves += 1
            push(a, li)
            push(b, ri)
        for name in ("feature", "left", "right", "parent", "depth", "n_node"):
            setattr(self, name, np.array(getattr(self, name), dtype=int))
        for name in ("threshold", "w_node", "risk"):
            setattr(self, name, np.array(getattr(self, name), dtype=float))
        self.value = np.array(self.value, dtype=float)
        if self.task == "classification":
            self.counts = np.array(self.counts)
        self.is_leaf_ = self.left < 0
        del self.X_, self.y_, self.w_
        if self.task == "classification":
            del self.Y1_
        return self

    # -- prédiction -----------------------------------------------------------
    def apply(self, X, is_leaf=None):
        """Indice du nœud terminal atteint par chaque observation.

        ``is_leaf`` (masque booléen) permet d'utiliser un sous-arbre élagué.
        """
        X = np.asarray(X, dtype=float)
        leaf = self.is_leaf_ if is_leaf is None else is_leaf
        node = np.zeros(len(X), dtype=int)
        active = ~leaf[node]
        while active.any():
            nd = node[active]
            go_left = X[active, self.feature[nd]] <= self.threshold[nd]
            node[active] = np.where(go_left, self.left[nd], self.right[nd])
            active = ~leaf[node]
        return node

    def predict_value(self, X, is_leaf=None):
        """Valeur du nœud terminal (moyenne, ou vecteur de proportions de classes)."""
        return self.value[self.apply(X, is_leaf)]

    def predict_proba(self, X, is_leaf=None):
        return self.predict_value(X, is_leaf)

    def predict(self, X, is_leaf=None):
        v = self.predict_value(X, is_leaf)
        if self.task == "classification":
            return np.argmax(v, axis=1)
        return v

    def n_leaves(self, is_leaf=None):
        """Nombre de feuilles du (sous-)arbre."""
        leaf = self.is_leaf_ if is_leaf is None else is_leaf
        return int(np.sum(leaf & self._reachable(leaf)))

    def _reachable(self, is_leaf):
        reach = np.zeros(len(self.feature), dtype=bool)
        reach[0] = True
        for t in range(len(self.feature)):  # les fils ont un indice > parent
            if reach[t] and not is_leaf[t]:
                reach[self.left[t]] = reach[self.right[t]] = True
        return reach

    # -- élagage coût-complexité ---------------------------------------------
    def _subtree_stats(self, is_leaf):
        """Risque :math:`R(T_t)` et nombre de feuilles :math:`|T_t|` de chaque branche."""
        n = len(self.feature)
        R = self.risk.copy()
        L = np.ones(n)
        for t in range(n - 1, -1, -1):
            if not is_leaf[t]:
                R[t] = R[self.left[t]] + R[self.right[t]]
                L[t] = L[self.left[t]] + L[self.right[t]]
        return R, L

    def pruning_path(self):
        """Suite des sous-arbres emboîtés de l'élagage du maillon le plus faible.

        Returns
        -------
        alphas : ndarray
            Seuils croissants :math:`0=\\alpha_0<\\alpha_1<\\dots` ; le sous-arbre
            ``trees[k]`` minimise :math:`C_\\alpha(T)` (éq. (9.16)) pour
            :math:`\\alpha\\in[\\alpha_k,\\alpha_{k+1})`.
        trees : list of ndarray (bool)
            Masques « est une feuille » de chaque sous-arbre.
        """
        is_leaf = self.is_leaf_.copy()
        reach = self._reachable(is_leaf)
        alphas, trees = [], []
        first = True
        while True:
            R, L = self._subtree_stats(is_leaf)
            internal = (~is_leaf) & reach
            if not internal.any():
                if first:
                    alphas.append(0.0)
                    trees.append(is_leaf.copy())
                break
            g = np.full(len(R), np.inf)
            g[internal] = (self.risk[internal] - R[internal]) / (L[internal] - 1)
            a = max(g.min(), 0.0)
            if first and a > 1e-12:
                alphas.append(0.0)
                trees.append(is_leaf.copy())
            first = False
            cut = internal & (g <= a + 1e-10 * max(1.0, abs(a)))
            is_leaf = is_leaf | cut
            reach = self._reachable(is_leaf)
            if alphas and abs(alphas[-1] - a) <= 1e-12 and len(trees) > 0 and a <= 1e-12:
                trees[-1] = is_leaf.copy()
            else:
                alphas.append(a)
                trees.append(is_leaf.copy())
            if is_leaf[0]:
                break
        return np.array(alphas), trees

    def subtree(self, alpha, path=None):
        """Masque du sous-arbre :math:`T_\\alpha` (plus petit minimiseur de :math:`C_\\alpha`)."""
        alphas, trees = self.pruning_path() if path is None else path
        k = np.searchsorted(alphas, alpha, side="right") - 1
        return trees[max(k, 0)]

    def describe(self, names=None, is_leaf=None, classes=None, digits=3):
        """Représentation textuelle (indentée) de l'arbre."""
        leaf = self.is_leaf_ if is_leaf is None else is_leaf
        names = names or ["x%d" % (j + 1) for j in range(self.p_)]
        lines = []

        def lab(t):
            if self.task == "regression":
                return "moyenne=%.*f" % (digits, self.value[t])
            k = int(np.argmax(self.value[t]))
            c = classes[k] if classes else str(k)
            return "%s (n=%d, p=%s)" % (c, self.n_node[t],
                                        np.array2string(self.value[t], precision=2))

        def rec(t, ind):
            if leaf[t]:
                lines.append("  " * ind + "-> " + lab(t))
                return
            f, s = names[self.feature[t]], self.threshold[t]
            lines.append("  " * ind + "%s < %.*g" % (f, digits + 1, s))
            rec(self.left[t], ind + 1)
            lines.append("  " * ind + "%s > %.*g" % (f, digits + 1, s))
            rec(self.right[t], ind + 1)

        rec(0, 0)
        return "\n".join(lines)


def cart_cv(X, y, alphas, n_folds=10, rng=None, sample_weight=None, **tree_kw):
    """Validation croisée de l'élagage coût-complexité.

    Pour chaque pli, on fait croître un arbre maximal sur les autres plis, on
    calcule sa suite d'élagage, et l'on évalue sur le pli retenu le sous-arbre
    :math:`T_\\alpha` pour chaque valeur de ``alphas``.

    Returns
    -------
    errors : ndarray, shape (n_folds, len(alphas))
        Erreur de classement (classification) ou erreur quadratique moyenne
        (régression) sur chaque pli.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    N = len(y)
    rng = np.random.RandomState(0) if rng is None else rng
    folds = rng.permutation(N) % n_folds
    errors = np.empty((n_folds, len(alphas)))
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        sw = None if sample_weight is None else sample_weight[tr]
        t = ArbreCART(**tree_kw).fit(X[tr], y[tr], sample_weight=sw)
        path = t.pruning_path()
        for a_i, a in enumerate(alphas):
            pred = t.predict(X[te], t.subtree(a, path))
            if t.task == "classification":
                errors[f, a_i] = np.mean(pred != y[te])
            else:
                errors[f, a_i] = np.mean((pred - y[te]) ** 2)
    return errors


# ---------------------------------------------------------------------------
# PRIM
# ---------------------------------------------------------------------------


def in_box(X, box):
    """Indicatrice d'appartenance à la boîte ``box = (lo, hi)`` (bornes incluses)."""
    lo, hi = box
    return np.all((X >= lo) & (X <= hi), axis=1)


def prim_peel(X, y, alpha=0.1, min_count=10, mask=None):
    """Phase descendante de PRIM (« épluchage », étapes 1-3 de l'algorithme 9.3).

    À chaque étape on retire, sur une face de la boîte, la fraction ``alpha``
    des observations ayant les plus petites (ou les plus grandes) valeurs d'une
    variable, en choisissant l'épluchage qui maximise la moyenne de ``y`` dans
    la boîte restante. En présence d'ex-aequo (variables discrètes ou
    nombreuses valeurs nulles), un épluchage peut retirer plus que la fraction
    prévue ; on compare alors les candidats par le gain de moyenne *par
    observation retirée*, :math:`(\bar y_{\text{nouv}} - \bar y)/n_{\text{retirés}}`,
    comme le proposent Friedman et Fisher (1999). Sans ex-aequo, ce critère
    revient à maximiser la moyenne de la boîte restante.

    Parameters
    ----------
    X : ndarray, shape (N, p)
    y : ndarray, shape (N,)
    alpha : float
        Fraction épluchée à chaque étape.
    min_count : int
        On s'arrête quand la boîte contient moins de ``min_count`` points.
    mask : ndarray of bool, optional
        Observations disponibles (les autres sont ignorées).

    Returns
    -------
    boxes : list of (lo, hi)
        Suite des boîtes (la première est la boîte maximale).
    means, counts : ndarray
        Moyenne de ``y`` et effectif dans chaque boîte.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    avail = np.ones(len(y), bool) if mask is None else mask.copy()
    lo, hi = np.full(p, -np.inf), np.full(p, np.inf)
    inside = avail.copy()
    boxes, means, counts = [(lo.copy(), hi.copy())], [y[inside].mean()], [inside.sum()]
    while inside.sum() >= min_count:
        Xi, yi = X[inside], y[inside]
        n = len(yi)
        k = max(1, int(np.ceil(alpha * n)))
        cur = yi.mean()
        best = None
        for j in range(p):
            xs = np.sort(Xi[:, j])
            # face basse : on retire tout x <= v, v = k-ième plus petite valeur
            v = xs[k - 1]
            keep = Xi[:, j] > v
            if 0 < keep.sum() < n:
                m = (yi[keep].mean() - cur) / (n - keep.sum())
                if best is None or m > best[0]:
                    nxt = xs[xs > v][0]
                    best = (m, j, "lo", 0.5 * (v + nxt), keep.sum())
            v = xs[n - k]
            keep = Xi[:, j] < v
            if 0 < keep.sum() < n:
                m = (yi[keep].mean() - cur) / (n - keep.sum())
                if best is None or m > best[0]:
                    prv = xs[xs < v][-1]
                    best = (m, j, "hi", 0.5 * (v + prv), keep.sum())
        if best is None or best[4] < min_count:
            break
        m, j, side, s, _ = best
        if side == "lo":
            lo[j] = s
        else:
            hi[j] = s
        inside = avail & in_box(X, (lo, hi))
        boxes.append((lo.copy(), hi.copy()))
        means.append(y[inside].mean())
        counts.append(inside.sum())
    return boxes, np.array(means), np.array(counts)
