"""Chapitre 10 - Boosting et arbres additifs.

Implémentations explicites (numpy) utilisées dans les notebooks du chapitre :

* :class:`RegressionTree` - arbre de régression par moindres carrés (pondérés),
  construit « au meilleur d'abord » jusqu'à ``n_leaves`` feuilles (les arbres à
  :math:`J` nœuds terminaux du livre) ; il sert aussi d'arbre de classification
  pondéré lorsque la cible vaut :math:`\\pm 1` (le critère des moindres carrés
  coïncide alors avec l'indice de Gini pondéré) ;
* :class:`AdaBoostM1` - algorithme 10.1 du livre (AdaBoost discret) ;
* :class:`GradientBoosting` - algorithme 10.3 (MART) pour les pertes
  quadratique, absolue, de Huber et la déviance binomiale, avec rétrécissement
  (éq. 10.41) et sous-échantillonnage ;
* :class:`GradientBoostingMulticlass` - algorithme 10.4 (K classes) ;
* fonctions de perte (fig. 10.4 et 10.5), importance relative des variables
  (éq. 10.42-10.43) et dépendance partielle (éq. 10.47-10.48), calculée soit par
  la moyenne empirique, soit par le parcours pondéré des arbres (exercice 10.11).
"""

import numpy as np

__all__ = [
    "simulate_nested_spheres", "SPAM_NAMES",
    "loss_misclassification", "loss_exponential", "loss_binomial_deviance",
    "loss_squared_margin", "loss_hinge", "loss_huber",
    "RegressionTree", "AdaBoostM1", "GradientBoosting",
    "GradientBoostingMulticlass", "partial_dependence_bruteforce",
]

#: Noms usuels des 57 prédicteurs du jeu spam (ordre des colonnes A.1 ... A.57).
SPAM_NAMES = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference",
    ";", "(", "[", "!", "$", "#", "CAPAVE", "CAPMAX", "CAPTOT",
]

CHI2_10_MEDIAN = 9.341817765591966  # médiane d'un khi-deux à 10 degrés de liberté


def simulate_nested_spheres(n, p=10, rng=None):
    """Simulation (10.2) du livre : sphères emboîtées.

    Les :math:`X_j` sont des gaussiennes centrées réduites indépendantes et
    :math:`Y = 1` si :math:`\\sum_j X_j^2 > \\chi^2_{10}(0.5)`, sinon :math:`-1`.

    Parameters
    ----------
    n : int
        Nombre d'observations.
    p : int
        Dimension (10 dans le livre ; le seuil est alors la médiane du khi-deux
        à 10 degrés de liberté, sinon la médiane du khi-deux à ``p`` degrés).
    rng : numpy.random.RandomState, optional

    Returns
    -------
    X : ndarray (n, p)
    y : ndarray (n,) à valeurs dans {-1, +1}
    """
    from scipy import stats
    rng = np.random.RandomState(0) if rng is None else rng
    X = rng.randn(n, p)
    thr = CHI2_10_MEDIAN if p == 10 else stats.chi2.median(p)
    y = np.where((X ** 2).sum(1) > thr, 1, -1)
    return X, y


# ---------------------------------------------------------------------------
# Fonctions de perte (en fonction de la marge y*f, y dans {-1,+1})
# ---------------------------------------------------------------------------

def loss_misclassification(m):
    """Perte 0-1 : :math:`I(yf < 0)` (on compte 1/2 en cas d'égalité ``m = 0``)."""
    m = np.asarray(m, float)
    return np.where(m < 0, 1.0, np.where(m == 0, 0.5, 0.0))


def loss_exponential(m):
    """Perte exponentielle (10.8) : :math:`e^{-yf}`."""
    return np.exp(-np.asarray(m, float))


def loss_binomial_deviance(m, scaled=True):
    """Déviance binomiale (10.18) : :math:`\\log(1+e^{-2yf})`.

    Si ``scaled``, divisée par :math:`\\log 2` pour passer par (0, 1) comme
    dans la figure 10.4.
    """
    v = np.logaddexp(0.0, -2.0 * np.asarray(m, float))
    return v / np.log(2.0) if scaled else v


def loss_squared_margin(m):
    """Perte quadratique écrite avec la marge : :math:`(y-f)^2 = (1-yf)^2`."""
    return (1.0 - np.asarray(m, float)) ** 2


def loss_hinge(m):
    """Perte « support vector » : :math:`(1-yf)_+`."""
    return np.maximum(0.0, 1.0 - np.asarray(m, float))


def loss_huber(r, delta):
    """Perte de Huber (10.23) en fonction du résidu :math:`r = y - f`."""
    a = np.abs(np.asarray(r, float))
    return np.where(a <= delta, a ** 2, 2 * delta * a - delta ** 2)


# ---------------------------------------------------------------------------
# Arbre de régression (moindres carrés pondérés), croissance best-first
# ---------------------------------------------------------------------------

class RegressionTree:
    """Arbre de régression à ``n_leaves`` feuilles (croissance « meilleur d'abord »).

    À chaque étape on coupe la feuille dont la meilleure coupure fait le plus
    baisser la somme des carrés pondérée

    .. math:: \\sum_i w_i (r_i - \\bar r_{R(x_i)})^2 .

    Les tris sont faits une seule fois (``presort``) ; la recherche de la
    meilleure coupure dans un nœud est entièrement vectorisée.

    Parameters
    ----------
    n_leaves : int
        Nombre :math:`J` de nœuds terminaux (2 = souche).
    min_leaf : int
        Nombre minimal d'observations dans une feuille.

    Attributes
    ----------
    feature, threshold, left, right : ndarray
        Description des nœuds (``feature == -1`` pour une feuille). Une
        observation va à gauche si ``x[feature] <= threshold``.
    value : ndarray
        Valeur constante de chaque nœud (moyenne pondérée des cibles).
    weight : ndarray
        Fraction (pondérée par le nombre d'observations) des données
        d'apprentissage passant par chaque nœud (utile pour la dépendance
        partielle).
    improvement : ndarray
        Gain :math:`\\hat\\imath^2_t` en somme des carrés pour chaque nœud
        interne (0 pour les feuilles), éq. (10.42).
    """

    def __init__(self, n_leaves=2, min_leaf=1):
        self.n_leaves = int(n_leaves)
        self.min_leaf = int(min_leaf)

    # -- outils ------------------------------------------------------------
    @staticmethod
    def presort(X):
        """Indices triant chaque colonne de ``X`` : tableau (N, p)."""
        return np.argsort(X, axis=0, kind="mergesort")

    def _best_split(self, X, order, mask, wr, w):
        n = int(mask.sum())
        if n < 2 * self.min_leaf:
            return None
        p = X.shape[1]
        MT = mask[order].T                   # (p, N)
        sub = order.T[MT].reshape(p, n)      # indices triés par variable, dans le nœud
        xs = self._XsT[MT].reshape(p, n)     # valeurs triées correspondantes
        cs = np.cumsum(wr[sub], axis=1)
        cw = np.cumsum(w[sub], axis=1)
        S, W = cs[0, -1], cw[0, -1]
        csl, cwl = cs[:, :-1], cw[:, :-1]
        cwr = W - cwl
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = csl ** 2 / cwl + (S - csl) ** 2 / cwr - S ** 2 / W
        k = np.arange(1, n)
        valid = (xs[:, :-1] < xs[:, 1:]) & (cwl > 0) & (cwr > 0)
        valid &= (k >= self.min_leaf) & (n - k >= self.min_leaf)
        gain = np.where(valid, gain, -np.inf)
        flat = int(np.argmax(gain))
        j, pos = divmod(flat, n - 1)
        g = gain[j, pos]
        if not np.isfinite(g) or g <= 1e-12 * max(1.0, abs(S ** 2 / W)):
            return None
        thr = 0.5 * (xs[j, pos] + xs[j, pos + 1])
        return g, j, thr

    # -- apprentissage -------------------------------------------------------
    def fit(self, X, r, sample_weight=None, order=None, mask=None):
        """Ajuste l'arbre aux cibles ``r`` (poids ``sample_weight``).

        Parameters
        ----------
        X : ndarray (N, p)
        r : ndarray (N,)
        sample_weight : ndarray (N,), optional
        order : ndarray (N, p), optional
            Résultat de :meth:`presort` (évite de retrier à chaque arbre).
        mask : ndarray (N,) de booléens, optional
            Observations utilisées (sous-échantillonnage) ; les autres sont
            ignorées.
        """
        X = np.asarray(X, float)
        r = np.asarray(r, float)
        N = X.shape[0]
        w = np.ones(N) if sample_weight is None else np.asarray(sample_weight, float)
        if mask is None:
            mask = np.ones(N, bool)
        w = np.where(mask, w, 0.0)
        order = self.presort(X) if order is None else order
        self._XsT = np.take_along_axis(X, order, axis=0).T
        wr = w * r
        W0 = w.sum()
        nodes = dict(feature=[-1], threshold=[0.0], left=[-1], right=[-1],
                     value=[wr.sum() / W0], weight=[mask.sum()], improvement=[0.0])
        members = {0: mask.copy()}
        cand = {}
        s = self._best_split(X, order, mask, wr, w) if self.n_leaves > 1 else None
        if s is not None:
            cand[0] = s
        n_leaves = 1
        while n_leaves < self.n_leaves and cand:
            t = max(cand, key=lambda k: cand[k][0])
            g, j, thr = cand.pop(t)
            m = members.pop(t)
            go_left = X[:, j] <= thr
            for side, mm in (("left", m & go_left), ("right", m & ~go_left)):
                idx = len(nodes["feature"])
                nodes[side][t] = idx
                for key, val in (("feature", -1), ("threshold", 0.0), ("left", -1),
                                 ("right", -1), ("improvement", 0.0)):
                    nodes[key].append(val)
                nodes["value"].append(wr[mm].sum() / w[mm].sum())
                nodes["weight"].append(mm.sum())
                members[idx] = mm
                # inutile de chercher une coupure si l'arbre est déjà complet
                if n_leaves + 1 < self.n_leaves:
                    s = self._best_split(X, order, mm, wr, w)
                    if s is not None:
                        cand[idx] = s
            nodes["feature"][t] = j
            nodes["threshold"][t] = thr
            nodes["improvement"][t] = g
            n_leaves += 1
        for key in ("feature", "left", "right"):
            setattr(self, key, np.array(nodes[key], int))
        for key in ("threshold", "value", "improvement"):
            setattr(self, key, np.array(nodes[key], float))
        cnt = np.array(nodes["weight"], float)
        self.weight = cnt / cnt[0]
        self.n_features_ = X.shape[1]
        del self._XsT
        return self

    # -- prédiction ------------------------------------------------------------
    def apply(self, X):
        """Indice de la feuille atteinte par chaque ligne de ``X``."""
        X = np.asarray(X, float)
        node = np.zeros(X.shape[0], int)
        rows = np.arange(X.shape[0])
        while True:
            inner = self.feature[node] >= 0
            if not inner.any():
                return node
            nd = node[inner]
            go_left = X[rows[inner], self.feature[nd]] <= self.threshold[nd]
            node[inner] = np.where(go_left, self.left[nd], self.right[nd])

    def predict(self, X):
        """Valeur de la feuille atteinte par chaque ligne de ``X``."""
        return self.value[self.apply(X)]

    @property
    def leaves(self):
        """Indices des nœuds terminaux."""
        return np.flatnonzero(self.feature < 0)

    def importance(self):
        """Importances au carré :math:`I_\\ell^2(T)` de l'éq. (10.42) (vecteur de taille p)."""
        imp = np.zeros(self.n_features_)
        inner = self.feature >= 0
        np.add.at(imp, self.feature[inner], self.improvement[inner])
        return imp

    def partial_dependence(self, XS, S, X=None):
        r"""Dépendance partielle de l'arbre sur les variables ``S`` (exercice 10.11).

        * ``X is None`` : parcours pondéré de Friedman. Si la coupure porte sur
          une variable de ``S`` on suit la branche indiquée ; sinon on descend
          des deux côtés, avec pour poids les fractions d'observations
          d'apprentissage de chaque fils. Aucune donnée n'est nécessaire, mais
          le résultat n'est qu'une approximation de (10.48).
        * ``X`` fourni : calcul exact de la moyenne empirique (10.48). Pour
          chaque feuille :math:`\ell`, on compte la fraction :math:`q_\ell`
          des :math:`x_{iC}` qui vérifient les contraintes portant sur les
          variables de :math:`C` le long du chemin, puis
          :math:`\bar f_S(x_S)=\sum_\ell q_\ell\, v_\ell\,
          I(x_S \text{ vérifie les contraintes de } S)`.

        Parameters
        ----------
        XS : ndarray (n, |S|)
            Valeurs des variables de ``S`` où évaluer la fonction.
        S : list of int
            Indices des variables.
        X : ndarray (N, p), optional
            Données d'apprentissage (méthode exacte).
        """
        XS = np.atleast_2d(np.asarray(XS, float))
        pos = {s: k for k, s in enumerate(S)}
        out = np.zeros(XS.shape[0])
        n = XS.shape[0]
        if X is not None:
            X = np.asarray(X, float)
            stack = [(0, np.arange(n), np.ones(X.shape[0], bool))]
        else:
            stack = [(0, np.arange(n), np.ones(n))]
        while stack:
            node, rows, wt = stack.pop()
            if rows.size == 0:
                continue
            f = self.feature[node]
            if f < 0:
                q = wt.mean() if X is not None else wt
                out[rows] += q * self.value[node]
                continue
            L, R = self.left[node], self.right[node]
            if f in pos:
                gl = XS[rows, pos[f]] <= self.threshold[node]
                if X is not None:
                    stack.append((L, rows[gl], wt))
                    stack.append((R, rows[~gl], wt))
                else:
                    stack.append((L, rows[gl], wt[gl]))
                    stack.append((R, rows[~gl], wt[~gl]))
            elif X is not None:
                cl = X[:, f] <= self.threshold[node]
                stack.append((L, rows, wt & cl))
                stack.append((R, rows, wt & ~cl))
            else:
                pl = self.weight[L] / self.weight[node]
                stack.append((L, rows, wt * pl))
                stack.append((R, rows, wt * (1 - pl)))
        return out


# ---------------------------------------------------------------------------
# AdaBoost.M1 (algorithme 10.1)
# ---------------------------------------------------------------------------

class AdaBoostM1:
    """AdaBoost discret (algorithme 10.1) avec des arbres à ``n_leaves`` feuilles.

    Le classifieur faible :math:`G_m` est un arbre de classification pondéré :
    arbre de moindres carrés pondérés sur :math:`y\\in\\{-1,1\\}` puis signe de
    la valeur de chaque feuille (vote majoritaire pondéré).

    Attributes
    ----------
    trees_ : list of RegressionTree
    alphas_ : ndarray
        :math:`\\alpha_m = \\log((1-\\mathrm{err}_m)/\\mathrm{err}_m)`.
    errs_ : ndarray
        Erreurs pondérées :math:`\\mathrm{err}_m`.
    """

    def __init__(self, n_estimators=100, n_leaves=2, min_leaf=1):
        self.n_estimators = n_estimators
        self.n_leaves = n_leaves
        self.min_leaf = min_leaf

    def fit(self, X, y):
        """Apprentissage ; ``y`` à valeurs dans {-1, +1}."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        N = len(y)
        order = RegressionTree.presort(X)
        w = np.full(N, 1.0 / N)
        self.trees_, alphas, errs = [], [], []
        self.weights_history_ = []
        for m in range(self.n_estimators):
            tree = RegressionTree(self.n_leaves, self.min_leaf).fit(X, y, w, order=order)
            tree.value = np.where(tree.value >= 0, 1.0, -1.0)
            G = tree.predict(X)
            miss = G != y
            err = np.clip(w[miss].sum() / w.sum(), 1e-12, 1 - 1e-12)
            alpha = np.log((1 - err) / err)
            w = w * np.exp(alpha * miss)
            w = w / w.sum()  # normalisation sans effet sur l'algorithme
            self.trees_.append(tree)
            alphas.append(alpha)
            errs.append(err)
            if m < 5:
                self.weights_history_.append(w.copy())
        self.alphas_ = np.array(alphas)
        self.errs_ = np.array(errs)
        return self

    def staged_decision_function(self, X):
        """Itérateur sur :math:`F_m(x)=\\sum_{k\\le m}\\alpha_k G_k(x)`, m = 1..M."""
        F = np.zeros(np.asarray(X).shape[0])
        for a, t in zip(self.alphas_, self.trees_):
            F = F + a * t.predict(X)
            yield F

    def decision_function(self, X):
        """:math:`F_M(x)=\\sum_m \\alpha_m G_m(x)`."""
        F = None
        for F in self.staged_decision_function(X):
            pass
        return F

    def predict(self, X):
        """:math:`G(x) = \\mathrm{signe}(F_M(x))` (égalité tranchée vers +1)."""
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def staged_errors(self, X, y):
        """Taux d'erreur après chaque itération (tableau de taille M)."""
        y = np.asarray(y)
        return np.array([np.mean(np.where(F >= 0, 1, -1) != y)
                         for F in self.staged_decision_function(X)])


# ---------------------------------------------------------------------------
# Gradient boosting (algorithme 10.3)
# ---------------------------------------------------------------------------

def _sigmoid(f):
    return 0.5 * (1.0 + np.tanh(0.5 * f))


def _weighted_median(v):
    return np.median(v) if v.size else 0.0


class GradientBoosting:
    """Gradient boosting d'arbres (algorithme 10.3, « MART »).

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'deviance'}
        Perte quadratique :math:`\\frac12(y-f)^2`, absolue :math:`|y-f|`, de
        Huber, ou déviance binomiale. Pour ``'deviance'``, ``y`` est codé 0/1 et
        :math:`f` est le log-rapport de cotes
        :math:`\\log\\frac{P(Y=1|x)}{P(Y=0|x)}`.
    n_estimators : int
        Nombre :math:`M` d'arbres.
    n_leaves : int
        Nombre :math:`J` de feuilles de chaque arbre.
    learning_rate : float
        Coefficient de rétrécissement :math:`\\nu` (éq. 10.41).
    subsample : float
        Fraction :math:`\\eta` d'observations tirées sans remise pour chaque
        arbre (gradient boosting stochastique).
    huber_alpha : float
        Quantile définissant :math:`\\delta_m` pour la perte de Huber.
    min_leaf : int
    random_state : int
    max_step : float
        Borne sur la valeur absolue du pas de Newton dans une feuille (déviance).
        Sans elle, une feuille ne contenant que des observations très mal
        classées et très « sûres » (:math:`p_i(1-p_i)\\approx 0`) recevrait une
        valeur gigantesque.
    """

    def __init__(self, loss="ls", n_estimators=100, n_leaves=6, learning_rate=0.1,
                 subsample=1.0, huber_alpha=0.9, min_leaf=1, random_state=0,
                 max_step=4.0):
        self.loss = loss
        self.n_estimators = n_estimators
        self.n_leaves = n_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.huber_alpha = huber_alpha
        self.min_leaf = min_leaf
        self.random_state = random_state
        self.max_step = max_step

    # -- briques dépendant de la perte ------------------------------------------
    def _init(self, y):
        if self.loss == "ls":
            return y.mean()
        if self.loss in ("lad", "huber"):
            return np.median(y)
        if self.loss == "deviance":
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        raise ValueError("perte inconnue : %s" % self.loss)

    def _neg_gradient(self, y, f):
        r = y - f
        if self.loss == "ls":
            return r
        if self.loss == "lad":
            return np.sign(r)
        if self.loss == "huber":
            self._delta = np.quantile(np.abs(r), self.huber_alpha)
            return np.where(np.abs(r) <= self._delta, r, self._delta * np.sign(r))
        return y - _sigmoid(f)  # déviance

    def _leaf_value(self, y, f, g):
        """Étape 2(c) : constante optimale (ou approchée) dans une feuille."""
        r = y - f
        if self.loss == "ls":
            return r.mean()
        if self.loss == "lad":
            return _weighted_median(r)
        if self.loss == "huber":
            med = np.median(r)
            d = r - med
            return med + np.mean(np.sign(d) * np.minimum(self._delta, np.abs(d)))
        p = _sigmoid(f)  # un pas de Newton–Raphson, borné (voir max_step)
        step = g.sum() / max((p * (1 - p)).sum(), 1e-12)
        return float(np.clip(step, -self.max_step, self.max_step))

    def loss_value(self, y, f):
        """Perte moyenne :math:`\\frac1N\\sum_i L(y_i, f(x_i))`.

        Pour ``'deviance'`` on renvoie :math:`-2\\times` la log-vraisemblance
        moyenne (déviance binomiale moyenne).
        """
        y = np.asarray(y, float)
        if self.loss == "ls":
            return np.mean((y - f) ** 2)
        if self.loss == "lad":
            return np.mean(np.abs(y - f))
        if self.loss == "huber":
            return np.mean(loss_huber(y - f, getattr(self, "_delta", 1.0)))
        return 2 * np.mean(np.logaddexp(0.0, f) - y * f)

    # -- apprentissage ---------------------------------------------------------
    def fit(self, X, y):
        """Ajuste le modèle :math:`f_M = f_0 + \\nu\\sum_m T_m`."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        N = len(y)
        rng = np.random.RandomState(self.random_state)
        order = RegressionTree.presort(X)
        self.f0_ = self._init(y)
        f = np.full(N, self.f0_)
        self.trees_ = []
        self.train_loss_ = np.empty(self.n_estimators)
        self.oob_improvement_ = np.full(self.n_estimators, np.nan)
        n_sub = int(round(self.subsample * N))
        for m in range(self.n_estimators):
            if self.subsample < 1:
                mask = np.zeros(N, bool)
                mask[rng.choice(N, n_sub, replace=False)] = True
            else:
                mask = np.ones(N, bool)
            g = self._neg_gradient(y, f)
            tree = RegressionTree(self.n_leaves, self.min_leaf).fit(
                X, g, order=order, mask=mask)
            leaf = tree.apply(X)
            for j in tree.leaves:
                inleaf = (leaf == j) & mask
                if inleaf.any():
                    tree.value[j] = self._leaf_value(y[inleaf], f[inleaf], g[inleaf])
                else:
                    tree.value[j] = 0.0
            tree.value *= self.learning_rate
            f_new = f + tree.value[leaf]
            if self.subsample < 1 and (~mask).any():
                oob = ~mask
                self.oob_improvement_[m] = (self.loss_value(y[oob], f[oob])
                                            - self.loss_value(y[oob], f_new[oob]))
            f = f_new
            self.trees_.append(tree)
            self.train_loss_[m] = self.loss_value(y, f)
        return self

    # -- prédiction ------------------------------------------------------------
    def staged_decision_function(self, X):
        """Itérateur sur :math:`f_m(x)`, m = 1..M."""
        X = np.asarray(X, float)
        f = np.full(X.shape[0], self.f0_)
        for t in self.trees_:
            f = f + t.predict(X)
            yield f

    def decision_function(self, X):
        """:math:`f_M(x)`."""
        X = np.asarray(X, float)
        return self.f0_ + sum(t.predict(X) for t in self.trees_)

    def predict(self, X):
        """Prédiction : :math:`f_M(x)` (régression) ou classe 0/1 (déviance)."""
        f = self.decision_function(X)
        return (f > 0).astype(int) if self.loss == "deviance" else f

    def predict_proba(self, X):
        """:math:`\\hat P(Y=1|x)` pour la déviance."""
        return _sigmoid(self.decision_function(X))

    def staged_scores(self, X, y, every=1):
        """Perte moyenne et taux d'erreur (déviance) sur ``(X, y)`` au fil des itérations.

        Returns
        -------
        iters : ndarray
        loss : ndarray
        err : ndarray (taux de mauvaise classification ; NaN en régression)
        """
        y = np.asarray(y, float)
        iters, L, E = [], [], []
        for m, f in enumerate(self.staged_decision_function(X), start=1):
            if m % every and m != len(self.trees_):
                continue
            iters.append(m)
            L.append(self.loss_value(y, f))
            E.append(np.mean((f > 0) != (y > 0.5)) if self.loss == "deviance" else np.nan)
        return np.array(iters), np.array(L), np.array(E)

    # -- interprétation --------------------------------------------------------
    def feature_importances(self):
        """Importance relative (10.43) : racine de la moyenne des :math:`I^2_\\ell(T_m)`, max = 100."""
        imp = np.mean([t.importance() for t in self.trees_], axis=0)
        imp = np.sqrt(imp)
        return 100 * imp / imp.max()

    def partial_dependence(self, XS, S, X=None):
        """Dépendance partielle :math:`\\bar f_S` par parcours des arbres (ex. 10.11).

        Voir :meth:`RegressionTree.partial_dependence` (``X`` fourni : exact).
        """
        XS = np.atleast_2d(np.asarray(XS, float))
        return self.f0_ + sum(t.partial_dependence(XS, S, X) for t in self.trees_)


def partial_dependence_bruteforce(predict, X, S, XS):
    """Dépendance partielle empirique (10.48) : :math:`\\frac1N\\sum_i f(x_S, x_{iC})`.

    Parameters
    ----------
    predict : callable
        Fonction :math:`x \\mapsto f(x)` vectorisée.
    X : ndarray (N, p)
        Données d'apprentissage (fournissent les :math:`x_{iC}`).
    S : list of int
    XS : ndarray (n, |S|)
    """
    XS = np.atleast_2d(np.asarray(XS, float))
    out = np.empty(XS.shape[0])
    Z = np.array(X, float, copy=True)
    for k, xs in enumerate(XS):
        Z[:, S] = xs
        out[k] = predict(Z).mean()
    return out
