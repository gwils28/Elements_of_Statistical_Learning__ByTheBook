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
