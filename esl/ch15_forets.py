"""Forêts aléatoires - chapitre 15 de ESL.

Ce module contient une implémentation pédagogique (en numpy pur) :

* d'un arbre de régression CART « aléatoire » : à chaque nœud, seules ``m``
  variables tirées au hasard sont candidates à la coupure (algorithme 15.1) ;
* d'une forêt aléatoire (régression ou classification binaire) construite sur
  des échantillons bootstrap ;
* des outils associés : erreur *out-of-bag* (OOB), importance des variables
  (Gini et permutation), matrice de proximité, positionnement multidimensionnel
  classique ;
* des formules fermées et des simulateurs utilisés dans le chapitre et ses
  exercices (formule (15.1), modèle (15.8), sphères emboîtées, modèle
  (15.11) de Mease et Wyner, etc.).

Pour la classification binaire (y dans {0, 1}), l'arbre est construit comme un
arbre de régression sur la variable 0/1 : la réduction de la somme des carrés
des écarts coïncide alors (au facteur 1/2 près) avec la réduction de l'indice
de Gini pondéré par les effectifs, donc les coupures choisies sont exactement
celles du critère de Gini.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Noms des 57 variables des données spam (ordre des colonnes A.1 ... A.57)
# ---------------------------------------------------------------------------

NOMS_SPAM = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference",
    ";", "(", "[", "!", "$", "#", "CAPAVE", "CAPMAX", "CAPTOT",
]


def separer_spam(df, n_test=1536, seed=0):
    """Sépare les données spam en apprentissage / test.

    Le livre utilise un ensemble test de 1536 courriels ; le fichier fourni ne
    contient pas l'indicateur correspondant, on tire donc une séparation
    aléatoire de même taille (graine fixée).

    Parameters
    ----------
    df : pandas.DataFrame
        Sortie de ``esl.datasets.load_spam()``.
    n_test : int
        Taille de l'ensemble test.
    seed : int
        Graine du générateur ``RandomState``.

    Returns
    -------
    Xtr, ytr, Xte, yte : ndarray
    """
    rng = np.random.RandomState(seed)
    X = df.drop(columns="spam").to_numpy(float)
    y = df["spam"].to_numpy(int)
    perm = rng.permutation(len(y))
    te, tr = perm[:n_test], perm[n_test:]
    return X[tr], y[tr], X[te], y[te]


# ---------------------------------------------------------------------------
# Formules fermées
# ---------------------------------------------------------------------------

def variance_moyenne_correlee(sigma2, rho, B):
    """Variance de la moyenne de B variables de variance ``sigma2`` et de
    corrélation deux à deux ``rho`` (formule (15.1)) :
    ``rho * sigma2 + (1 - rho) * sigma2 / B``."""
    B = np.asarray(B, dtype=float)
    return rho * sigma2 + (1.0 - rho) * sigma2 / B


def rho_minimal(B):
    """Plus petite corrélation commune admissible pour B variables
    équicorrélées : ``-1 / (B - 1)`` (exercice 15.1)."""
    return -1.0 / (np.asarray(B, dtype=float) - 1.0)


def correlation_moyennes_bootstrap(n):
    """Corrélation entre deux moyennes bootstrap d'un même échantillon de
    taille n : ``n / (2n - 1)`` (exercice 15.4)."""
    n = np.asarray(n, dtype=float)
    return n / (2.0 * n - 1.0)


def variance_moyenne_bootstrap(sigma2, n):
    """Variance (non conditionnelle) d'une moyenne bootstrap :
    ``(2n - 1) sigma2 / n**2`` (exercice 15.4)."""
    return (2.0 * n - 1.0) * sigma2 / n ** 2


def proba_variable_pertinente(p, J, m):
    """Probabilité qu'au moins une des ``J`` variables pertinentes figure parmi
    les ``m`` candidates tirées sans remise parmi ``p`` (loi hypergéométrique,
    figure 15.7)."""
    from scipy.special import comb
    return 1.0 - comb(p - J, m) / comb(p, m)


# ---------------------------------------------------------------------------
# Simulateurs
# ---------------------------------------------------------------------------

def simuler_spheres(N, p=10, rng=None):
    """Modèle des « sphères emboîtées » (éq. (10.2) du livre).

    ``X`` gaussien standard dans R^p et ``Y = 1`` si ``sum X_j^2`` dépasse la
    médiane de la loi du khi-deux à p degrés de liberté, ``Y = 0`` sinon.
    L'erreur de Bayes est nulle.
    """
    from scipy.stats import chi2
    rng = np.random.RandomState(0) if rng is None else rng
    X = rng.randn(N, p)
    y = (np.sum(X ** 2, axis=1) > chi2.ppf(0.5, p)).astype(int)
    return X, y


def simuler_additif(N, p=50, rng=None):
    """Modèle (15.8) : ``Y = (1/sqrt(50)) sum_j X_j + eps``, tout gaussien
    standard. Retourne ``(X, y, mu)`` où ``mu`` est la vraie moyenne."""
    rng = np.random.RandomState(0) if rng is None else rng
    X = rng.randn(N, p)
    mu = X.sum(axis=1) / np.sqrt(p)
    return X, mu + rng.randn(N), mu


def simuler_mease_wyner(N, p, J=2, q=0.15, rng=None):
    """Modèle (15.11) : ``X ~ U[0,1]^p`` et
    ``P(Y=1|X) = q + (1-2q) 1{sum_{j<=J} X_j > J/2}``."""
    rng = np.random.RandomState(0) if rng is None else rng
    X = rng.rand(N, p)
    s = (X[:, :J].sum(axis=1) > J / 2.0)
    prob = q + (1 - 2 * q) * s
    y = (rng.rand(N) < prob).astype(int)
    return X, y


def simuler_melange(means, N, rng=None):
    """Tire N points du modèle « mixture » (chapitre 2) : la classe est tirée
    à pile ou face, puis un centre parmi les 10 de la classe (lignes 0-9 de
    ``means`` pour la classe 0, 10-19 pour la classe 1), puis
    ``x ~ N(centre, I/5)``."""
    rng = np.random.RandomState(0) if rng is None else rng
    y = rng.randint(0, 2, N)
    k = rng.randint(0, 10, N) + 10 * y
    X = means[k] + rng.randn(N, 2) / np.sqrt(5.0)
    return X, y


# ---------------------------------------------------------------------------
# Arbre CART aléatoire
# ---------------------------------------------------------------------------

def _meilleure_coupure(x, y, nmin):
    """Meilleure coupure ``x <= s`` pour une variable : maximise la réduction
    de la somme des carrés des écarts. Retourne ``(gain, seuil)`` ou ``None``."""
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o]
    n = len(ys)
    cs = np.cumsum(ys)
    S = cs[-1]
    nl = np.arange(1, n)
    sl = cs[:-1]
    nr = n - nl
    sr = S - sl
    valide = (xs[:-1] < xs[1:]) & (nl >= nmin) & (nr >= nmin)
    if not valide.any():
        return None
    gain = sl ** 2 / nl + sr ** 2 / nr - S ** 2 / n
    gain = np.where(valide, gain, -np.inf)
    i = int(np.argmax(gain))
    return gain[i], 0.5 * (xs[i] + xs[i + 1])


class ArbreAleatoire:
    """Arbre de régression CART avec sélection aléatoire de ``m`` variables
    candidates à chaque nœud (étape 1(b) de l'algorithme 15.1).

    Parameters
    ----------
    m : int or None
        Nombre de variables candidates par nœud (``None`` : toutes, arbre CART
        classique). Comme dans la plupart des logiciels, si les ``m`` premières
        variables tirées sont constantes dans le nœud, on continue le tirage
        jusqu'à en trouver ``m`` qui permettent une coupure.
    nmin : int
        Effectif minimal de chaque nœud terminal.
    rng : numpy.random.RandomState

    Attributes
    ----------
    variable, seuil, gauche, droite, valeur : ndarray
        Description de l'arbre (``variable = -1`` pour une feuille).
    importance_ : ndarray (p,)
        Somme, pour chaque variable, des réductions de la somme des carrés
        des écarts obtenues aux nœuds où elle sert à couper.
    """

    def __init__(self, m=None, nmin=1, rng=None):
        self.m = m
        self.nmin = nmin
        self.rng = np.random.RandomState(0) if rng is None else rng

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        N, p = X.shape
        m = p if self.m is None else min(self.m, p)
        var, seu, ga, dr, val = [], [], [], [], []
        self.importance_ = np.zeros(p)

        def nouveau_noeud(idx):
            var.append(-1)
            seu.append(0.0)
            ga.append(-1)
            dr.append(-1)
            val.append(y[idx].mean())
            return len(var) - 1

        pile = [(np.arange(N), nouveau_noeud(np.arange(N)))]
        while pile:
            idx, k = pile.pop()
            yk = y[idx]
            if len(idx) < 2 * self.nmin or np.ptp(yk) == 0:
                continue
            meilleur = None
            vues = 0
            for j in self.rng.permutation(p):
                res = _meilleure_coupure(X[idx, j], yk, self.nmin)
                if res is None:
                    continue
                vues += 1
                if meilleur is None or res[0] > meilleur[0]:
                    meilleur = (res[0], res[1], j)
                if vues >= m:
                    break
            if meilleur is None:
                continue
            gain, s, j = meilleur
            self.importance_[j] += gain
            g = idx[X[idx, j] <= s]
            d = idx[X[idx, j] > s]
            var[k], seu[k] = j, s
            ga[k] = nouveau_noeud(g)
            dr[k] = nouveau_noeud(d)
            pile.append((g, ga[k]))
            pile.append((d, dr[k]))
        self.variable = np.array(var)
        self.seuil = np.array(seu)
        self.gauche = np.array(ga)
        self.droite = np.array(dr)
        self.valeur = np.array(val)
        return self

    @property
    def n_feuilles(self):
        """Nombre de nœuds terminaux."""
        return int(np.sum(self.variable < 0))

    def apply(self, X):
        """Indice du nœud terminal atteint par chaque ligne de ``X``."""
        X = np.asarray(X, float)
        noeud = np.zeros(len(X), dtype=int)
        actifs = np.flatnonzero(self.variable[noeud] >= 0)
        while actifs.size:
            k = noeud[actifs]
            aller_gauche = X[actifs, self.variable[k]] <= self.seuil[k]
            noeud[actifs] = np.where(aller_gauche, self.gauche[k], self.droite[k])
            actifs = actifs[self.variable[noeud[actifs]] >= 0]
        return noeud

    def predict(self, X):
        """Valeur (moyenne de y) du nœud terminal atteint."""
        return self.valeur[self.apply(X)]


# ---------------------------------------------------------------------------
# Forêt aléatoire
# ---------------------------------------------------------------------------

class ForetAleatoire:
    """Forêt aléatoire (algorithme 15.1) pour la régression ou la
    classification binaire (y dans {0, 1}).

    Parameters
    ----------
    B : int
        Nombre d'arbres.
    m : int or None
        Nombre de variables candidates par nœud (``None`` : toutes, ce qui
        donne le *bagging*).
    nmin : int
        Effectif minimal des feuilles (1 en classification, 5 en régression
        selon les recommandations de la section 15.3).
    classification : bool
        Si vrai, la prédiction est le vote majoritaire des arbres.
    seed : int
        Graine.

    Attributes
    ----------
    arbres : list of ArbreAleatoire
    dans_sac : ndarray (B, N) de booléens
        ``dans_sac[b, i]`` vaut ``True`` si l'observation i figure dans
        l'échantillon bootstrap de l'arbre b.
    """

    def __init__(self, B=100, m=None, nmin=1, classification=False, seed=0):
        self.B, self.m, self.nmin = B, m, nmin
        self.classification = classification
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        N = len(y)
        rng = np.random.RandomState(self.seed)
        self.arbres = []
        self.dans_sac = np.zeros((self.B, N), dtype=bool)
        for b in range(self.B):
            boot = rng.randint(0, N, N)
            self.dans_sac[b, boot] = True
            arbre = ArbreAleatoire(self.m, self.nmin, rng).fit(X[boot], y[boot])
            self.arbres.append(arbre)
        return self

    def predictions_arbres(self, X):
        """Matrice (B, n) des prédictions de chaque arbre. En classification,
        il s'agit du vote 0/1 de chaque arbre."""
        P = np.array([a.predict(X) for a in self.arbres])
        if self.classification:
            P = (P > 0.5).astype(float)
        return P

    def predict(self, X):
        """Moyenne des arbres (régression) ou vote majoritaire (classification)."""
        moy = self.predictions_arbres(X).mean(axis=0)
        return (moy > 0.5).astype(int) if self.classification else moy

    @property
    def importance_gini(self):
        """Importance « Gini » : somme sur les arbres des réductions du
        critère, en pourcentage du maximum."""
        imp = np.sum([a.importance_ for a in self.arbres], axis=0)
        return 100 * imp / imp.max()

    def feuilles(self, X):
        """Matrice (B, n) des feuilles atteintes dans chaque arbre."""
        return np.array([a.apply(X) for a in self.arbres])


def erreur_oob_cumulee(P, dans_sac, y, classification=True):
    """Erreur OOB en fonction du nombre d'arbres.

    Parameters
    ----------
    P : ndarray (B, N)
        Prédictions de chaque arbre sur les données d'apprentissage (votes 0/1
        en classification).
    dans_sac : ndarray (B, N) de booléens
    y : ndarray (N,)
    classification : bool
        Taux d'erreur (vote majoritaire) si vrai, erreur quadratique sinon.

    Returns
    -------
    ndarray (B,)
        Erreur OOB calculée avec les b premiers arbres, sur les observations
        ayant au moins un arbre OOB.
    """
    hors = ~dans_sac
    somme = np.cumsum(P * hors, axis=0)
    nb = np.cumsum(hors, axis=0)
    out = np.full(P.shape[0], np.nan)
    for b in range(P.shape[0]):
        ok = nb[b] > 0
        f = somme[b, ok] / nb[b, ok]
        if classification:
            out[b] = np.mean((f > 0.5).astype(int) != y[ok])
        else:
            out[b] = np.mean((f - y[ok]) ** 2)
    return out


def erreur_test_cumulee(P, y, classification=True):
    """Erreur test (vote majoritaire ou erreur quadratique) en fonction du
    nombre d'arbres, à partir de la matrice (B, n) des prédictions des arbres."""
    f = np.cumsum(P, axis=0) / np.arange(1, P.shape[0] + 1)[:, None]
    if classification:
        return np.mean((f > 0.5).astype(int) != y[None, :], axis=1)
    return np.mean((f - y[None, :]) ** 2, axis=1)
