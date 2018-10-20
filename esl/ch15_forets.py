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
