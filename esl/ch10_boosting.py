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
