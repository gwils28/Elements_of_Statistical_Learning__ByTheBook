"""Méthodes de lissage par noyaux - chapitre 6 d'ESL.

Ce module regroupe les outils implémentés « à la main » dans les notebooks du
chapitre 6 :

* noyaux de lissage (Epanechnikov, tri-cube, gaussien) et poids
  :math:`K_\\lambda(x_0, x_i)` à fenêtre métrique ou à k plus proches voisins ;
* moyenne mobile des k-ppv, estimateur de Nadaraya-Watson ;
* régression polynomiale locale en dimension quelconque (monômes au choix),
  noyaux équivalents, matrice de lissage, critères LOO / GCV / :math:`C_p` ;
* régression logistique (globale et locale) par IRLS ;
* estimation de densité par noyau, classifieur par densités, Bayes naïf ;
* fonctions de base radiales, mélanges gaussiens (algorithme EM) ;
* analyse discriminante linéaire locale (exercice 6.12).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Noyaux
# ---------------------------------------------------------------------------


def epanechnikov(t):
    """Noyau d'Epanechnikov :math:`D(t) = \\tfrac34 (1 - t^2)` sur [-1, 1] (éq. 6.4).

    Parameters
    ----------
    t : array_like
        Distances réduites :math:`|x - x_0| / \\lambda`.

    Returns
    -------
    ndarray
        Valeurs de :math:`D(t)` (nulles hors de [-1, 1]). Intègre à 1.
    """
    t = np.asarray(t, dtype=float)
    return np.where(np.abs(t) <= 1, 0.75 * (1 - t ** 2), 0.0)


def tricube(t):
    """Noyau tri-cube :math:`D(t) = (1 - |t|^3)^3` sur [-1, 1] (éq. 6.6).

    Non normalisé : son intégrale vaut 81/70.
    """
    a = np.abs(np.asarray(t, dtype=float))
    return np.where(a <= 1, (1 - a ** 3) ** 3, 0.0)
