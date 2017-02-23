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
