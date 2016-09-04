"""Outils du chapitre 7 : évaluation et sélection de modèles.

Ce module regroupe les briques réutilisées dans les notebooks du chapitre 7 :

* fonctions de perte et lisseurs linéaires (moindres carrés, k plus proches
  voisins, splines de lissage) ;
* critères analytiques (Cp, AIC, BIC, GCV) et nombre effectif de paramètres ;
* validation croisée à K blocs et estimateurs bootstrap de l'erreur
  (bootstrap naïf, bootstrap « leave-one-out », .632 et .632+) ;
* sélection de sous-ensembles (meilleur sous-ensemble, pas à pas ascendant) ;
* régression logistique par Newton-Raphson et base de splines cubiques
  naturelles (exemple des phonèmes) ;
* construction explicite de la famille {I(sin(alpha x) > 0)} qui pulvérise
  les points 10^{-1}, ..., 10^{-l} (dimension VC infinie).
"""

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Pertes
# ---------------------------------------------------------------------------

def squared_loss(y, yhat):
    """Perte quadratique ``(y - yhat)^2``, élément par élément."""
    return (np.asarray(y, float) - np.asarray(yhat, float)) ** 2
