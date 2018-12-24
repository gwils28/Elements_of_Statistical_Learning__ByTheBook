"""Outils du chapitre 16 (apprentissage ensembliste).

Contenu :

* régression « forward stagewise » incrémentale (algorithme 16.1 du livre) ;
* lasso par descente de coordonnées (avec chemin de régularisation) ;
* régression ridge paramétrée par les degrés de liberté effectifs ;
* données gaussiennes corrélées par blocs (section 16.2.3, exercice 16.1) ;
* longueur d'arc L1 d'un chemin de coefficients (exercice 16.2) ;
* AdaBoost discret avec rétrécissement et marge L1 (figure 16.5) ;
* génération d'ensembles ISLE (algorithme 16.2) et post-traitement lasso (16.9) ;
* extraction de règles à partir d'arbres (16.14) et régression logistique (IRLS).

Les arbres élémentaires sont construits avec scikit-learn (la construction
gloutonne des arbres a été recodée au chapitre 9 ; ici seul compte leur rôle de
fonctions de base).
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------

def standardize(X, mean=None, sd=None):
    """Centre et réduit les colonnes de ``X``.

    Parameters
    ----------
    X : ndarray (N, p)
    mean, sd : ndarray (p,), optionnels
        Moyennes et écarts-types à utiliser (par ex. ceux de l'échantillon
        d'apprentissage pour transformer un échantillon test).

    Returns
    -------
    Xs, mean, sd : ndarray
        Matrice standardisée, moyennes et écarts-types utilisés
        (les colonnes constantes reçoivent ``sd = 1``).
    """
    X = np.asarray(X, float)
    if mean is None:
        mean = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
    return (X - mean) / sd, mean, sd
