"""Outils du chapitre 14 : apprentissage non supervisé.

Ce module regroupe les implémentations « à la main » utilisées dans les
notebooks du chapitre 14 :

* règles d'association (codage disjonctif, algorithme Apriori, règles) ;
* apprentissage non supervisé vu comme supervisé (échantillon de référence,
  régression logistique pénalisée sur une base de splines, PRIM simplifié) ;
* classification : K-means, K-médoïdes, statistique Gap, classification
  hiérarchique ascendante, dissimilarités cophénétiques, indice de Rand ajusté,
  algorithme EM pour un mélange gaussien isotrope ;
* cartes auto-organisatrices (SOM) ;
* ACP, courbes principales, Procrustes et formes moyennes ;
* positionnement multidimensionnel (MDS classique et par moindres carrés) ;
* clustering spectral et ACP à noyau ;
* ACP parcimonieuse, factorisation en matrices positives (NMF), ACI (FastICA).

Toutes les fonctions prennent des tableaux numpy ; les observations sont en
lignes (matrice ``X`` de taille ``(N, p)``).
"""

from itertools import combinations

import numpy as np
import pandas as pd


def _rng(rng):
    """Renvoie un ``RandomState`` (graine 0 par défaut)."""
    if rng is None:
        return np.random.RandomState(0)
    if isinstance(rng, (int, np.integer)):
        return np.random.RandomState(rng)
    return rng


# ---------------------------------------------------------------------------
# 1. Règles d'association
# ---------------------------------------------------------------------------

#: Libellés des modalités du questionnaire « marketing » (tableau 14.1 du livre).
MODALITES_MARKETING = {
    "Income": {1: "<10k$", 2: "10-15k$", 3: "15-20k$", 4: "20-25k$", 5: "25-30k$",
               6: "30-40k$", 7: "40-50k$", 8: "50-75k$", 9: ">75k$"},
    "Sex": {1: "homme", 2: "femme"},
    "Marital": {1: "marié", 2: "en couple non marié", 3: "divorcé/séparé",
                4: "veuf", 5: "célibataire"},
    "Age": {1: "14-17", 2: "18-24", 3: "25-34", 4: "35-44", 5: "45-54",
            6: "55-64", 7: "65+"},
    "Edu": {1: "<= 8e année", 2: "9e-11e année", 3: "fin du secondaire",
            4: "1-3 ans d'université", 5: "diplômé université", 6: "3e cycle"},
    "Occupation": {1: "cadre/profession libérale", 2: "commercial",
                   3: "ouvrier/chauffeur", 4: "employé/service", 5: "au foyer",
                   6: "étudiant", 7: "militaire", 8: "retraité", 9: "sans emploi"},
    "Lived": {1: "<1 an", 2: "1-3 ans", 3: "4-6 ans", 4: "7-10 ans", 5: ">10 ans"},
    "Dual_Income": {1: "non marié", 2: "oui", 3: "non"},
    "Status": {1: "propriétaire", 2: "locataire", 3: "vit en famille"},
    "Home_Type": {1: "maison", 2: "copropriété", 3: "appartement",
                  4: "mobile home", 5: "autre"},
    "Ethnic": {1: "amérindien", 2: "asiatique", 3: "noir", 4: "indien",
               5: "hispanique", 6: "océanien", 7: "blanc", 8: "autre"},
    "Language": {1: "anglais", 2: "espagnol", 3: "autre"},
}

#: Variables ordinales / catégorielles du jeu « marketing » (tableau 14.1).
ORDINALES_MARKETING = ["Age", "Edu", "Income", "Lived", "Household", "Householdu18"]
CATEGORIELLES_MARKETING = ["Sex", "Marital", "Occupation", "Dual_Income",
                           "Status", "Home_Type", "Ethnic", "Language"]


def codage_disjonctif(df, ordinales, categorielles):
    """Code un tableau de variables discrètes en variables indicatrices.

    Comme dans la section 14.2.3 du livre : chaque variable ordinale est coupée
    à sa médiane (deux indicatrices ``x <= médiane`` et ``x > médiane``) et
    chaque variable catégorielle à ``k`` modalités donne ``k`` indicatrices.
    Si la médiane coïncide avec la valeur maximale, on coupe en
    ``x < médiane`` / ``x >= médiane`` pour que les deux indicatrices soient non vides.

    Parameters
    ----------
    df : pandas.DataFrame
        Données sans valeurs manquantes.
    ordinales, categorielles : list of str
        Noms des colonnes de chaque type.

    Returns
    -------
    Z : ndarray de booléens, forme (N, K)
        Matrice des « items ».
    noms : list of str
        Libellé de chaque item (par ex. ``"Age<=3"`` ou ``"Sex=2"``).
    variables : list of str
        Variable d'origine de chaque item.
    """
    cols, noms, variables = [], [], []
    for v in df.columns:
        x = df[v].to_numpy()
        if v in ordinales:
            med = np.median(x)
            if med < x.max():
                cols += [x <= med, x > med]
                noms += ["%s<=%g" % (v, med), "%s>%g" % (v, med)]
            else:  # médiane = maximum : on coupe juste en dessous
                cols += [x < med, x >= med]
                noms += ["%s<%g" % (v, med), "%s>=%g" % (v, med)]
            variables += [v, v]
        elif v in categorielles:
            for m in np.unique(x):
                cols.append(x == m)
                noms.append("%s=%g" % (v, m))
                variables.append(v)
    return np.column_stack(cols), noms, variables
