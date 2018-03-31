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


def apriori(Z, support_min, taille_max=None):
    """Algorithme Apriori (Agrawal et al., 1995) : ensembles d'items fréquents.

    On parcourt les tailles ``k = 1, 2, ...`` ; un ensemble de taille ``k`` n'est
    candidat que si tous ses sous-ensembles de taille ``k-1`` sont fréquents
    (propriété d'anti-monotonie du support).

    Parameters
    ----------
    Z : ndarray booléen (N, K)
        Matrice des items.
    support_min : float
        Seuil ``t`` sur le support (proportion d'observations), éq. (14.6).
    taille_max : int ou None
        Taille maximale des ensembles d'items.

    Returns
    -------
    dict
        ``{tuple trié d'indices d'items : support}``.
    """
    Z = np.asarray(Z, dtype=bool)
    N, K = Z.shape
    sup1 = Z.mean(axis=0)
    niveau = {(j,): Z[:, j] for j in range(K) if sup1[j] >= support_min}
    frequents = {k: float(v.mean()) for k, v in niveau.items()}
    taille = 1
    while niveau and (taille_max is None or taille < taille_max):
        cles = sorted(niveau)
        nouveau = {}
        for a in range(len(cles)):
            for b in range(a + 1, len(cles)):
                A, B = cles[a], cles[b]
                if A[:-1] != B[:-1]:
                    break  # les clés sont triées : plus de préfixe commun
                cand = A + (B[-1],)
                # élagage : tous les sous-ensembles de taille k-1 fréquents
                if any(sub not in niveau for sub in combinations(cand, taille)):
                    continue
                couv = niveau[A] & Z[:, B[-1]]
                s = couv.mean()
                if s >= support_min:
                    nouveau[cand] = couv
                    frequents[cand] = float(s)
        niveau = nouveau
        taille += 1
    return frequents


def regles_association(frequents, confiance_min=0.0, noms=None):
    """Construit les règles ``A => B`` à partir des ensembles fréquents.

    Pour chaque ensemble fréquent ``K`` (``|K| >= 2``) et chaque partition
    ``K = A ∪ B`` en deux parties non vides, on calcule

    * le support ``T(A => B) = T(K)`` ;
    * la confiance ``C(A => B) = T(K) / T(A)``, éq. (14.8) ;
    * le lift ``L(A => B) = C(A => B) / T(B)``.

    Returns
    -------
    pandas.DataFrame
        Colonnes ``antecedent``, ``consequent`` (tuples d'indices ou de noms),
        ``taille``, ``support``, ``confiance``, ``lift``.
    """
    lignes = []
    for K, sK in frequents.items():
        if len(K) < 2:
            continue
        for r in range(1, len(K)):
            for A in combinations(K, r):
                B = tuple(i for i in K if i not in A)
                conf = sK / frequents[A]
                if conf >= confiance_min:
                    lignes.append((A, B, len(K), sK, conf, conf / frequents[B]))
    df = pd.DataFrame(lignes, columns=["antecedent", "consequent", "taille",
                                       "support", "confiance", "lift"])
    if noms is not None:
        for c in ("antecedent", "consequent"):
            df[c] = df[c].apply(lambda t: tuple(noms[i] for i in t))
    return df


# ---------------------------------------------------------------------------
# 2. Non supervisé vu comme supervisé
# ---------------------------------------------------------------------------

def echantillon_reference(X, methode="uniforme", rng=None, n=None):
    """Tire un échantillon de la densité de référence ``g0`` (section 14.2.4).

    Parameters
    ----------
    X : ndarray (N, p)
    methode : {"uniforme", "permutation"}
        ``"uniforme"`` : loi uniforme sur le pavé contenant les données ;
        ``"permutation"`` : produit des lois marginales (14.15), obtenu en
        permutant indépendamment chaque colonne.
    n : int ou None
        Taille de l'échantillon (``N`` par défaut ; uniquement pour « uniforme »).
    """
    rng = _rng(rng)
    X = np.asarray(X, float)
    N, p = X.shape
    if methode == "uniforme":
        n = N if n is None else n
        lo, hi = X.min(axis=0), X.max(axis=0)
        return lo + (hi - lo) * rng.uniform(size=(n, p))
    if methode == "permutation":
        return np.column_stack([X[rng.permutation(N), j] for j in range(p)])
    raise ValueError(methode)


def base_splines_naturelles(x, noeuds):
    """Base des splines cubiques naturelles (éq. (5.4)-(5.5) du livre).

    Renvoie la matrice ``(N, K)`` des fonctions ``N_1 = 1, N_2 = x`` et
    ``N_{k+2} = d_k - d_{K-1}`` avec
    ``d_k(x) = ((x - ξ_k)_+^3 - (x - ξ_K)_+^3) / (ξ_K - ξ_k)``.
    """
    x = np.asarray(x, float)
    xi = np.asarray(noeuds, float)
    K = len(xi)

    def d(k):
        return (np.maximum(x - xi[k], 0) ** 3 - np.maximum(x - xi[-1], 0) ** 3) / (xi[-1] - xi[k])

    cols = [np.ones_like(x), x] + [d(k) - d(K - 2) for k in range(K - 2)]
    return np.column_stack(cols)


def regression_logistique(B, y, lam=0.0, n_iter=50, tol=1e-10):
    """Régression logistique pénalisée (ridge) par l'algorithme IRLS.

    Maximise ``sum_i [y_i f_i - log(1 + e^{f_i})] - lam/2 * ||beta[1:]||^2`` avec
    ``f = B beta`` (la première colonne de ``B``, la constante, n'est pas
    pénalisée).

    Returns
    -------
    beta : ndarray (q,)
    """
    B = np.asarray(B, float)
    y = np.asarray(y, float)
    q = B.shape[1]
    P = lam * np.eye(q)
    P[0, 0] = 0.0
    beta = np.zeros(q)
    for _ in range(n_iter):
        f = B @ beta
        mu = 1 / (1 + np.exp(-f))
        w = np.maximum(mu * (1 - mu), 1e-10)
        H = B.T @ (B * w[:, None]) + P
        g = B.T @ (y - mu) - P @ beta
        pas = np.linalg.solve(H, g)
        beta = beta + pas
        if np.max(np.abs(pas)) < tol:
            break
    return beta


def prim_peeling(X, y, categorielles=(), alpha=0.1, support_min=0.1):
    """Phase d'« épluchage » de PRIM (section 9.3), version simplifiée.

    On part de la boîte contenant toutes les données ; à chaque étape on
    considère, pour chaque variable ordinale, le retrait de la tranche des
    ``alpha`` plus petites (ou plus grandes) valeurs, et pour chaque variable
    catégorielle le retrait d'une modalité ; on garde le retrait qui maximise la
    moyenne de ``y`` dans la boîte restante. On s'arrête quand le support de la
    boîte passe sous ``support_min``.

    Parameters
    ----------
    X : ndarray (N, p)
        Variables (codes numériques).
    y : ndarray (N,)
        Réponse (0/1 pour la recherche de régions de forte densité).
    categorielles : iterable d'int
        Indices des colonnes catégorielles.

    Returns
    -------
    dict
        ``boite`` : liste de contraintes par variable (``("intervalle", lo, hi)``
        ou ``("modalites", ensemble)``) ; ``moyennes`` et ``supports`` : trajectoire
        de l'épluchage ; ``dedans`` : masque des observations dans la boîte finale.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, p = X.shape
    cat = set(categorielles)
    dedans = np.ones(N, bool)
    boite = []
    for j in range(p):
        if j in cat:
            boite.append(("modalites", set(np.unique(X[:, j]).tolist())))
        else:
            boite.append(("intervalle", X[:, j].min(), X[:, j].max()))
    moyennes, supports = [y.mean()], [1.0]
    while True:
        meilleur = None
        n_in = dedans.sum()
        for j in range(p):
            xj = X[dedans, j]
            if j in cat:
                for m in boite[j][1]:
                    reste = dedans & (X[:, j] != m)
                    if 0 < reste.sum() < n_in:
                        cand = (y[reste].mean(), j, ("modalites", boite[j][1] - {m}), reste)
                        if meilleur is None or cand[0] > meilleur[0]:
                            meilleur = cand
            else:
                lo, hi = np.quantile(xj, [alpha, 1 - alpha])
                for reste in (dedans & (X[:, j] > lo), dedans & (X[:, j] < hi)):
                    if 0 < reste.sum() < n_in:
                        xr = X[reste, j]
                        cand = (y[reste].mean(), j, ("intervalle", xr.min(), xr.max()), reste)
                        if meilleur is None or cand[0] > meilleur[0]:
                            meilleur = cand
        if meilleur is None or meilleur[3].sum() < support_min * N:
            break
        _, j, nouvelle, reste = meilleur
        boite[j] = nouvelle
        dedans = reste
        moyennes.append(y[dedans].mean())
        supports.append(dedans.mean())
    return {"boite": boite, "moyennes": np.array(moyennes),
            "supports": np.array(supports), "dedans": dedans}


# ---------------------------------------------------------------------------
# 3. Classification (clustering)
# ---------------------------------------------------------------------------

def distances_carrees(X, Y=None):
    """Matrice des distances euclidiennes au carré ``||x_i - y_j||^2``."""
    X = np.asarray(X, float)
    Y = X if Y is None else np.asarray(Y, float)
    D = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.maximum(D, 0)


def inertie_intra(X, labels, centres=None):
    """Somme des carrés intra-classes ``sum_k sum_{C(i)=k} ||x_i - xbar_k||^2``.

    Le critère ``W(C)`` du livre (14.31) pondère en plus chaque classe par son
    effectif ``N_k`` ; nous utilisons, comme la plupart des logiciels (et la
    figure 14.8), la somme simple ci-dessus, minimisée elle aussi par K-means.
    """
    X = np.asarray(X, float)
    tot = 0.0
    for k in np.unique(labels):
        Xk = X[labels == k]
        c = Xk.mean(0) if centres is None else centres[k]
        tot += ((Xk - c) ** 2).sum()
    return tot


def kmeans(X, K, n_init=10, max_iter=300, rng=None, centres_init=None):
    """K-means (algorithme 14.1, itérations de Lloyd).

    Alterne (i) l'affectation de chaque point au centre le plus proche (14.34)
    et (ii) le remplacement de chaque centre par la moyenne de sa classe
    (14.32), à partir de ``n_init`` initialisations aléatoires (``K``
    observations tirées au hasard) ; on garde la solution d'inertie minimale.

    Returns
    -------
    labels : ndarray (N,)
    centres : ndarray (K, p)
    inertie : float
        Somme des carrés intra-classes.
    """
    rng = _rng(rng)
    X = np.asarray(X, float)
    N = X.shape[0]
    meilleur = None
    inits = [np.asarray(centres_init, float)] if centres_init is not None else \
        [X[rng.choice(N, K, replace=False)] for _ in range(n_init)]
    for C in inits:
        C = C.copy()
        labels = None
        for _ in range(max_iter):
            nouv = distances_carrees(X, C).argmin(1)
            if labels is not None and np.array_equal(nouv, labels):
                break
            labels = nouv
            for k in range(K):
                if np.any(labels == k):
                    C[k] = X[labels == k].mean(0)
                else:  # classe vide : on la réinitialise sur le point le plus mal servi
                    d = distances_carrees(X, C).min(1)
                    C[k] = X[d.argmax()]
        inertie = ((X - C[labels]) ** 2).sum()
        if meilleur is None or inertie < meilleur[2]:
            meilleur = (labels, C, inertie)
    return meilleur


def kmedoides(D, K, n_init=10, max_iter=100, rng=None, medoides_init=None):
    """K-médoïdes (algorithme 14.2) à partir d'une matrice de dissimilarités.

    Alterne (i) pour chaque classe, le choix comme centre de l'observation qui
    minimise la dissimilarité totale aux autres membres (14.35), et (ii)
    l'affectation de chaque observation au médoïde le plus proche (14.37).

    Returns
    -------
    labels : ndarray (N,)
    medoides : ndarray (K,) d'indices
    critere : float
        Valeur de (14.38), ``sum_i d(i, medoide de i)``.
    """
    rng = _rng(rng)
    D = np.asarray(D, float)
    N = D.shape[0]
    inits = [np.asarray(medoides_init)] if medoides_init is not None else \
        [rng.choice(N, K, replace=False) for _ in range(n_init)]
    meilleur = None
    for med in inits:
        med = np.array(med)
        for _ in range(max_iter):
            labels = D[:, med].argmin(1)
            nouv = med.copy()
            for k in range(K):
                membres = np.where(labels == k)[0]
                if len(membres):
                    nouv[k] = membres[D[np.ix_(membres, membres)].sum(1).argmin()]
            if np.array_equal(nouv, med):
                break
            med = nouv
        labels = D[:, med].argmin(1)
        crit = D[np.arange(N), med[labels]].sum()
        if meilleur is None or crit < meilleur[2]:
            meilleur = (labels, med, crit)
    return meilleur
