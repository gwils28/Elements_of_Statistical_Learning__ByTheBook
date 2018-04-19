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


def statistique_gap(X, K_max, B=20, rng=None, n_init=5):
    """Statistique Gap (Tibshirani et al., 2001), section 14.3.11.

    Compare ``log W_K`` observé à son espérance sous une loi uniforme sur le
    pavé contenant les données (estimée par ``B`` simulations).

    Returns
    -------
    dict
        ``logW``, ``logW_ref`` (moyenne), ``gap``, ``s`` (= ``s_K sqrt(1+1/B)``),
        ``K_etoile`` (règle (14.39)).
    """
    rng = _rng(rng)
    X = np.asarray(X, float)
    Ks = np.arange(1, K_max + 1)

    def logW(Y):
        return np.array([np.log(kmeans(Y, K, n_init=n_init, rng=rng)[2]) for K in Ks])

    lw = logW(X)
    ref = np.array([logW(echantillon_reference(X, "uniforme", rng)) for _ in range(B)])
    gap = ref.mean(0) - lw
    s = ref.std(0) * np.sqrt(1 + 1 / B)
    K_etoile = Ks[-1]
    for i in range(len(Ks) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            K_etoile = Ks[i]
            break
    return {"K": Ks, "logW": lw, "logW_ref": ref.mean(0), "gap": gap, "s": s,
            "K_etoile": int(K_etoile)}


def classification_hierarchique(D, methode="average"):
    """Classification hiérarchique ascendante (section 14.3.12).

    À chaque étape on fusionne les deux groupes les moins dissemblables, la
    dissimilarité entre groupes étant mise à jour par la formule de
    Lance-Williams :

    * ``"single"`` (lien simple, 14.41) : ``min(d_kG, d_kH)`` ;
    * ``"complete"`` (lien complet, 14.42) : ``max(d_kG, d_kH)`` ;
    * ``"average"`` (lien moyen, 14.43) : ``(N_G d_kG + N_H d_kH)/(N_G + N_H)``.

    Parameters
    ----------
    D : ndarray (N, N)
        Matrice de dissimilarités symétrique.

    Returns
    -------
    ndarray (N-1, 4)
        Matrice de liaison au format de ``scipy.cluster.hierarchy`` :
        ``[groupe 1, groupe 2, hauteur, effectif]``, le groupe créé à l'étape
        ``s`` recevant le numéro ``N + s``.
    """
    D = np.array(D, float)
    N = D.shape[0]
    np.fill_diagonal(D, np.inf)
    ids = np.arange(N)          # numéro scipy du groupe occupant chaque ligne
    tailles = np.ones(N)
    actif = np.ones(N, bool)
    Z = np.zeros((N - 1, 4))
    for s in range(N - 1):
        sub = np.where(actif)[0]
        M = D[np.ix_(sub, sub)]
        a, b = np.unravel_index(np.argmin(M), M.shape)
        i, j = sub[min(a, b)], sub[max(a, b)]
        h = D[i, j]
        ni, nj = tailles[i], tailles[j]
        if methode == "single":
            nouv = np.minimum(D[i], D[j])
        elif methode == "complete":
            nouv = np.maximum(D[i], D[j])
        elif methode == "average":
            nouv = (ni * D[i] + nj * D[j]) / (ni + nj)
        else:
            raise ValueError(methode)
        Z[s] = [min(ids[i], ids[j]), max(ids[i], ids[j]), h, ni + nj]
        D[i, :] = nouv
        D[:, i] = nouv
        D[i, i] = np.inf
        actif[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        tailles[i] = ni + nj
        ids[i] = N + s
    return Z


def dissimilarites_cophenetiques(Z):
    """Matrice ``C`` des dissimilarités cophénétiques d'une hiérarchie.

    ``C_ii'`` est la hauteur de la fusion à laquelle ``i`` et ``i'`` se retrouvent
    pour la première fois dans le même groupe (section 14.3.12).
    """
    Z = np.asarray(Z)
    N = Z.shape[0] + 1
    membres = {i: [i] for i in range(N)}
    C = np.zeros((N, N))
    for s, (a, b, h, _) in enumerate(Z):
        A, B = membres.pop(int(a)), membres.pop(int(b))
        C[np.ix_(A, B)] = h
        C[np.ix_(B, A)] = h
        membres[N + s] = A + B
    return C


def couper_hierarchie(Z, K):
    """Coupe une hiérarchie pour obtenir ``K`` groupes ; renvoie les étiquettes."""
    Z = np.asarray(Z)
    N = Z.shape[0] + 1
    membres = {i: [i] for i in range(N)}
    for s in range(N - K):
        a, b = int(Z[s, 0]), int(Z[s, 1])
        membres[N + s] = membres.pop(a) + membres.pop(b)
    labels = np.empty(N, int)
    for k, (_, m) in enumerate(sorted(membres.items(), key=lambda t: min(t[1]))):
        labels[m] = k
    return labels


def division_macnaughton_smith(D, membres=None):
    """Une étape de classification descendante (Macnaughton-Smith et al., 1965).

    On part du groupe ``G`` (``membres``, toutes les observations par défaut).
    L'observation de plus grande dissimilarité moyenne aux autres fonde le
    groupe ``H`` ; puis, tant qu'il existe dans ``G`` une observation dont la
    dissimilarité moyenne à ``H`` est plus petite que sa dissimilarité moyenne
    aux autres membres de ``G``, on transfère celle pour qui l'écart est le plus
    grand.

    Returns
    -------
    G, H : listes d'indices
    """
    D = np.asarray(D, float)
    G = list(range(D.shape[0])) if membres is None else list(membres)
    if len(G) < 2:
        return G, []
    moy = [D[i, [j for j in G if j != i]].mean() for i in G]
    H = [G.pop(int(np.argmax(moy)))]
    while len(G) > 1:
        ecarts = [D[i, [j for j in G if j != i]].mean() - D[i, H].mean() for i in G]
        k = int(np.argmax(ecarts))
        if ecarts[k] <= 0:
            break
        H.append(G.pop(k))
    return G, H


def indice_rand_ajuste(a, b):
    """Indice de Rand ajusté (Hubert et Arabie, 1985) entre deux partitions.

    Vaut 1 pour deux partitions identiques (à renumérotation près) et a une
    espérance nulle pour des partitions indépendantes.
    """
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    T = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(T, (ia, ib), 1)

    def c2(x):
        return (x * (x - 1) / 2).sum()

    n = len(ia)
    s_ij, s_a, s_b = c2(T), c2(T.sum(1)), c2(T.sum(0))
    attendu = s_a * s_b / (n * (n - 1) / 2)
    maxi = (s_a + s_b) / 2
    if maxi == attendu:
        return 1.0
    return (s_ij - attendu) / (maxi - attendu)


def em_melange_isotrope(X, K, sigma2=None, n_iter=200, rng=None, mu_init=None, tol=1e-10):
    """EM pour un mélange gaussien ``sum_k pi_k N(mu_k, sigma^2 I)`` (exercice 14.2).

    Parameters
    ----------
    sigma2 : float ou None
        Si fourni, la variance est fixée à cette valeur ; sinon elle est estimée.

    Returns
    -------
    dict
        ``pi``, ``mu``, ``sigma2``, ``resp`` (responsabilités ``(N, K)``) et
        ``loglik`` (historique de la log-vraisemblance).
    """
    rng = _rng(rng)
    X = np.asarray(X, float)
    N, p = X.shape
    mu = X[rng.choice(N, K, replace=False)].copy() if mu_init is None else np.array(mu_init, float)
    pi = np.full(K, 1 / K)
    s2 = X.var() if sigma2 is None else sigma2
    hist = []
    for _ in range(n_iter):
        # étape E : log(pi_k phi(x_i; mu_k, s2 I)), normalisé par log-sum-exp
        L = np.log(pi)[None, :] - distances_carrees(X, mu) / (2 * s2) - p / 2 * np.log(2 * np.pi * s2)
        m = L.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(L - m).sum(1))
        resp = np.exp(L - lse[:, None])
        hist.append(lse.sum())
        # étape M
        Nk = resp.sum(0)
        pi = Nk / N
        mu = (resp.T @ X) / np.maximum(Nk[:, None], 1e-300)
        if sigma2 is None:
            s2 = (resp * distances_carrees(X, mu)).sum() / (N * p)
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-1]):
            break
    return {"pi": pi, "mu": mu, "sigma2": s2, "resp": resp, "loglik": np.array(hist)}


# ---------------------------------------------------------------------------
# 4. Cartes auto-organisatrices
# ---------------------------------------------------------------------------

def grille_som(q1, q2):
    """Coordonnées entières ``l_j`` des ``K = q1 q2`` prototypes d'une grille."""
    return np.array([(a, b) for a in range(q1) for b in range(q2)], float)


def init_som_acp(X, q1, q2):
    """Prototypes initiaux régulièrement répartis dans le plan des deux premières CP.

    Les coordonnées de la grille sont envoyées linéairement sur
    ``[-s_1, s_1] x [-s_2, s_2]`` où ``s_m`` est l'écart-type de la m-ième
    composante principale.
    """
    X = np.asarray(X, float)
    xbar = X.mean(0)
    _, d, Vt = np.linalg.svd(X - xbar, full_matrices=False)
    s = d[:2] / np.sqrt(X.shape[0])
    L = grille_som(q1, q2)
    u = np.column_stack([np.linspace(-1, 1, q)[L[:, m].astype(int)] if q > 1 else
                         np.zeros(len(L)) for m, q in enumerate((q1, q2))])
    return xbar + (u * s) @ Vt[:2]


def som(X, q1, q2, n_iter, R, alpha=(1.0, 0.0), rng=None, init=None,
        voisinage="seuil", suivi=None):
    """Carte auto-organisatrice de Kohonen, version en ligne (section 14.4).

    À l'itération ``t`` on prend une observation ``x_i`` (passages successifs
    dans un ordre aléatoire), on trouve le prototype le plus proche ``m_j`` et on
    met à jour tous ses voisins sur la grille :
    ``m_k <- m_k + alpha_t h(||l_j - l_k||) (x_i - m_k)`` (14.46)-(14.47).
    Le taux ``alpha_t`` et le rayon ``r_t`` décroissent linéairement
    (de ``alpha[0]`` à ``alpha[1]`` et de ``R`` à 1).

    Parameters
    ----------
    voisinage : {"seuil", "gaussien"}
        ``"seuil"`` : ``h = 1`` si ``||l_j - l_k|| < r_t`` (14.46) ; comme
        ``r_t >= 1``, le prototype gagnant est toujours mis à jour, et en fin
        d'apprentissage (``r_t -> 1``) il est le seul ;
        ``"gaussien"`` : ``h = exp(-||l_j - l_k||^2 / (2 r_t^2))``.
    suivi : int ou None
        Si fourni, on enregistre l'erreur de reconstruction toutes les
        ``suivi`` itérations.

    Returns
    -------
    dict
        ``prototypes`` (K, p), ``grille`` (K, 2), ``labels`` (N,), ``erreur``
        (erreur de reconstruction finale) et éventuellement ``historique``.
    """
    rng = _rng(rng)
    X = np.asarray(X, float)
    N = X.shape[0]
    L = grille_som(q1, q2)
    dgrille = np.sqrt(distances_carrees(L))
    M = init_som_acp(X, q1, q2) if init is None else np.array(init, float)
    ordre = np.concatenate([rng.permutation(N) for _ in range(n_iter // N + 1)])[:n_iter]
    hist = []
    for t in range(n_iter):
        frac = t / max(n_iter - 1, 1)
        a = alpha[0] + (alpha[1] - alpha[0]) * frac
        r = R + (1 - R) * frac
        x = X[ordre[t]]
        j = ((M - x) ** 2).sum(1).argmin()
        if voisinage == "seuil":
            h = (dgrille[j] < r - 1e-12).astype(float)
        else:
            h = np.exp(-dgrille[j] ** 2 / (2 * r ** 2))
        M += a * h[:, None] * (x - M)
        if suivi and (t % suivi == 0 or t == n_iter - 1):
            hist.append((t, distances_carrees(X, M).min(1).sum()))
    d = distances_carrees(X, M)
    out = {"prototypes": M, "grille": L, "labels": d.argmin(1), "erreur": d.min(1).sum()}
    if suivi:
        out["historique"] = np.array(hist)
    return out


def som_batch(X, q1, q2, n_iter, R, r_final=1.0, init=None, voisinage="seuil"):
    """Carte auto-organisatrice, version « batch » (éq. (14.48)).

    À chaque itération, chaque observation est affectée au prototype le plus
    proche, puis chaque prototype ``m_j`` est remplacé par la moyenne pondérée
    ``sum_k w_k x_k / sum_k w_k`` des observations affectées à ses voisins sur la
    grille (poids rectangulaires ``1{||l_j - l_k|| < r}`` ou gaussiens). Le rayon
    décroît linéairement de ``R`` à ``r_final``. Avec ``r_final <= 1`` et des
    poids rectangulaires, les dernières itérations sont exactement celles des
    K-means.

    Returns
    -------
    dict
        ``prototypes``, ``grille``, ``labels``, ``erreur``, ``historique``
        (erreur de reconstruction à chaque itération).
    """
    X = np.asarray(X, float)
    L = grille_som(q1, q2)
    dgrille = np.sqrt(distances_carrees(L))
    M = init_som_acp(X, q1, q2) if init is None else np.array(init, float)
    hist = []
    for t in range(n_iter):
        r = R + (r_final - R) * t / max(n_iter - 1, 1)
        lab = distances_carrees(X, M).argmin(1)
        if voisinage == "seuil":
            H = (dgrille < r - 1e-12).astype(float)
        else:
            H = np.exp(-dgrille ** 2 / (2 * r ** 2))
        A = np.zeros((len(M), len(X)))
        A[lab, np.arange(len(X))] = 1.0      # A[k, i] = 1 si x_i est affecté à m_k
        Wt = H @ A                           # poids de x_i pour le prototype j
        tot = Wt.sum(1)
        ok = tot > 0
        M[ok] = (Wt[ok] @ X) / tot[ok, None]
        hist.append(distances_carrees(X, M).min(1).sum())
    d = distances_carrees(X, M)
    return {"prototypes": M, "grille": L, "labels": d.argmin(1), "erreur": d.min(1).sum(),
            "historique": np.array(hist)}


def donnees_demi_sphere(n_par_classe=30, sd=0.6, rng=None):
    """Données simulées près d'une sphère (exercice 14.5, figures 14.15-14.18).

    Trois groupes centrés près de ``(0,0,1)``, ``(1,0,0)`` et ``(0,1,0)`` ; les
    angles sont uniformes et un bruit gaussien d'écart-type ``sd`` est ajouté à
    chaque coordonnée. Renvoie ``(X, y)`` avec ``y`` dans {0, 1, 2}.
    """
    rng = _rng(rng)
    q = np.pi / 4
    params = [((-np.pi / 8, np.pi / 8), (0, 2 * np.pi)),
              ((np.pi / 2 - q, np.pi / 2 + q), (-q, q)),
              ((np.pi / 2 - q, np.pi / 2 + q), (np.pi / 2 - q, np.pi / 2 + q))]
    X, y = [], []
    for k, (th, ph) in enumerate(params):
        t = rng.uniform(*th, size=n_par_classe)
        f = rng.uniform(*ph, size=n_par_classe)
        P = np.column_stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])
        X.append(P + sd * rng.normal(size=P.shape))
        y += [k] * n_par_classe
    return np.vstack(X), np.array(y)


# ---------------------------------------------------------------------------
# 5. ACP, courbes principales, Procrustes
# ---------------------------------------------------------------------------

def acp(X, q):
    """ACP par décomposition en valeurs singulières (14.49)-(14.54).

    Returns
    -------
    dict
        ``moyenne`` (= ``mu chapeau``, 14.51), ``V`` (p, q) directions,
        ``scores`` (N, q) = ``U_q D_q`` (= ``lambda chapeau``, 14.52),
        ``d`` (toutes les valeurs singulières), ``reconstruction`` (N, p).
    """
    X = np.asarray(X, float)
    xbar = X.mean(0)
    U, d, Vt = np.linalg.svd(X - xbar, full_matrices=False)
    V = Vt[:q].T
    scores = U[:, :q] * d[:q]
    return {"moyenne": xbar, "V": V, "scores": scores, "d": d,
            "reconstruction": xbar + scores @ V.T}


def _lisseur_local_lineaire(t, Y, t_eval, h):
    """Régression locale linéaire à noyau gaussien de chaque colonne de ``Y`` sur ``t``."""
    W = np.exp(-0.5 * ((t_eval[:, None] - t[None, :]) / h) ** 2)
    S0 = W.sum(1)
    S1 = W @ t
    S2 = W @ t ** 2
    T0 = W @ Y
    T1 = W @ (t[:, None] * Y)
    det = S0 * S2 - S1 ** 2
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    a = (S2[:, None] * T0 - S1[:, None] * T1) / det[:, None]
    b = (S0[:, None] * T1 - S1[:, None] * T0) / det[:, None]
    return a + b * t_eval[:, None]


def projeter_sur_polyligne(X, P):
    """Projette chaque point de ``X`` sur la ligne brisée de sommets ``P``.

    Returns
    -------
    lam : ndarray (N,)
        Abscisse curviligne du projeté.
    proj : ndarray (N, p)
        Points projetés.
    """
    X = np.asarray(X, float)
    seg = np.diff(P, axis=0)
    long = np.sqrt((seg ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(long)])
    A = P[:-1]
    # paramètre t dans [0, 1] sur chaque segment
    t = ((X[:, None, :] - A[None]) * seg[None]).sum(2) / np.maximum(long ** 2, 1e-300)[None]
    t = np.clip(t, 0, 1)
    Q = A[None] + t[..., None] * seg[None]
    d = ((X[:, None, :] - Q) ** 2).sum(2)
    k = d.argmin(1)
    idx = np.arange(len(X))
    return cum[k] + t[idx, k] * long[k], Q[idx, k]


def courbe_principale(X, h=None, n_iter=20, tol=1e-4):
    """Courbe principale de Hastie et Stuetzle (1989), algorithme (14.62).

    On part de la première composante principale puis on alterne :
    (a) lissage de chaque coordonnée ``X_j`` en fonction de l'abscisse
    curviligne ``lambda`` (régression locale linéaire, fenêtre ``h``) ;
    (b) projection de chaque point sur la courbe obtenue.

    Parameters
    ----------
    h : float ou None
        Fenêtre du lisseur, exprimée comme une fraction de la longueur de la
        courbe (0.1 par défaut).

    Returns
    -------
    dict
        ``lam`` (abscisses curvilignes des données), ``courbe`` (points de la
        courbe aux ``lam`` triés), ``ordre``, ``distances`` (historique de la
        somme des carrés des distances).
    """
    X = np.asarray(X, float)
    pc = acp(X, 1)
    lam = pc["scores"][:, 0] - pc["scores"][:, 0].min()
    hist = []
    for _ in range(n_iter):
        ordre = np.argsort(lam)
        ls = lam[ordre]
        hh = (h if h is not None else 0.1) * (ls[-1] - ls[0])
        F = _lisseur_local_lineaire(lam, X, ls, hh)
        lam, proj = projeter_sur_polyligne(X, F)
        hist.append(((X - proj) ** 2).sum())
        if len(hist) > 1 and abs(hist[-2] - hist[-1]) < tol * hist[-1]:
            break
    ordre = np.argsort(lam)
    return {"lam": lam, "courbe": proj[ordre], "ordre": ordre, "distances": np.array(hist)}


def procrustes(X1, X2, echelle=False):
    """Problème de Procrustes (14.56), éventuellement avec facteur d'échelle (14.58).

    Cherche ``min ||X2 - (beta X1 R + 1 mu^T)||_F`` avec ``R`` orthogonale.
    Avec ``X1~, X2~`` centrées et ``X1~^T X2~ = U D V^T`` (SVD) :
    ``R = U V^T``, ``beta = tr(D) / ||X1~||_F^2`` (si ``echelle``) et
    ``mu = xbar2 - beta R^T xbar1``.

    Returns
    -------
    dict
        ``R``, ``beta``, ``mu``, ``ajuste`` (= ``beta X1 R + 1 mu^T``),
        ``distance`` (norme de Frobenius du résidu).
    """
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    m1, m2 = X1.mean(0), X2.mean(0)
    A, B = X1 - m1, X2 - m2
    U, d, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    beta = d.sum() / (A ** 2).sum() if echelle else 1.0
    mu = m2 - beta * R.T @ m1
    fit = beta * X1 @ R + mu
    return {"R": R, "beta": beta, "mu": mu, "ajuste": fit,
            "distance": np.sqrt(((X2 - fit) ** 2).sum())}


def moyenne_procrustes(Xs, echelle=False, n_iter=100, tol=1e-12):
    """Moyenne de Procrustes de ``L`` formes (14.59), avec échelle en option.

    Les formes sont d'abord centrées. Algorithme alterné : (1) chaque forme est
    alignée (rotation, et facteur ``beta_l`` si ``echelle``) sur la moyenne
    courante ``M`` ; (2) ``M`` est remplacée par la moyenne des formes alignées.
    Avec échelle, on impose ``||M||_F^2 = (1/L) sum_l ||X_l||_F^2`` pour exclure
    la solution triviale ``beta_l = 0, M = 0`` (exercice 14.9).

    Returns
    -------
    dict
        ``M``, ``alignees`` (liste), ``R`` (liste), ``beta`` (array),
        ``critere`` (historique de ``sum_l ||beta_l X_l R_l - M||_F^2``).
    """
    Xs = [np.asarray(X, float) - np.asarray(X, float).mean(0) for X in Xs]
    L = len(Xs)
    norme = np.sqrt(np.mean([(X ** 2).sum() for X in Xs]))
    M = Xs[0].copy()
    hist = []
    for _ in range(n_iter):
        res = [procrustes(X, M, echelle=echelle) for X in Xs]
        al = [r["beta"] * X @ r["R"] for r, X in zip(res, Xs)]
        M = np.mean(al, axis=0)
        if echelle:
            M *= norme / np.sqrt((M ** 2).sum())
        hist.append(sum(((A - M) ** 2).sum() for A in al))
        if len(hist) > 1 and hist[-2] - hist[-1] < tol * max(hist[-1], 1e-300):
            break
    res = [procrustes(X, M, echelle=echelle) for X in Xs]
    al = [r["beta"] * X @ r["R"] for r, X in zip(res, Xs)]
    return {"M": M, "alignees": al, "R": [r["R"] for r in res],
            "beta": np.array([r["beta"] for r in res]), "critere": np.array(hist)}


def moyenne_affine(Xs):
    """Moyenne invariante par transformation affine (14.60), sans itération.

    Avec ``H_l = X_l (X_l^T X_l)^{-1} X_l^T`` (projecteur sur l'espace des
    colonnes de ``X_l`` centrée), ``M`` est formée des ``p`` premiers vecteurs
    propres de ``Hbar = (1/L) sum_l H_l`` (donc ``M^T M = I``) et
    ``A_l = (X_l^T X_l)^{-1} X_l^T M``.

    Returns
    -------
    dict
        ``M`` (N, p), ``A`` (liste de matrices p x p), ``alignees``
        (``X_l A_l``), ``valeurs_propres`` de ``Hbar``.
    """
    Xs = [np.asarray(X, float) - np.asarray(X, float).mean(0) for X in Xs]
    p = Xs[0].shape[1]
    H = np.mean([X @ np.linalg.solve(X.T @ X, X.T) for X in Xs], axis=0)
    vals, vecs = np.linalg.eigh(H)
    ordre = np.argsort(vals)[::-1]
    M = vecs[:, ordre[:p]]
    A = [np.linalg.solve(X.T @ X, X.T @ M) for X in Xs]
    return {"M": M, "A": A, "alignees": [X @ a for X, a in zip(Xs, A)],
            "valeurs_propres": vals[ordre]}


def forme_s(n=96, rng=None, bruit=0.0, deformation=0.0):
    """Courbe en forme de « S » échantillonnée en ``n`` points repères.

    Deux demi-cercles raccordés ; ``deformation`` ajoute une petite
    déformation lisse aléatoire et ``bruit`` un bruit gaussien. Sert de
    substitut aux « S » manuscrits de la figure 14.25 (données non fournies).
    """
    rng = _rng(rng)
    t = np.linspace(0, 1, n)
    u = np.clip(2 * t, 0, 1)          # parcours de l'arc supérieur
    v = np.clip(2 * t - 1, 0, 1)      # parcours de l'arc inférieur
    a1 = np.pi / 6 + 4 * np.pi / 3 * u
    a2 = np.pi / 2 - 4 * np.pi / 3 * v
    haut = t <= 0.5
    x = np.where(haut, np.cos(a1), np.cos(a2))
    y = np.where(haut, 1 + np.sin(a1), -1 + np.sin(a2))
    P = np.column_stack([x, y])
    if deformation:
        coef = rng.normal(scale=deformation, size=(3, 2))
        base = np.column_stack([np.sin(np.pi * t), np.sin(2 * np.pi * t), np.cos(np.pi * t)])
        P = P + base @ coef
    if bruit:
        P = P + bruit * rng.normal(size=P.shape)
    return P


def rotation2d(theta):
    """Matrice de rotation plane d'angle ``theta`` (convention ``x -> x R``)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


# ---------------------------------------------------------------------------
# 6. Positionnement multidimensionnel
# ---------------------------------------------------------------------------

def double_centrage(D2):
    """``B = -1/2 J D2 J`` avec ``J = I - 11^T/N`` (``D2`` : distances au carré)."""
    D2 = np.asarray(D2, float)
    N = D2.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    return -0.5 * J @ D2 @ J


def mds_classique(D, k=2, similarites=False):
    """Positionnement multidimensionnel classique (échelle de Torgerson-Gower).

    Si ``similarites`` est faux, ``D`` contient des dissimilarités et l'on forme
    ``S = -1/2 J D^2 J`` ; sinon ``D`` est déjà la matrice des produits scalaires
    centrés ``s_ii'``. La solution de (14.100) est ``Z = E_k D_k`` où ``E_k``
    contient les ``k`` premiers vecteurs propres de ``S`` et ``D_k`` les racines
    des valeurs propres (exercice 14.11).

    Returns
    -------
    Z : ndarray (N, k)
    valeurs_propres : ndarray (N,) par ordre décroissant
    """
    S = np.asarray(D, float) if similarites else double_centrage(np.asarray(D, float) ** 2)
    vals, vecs = np.linalg.eigh(S)
    ordre = np.argsort(vals)[::-1]
    vals, vecs = vals[ordre], vecs[:, ordre]
    return vecs[:, :k] * np.sqrt(np.maximum(vals[:k], 0)), vals


def stress(D, Z):
    """Stress de Kruskal-Shephard (14.98) : ``sum_{i<i'} (d_ii' - ||z_i - z_i'||)^2``."""
    E = np.sqrt(distances_carrees(Z))
    iu = np.triu_indices(len(Z), 1)
    return ((np.asarray(D)[iu] - E[iu]) ** 2).sum()


def mds_moindres_carres(D, k=2, n_iter=300, Z0=None, tol=1e-10):
    """MDS par moindres carrés (Kruskal-Shephard, 14.98) via l'algorithme SMACOF.

    Chaque itération applique la transformation de Guttman
    ``Z <- (1/N) B(Z) Z`` qui majore le stress par une quadratique et fait
    décroître (14.98) de façon monotone. Initialisation par le MDS classique.

    Returns
    -------
    Z : ndarray (N, k)
    historique : ndarray des valeurs du stress
    """
    D = np.asarray(D, float)
    N = D.shape[0]
    Z = mds_classique(D, k)[0] if Z0 is None else np.array(Z0, float)
    hist = [stress(D, Z)]
    for _ in range(n_iter):
        E = np.sqrt(distances_carrees(Z))
        with np.errstate(divide="ignore", invalid="ignore"):
            Bm = np.where(E > 1e-12, -D / E, 0.0)
        np.fill_diagonal(Bm, 0)
        np.fill_diagonal(Bm, -Bm.sum(1))
        Z = Bm @ Z / N
        hist.append(stress(D, Z))
        if hist[-2] - hist[-1] < tol * hist[-2]:
            break
    return Z, np.array(hist)


# ---------------------------------------------------------------------------
# 7. Clustering spectral et ACP à noyau
# ---------------------------------------------------------------------------

def donnees_cercles(n_par_cercle=150, rayons=(1.0, 2.8, 5.0), sd=0.25, rng=None):
    """Trois cercles concentriques bruités (figure 14.29). Renvoie ``(X, y)``."""
    rng = _rng(rng)
    X, y = [], []
    for k, r in enumerate(rayons):
        a = rng.uniform(0, 2 * np.pi, n_par_cercle)
        X.append(np.column_stack([r * np.cos(a), r * np.sin(a)]) + sd * rng.normal(size=(n_par_cercle, 2)))
        y += [k] * n_par_cercle
    return np.vstack(X), np.array(y)


def matrice_adjacence(X, c, k_voisins=None):
    """Matrice des poids ``W`` d'un graphe de similarité (section 14.5.3).

    ``s_ii' = exp(-||x_i - x_i'||^2 / c)`` ; si ``k_voisins`` est donné, on ne
    garde que les paires ``(i, i')`` où l'un des deux points est parmi les
    ``k`` plus proches voisins de l'autre (graphe des ``k`` voisins symétrisé).
    """
    D2 = distances_carrees(X)
    S = np.exp(-D2 / c)
    np.fill_diagonal(S, 0)
    if k_voisins is None:
        return S
    N = len(X)
    D2b = D2.copy()
    np.fill_diagonal(D2b, np.inf)
    voisins = np.argsort(D2b, axis=1)[:, :k_voisins]
    A = np.zeros((N, N), bool)
    A[np.repeat(np.arange(N), k_voisins), voisins.ravel()] = True
    A = A | A.T
    return np.where(A, S, 0.0)


def laplacien(W, normalise=False):
    """Laplacien de graphe ``L = G - W`` (14.63), ou ``I - G^{-1} W`` si ``normalise``."""
    W = np.asarray(W, float)
    g = W.sum(1)
    if normalise:
        return np.eye(len(W)) - W / g[:, None]
    return np.diag(g) - W


def clustering_spectral(X, K, c=2.0, k_voisins=10, m=None, rng=None):
    """Clustering spectral (section 14.5.3).

    Construit le graphe des ``k`` voisins, calcule les ``m`` vecteurs propres du
    laplacien associés aux plus petites valeurs propres (en ignorant le vecteur
    constant) puis applique les K-means aux lignes de la matrice ``Z`` obtenue.

    Returns
    -------
    dict
        ``labels``, ``Z`` (N, m), ``valeurs_propres`` (toutes, croissantes),
        ``vecteurs`` (vecteurs propres, mêmes colonnes).
    """
    m = K - 1 if m is None else m
    W = matrice_adjacence(X, c, k_voisins)
    vals, vecs = np.linalg.eigh(laplacien(W))
    Z = vecs[:, 1:m + 1]
    labels = kmeans(Z, K, n_init=10, rng=rng)[0]
    return {"labels": labels, "Z": Z, "valeurs_propres": vals, "vecteurs": vecs}


def noyau_gaussien(X, Y=None, c=1.0):
    """Noyau radial ``K(x, x') = exp(-||x - x'||^2 / c)`` (14.67)."""
    return np.exp(-distances_carrees(X, Y) / c)


def acp_noyau(K, m=2):
    """ACP à noyau (section 14.5.4) à partir d'une matrice de Gram ``K``.

    On double-centre : ``K~ = (I - M) K (I - M) = U D^2 U^T`` (14.65), puis
    ``Z = U D``. Les coefficients ``alpha_jm = u_jm / d_m`` permettent d'écrire
    ``z_im = sum_j alpha_jm K~(x_i, x_j)`` (exercice 14.16).

    Returns
    -------
    dict
        ``Z`` (N, m), ``alpha`` (N, m), ``valeurs_propres`` (= ``d^2`` triées).
    """
    K = np.asarray(K, float)
    N = K.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    Kc = J @ K @ J
    vals, vecs = np.linalg.eigh(Kc)
    ordre = np.argsort(vals)[::-1]
    vals, vecs = vals[ordre], vecs[:, ordre]
    d = np.sqrt(np.maximum(vals[:m], 0))
    with np.errstate(divide="ignore"):
        alpha = np.where(d > 1e-12, vecs[:, :m] / np.where(d > 0, d, 1), 0.0)
    return {"Z": vecs[:, :m] * d, "alpha": alpha, "valeurs_propres": vals}


# ---------------------------------------------------------------------------
# 8. ACP parcimonieuse, NMF, ACI
# ---------------------------------------------------------------------------

def seuillage_doux(z, gamma):
    """Opérateur de seuillage doux ``sign(z) (|z| - gamma)_+``."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)


def elastic_net(X, y, lam2, lam1, n_iter=500, tol=1e-10, beta0=None):
    """Minimise ``||y - X b||^2 + lam2 ||b||^2 + lam1 ||b||_1`` par descente par coordonnées.

    La mise à jour de la coordonnée ``j`` est
    ``b_j = S(x_j^T r_j, lam1/2) / (||x_j||^2 + lam2)`` où ``r_j`` est le
    résidu partiel (sans la contribution de ``b_j``).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    b = np.zeros(p) if beta0 is None else np.array(beta0, float)
    nrm = (X ** 2).sum(0)
    r = y - X @ b
    for _ in range(n_iter):
        ecart = 0.0
        for j in range(p):
            ancien = b[j]
            rho = X[:, j] @ r + nrm[j] * ancien
            b[j] = seuillage_doux(rho, lam1 / 2) / (nrm[j] + lam2)
            if b[j] != ancien:
                r -= X[:, j] * (b[j] - ancien)
                ecart = max(ecart, abs(b[j] - ancien))
        if ecart < tol:
            break
    return b


def procrustes_reduit(Mat):
    """Solution de ``max tr(Theta^T M)`` sous ``Theta^T Theta = I`` (14.116).

    Si ``M = U D Q^T`` (SVD réduite), ``Theta = U Q^T``.
    """
    U, _, Qt = np.linalg.svd(np.asarray(Mat, float), full_matrices=False)
    return U @ Qt


def acp_parcimonieuse(X, K, lam2, lam1, n_iter=100, tol=1e-8):
    """ACP parcimonieuse de Zou, Hastie et Tibshirani (2006), critère (14.71).

    Alterne : (i) à ``Theta`` fixé, ``K`` problèmes elastic net de réponses
    ``X theta_k`` ; (ii) à ``V`` fixé, ``Theta = U Q^T`` avec
    ``X^T X V = U D Q^T`` (exercice 14.12). Les colonnes de ``X`` sont centrées.

    Parameters
    ----------
    lam1 : float ou array de taille K

    Returns
    -------
    dict
        ``V`` (p, K) non normalisée, ``directions`` (colonnes de ``V`` normées),
        ``Theta``, ``critere`` (historique).
    """
    X = np.asarray(X, float)
    X = X - X.mean(0)
    lam1 = np.broadcast_to(np.asarray(lam1, float), (K,))
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Theta = Vt[:K].T.copy()
    V = Theta.copy()
    hist = []
    for _ in range(n_iter):
        V = np.column_stack([elastic_net(X, X @ Theta[:, k], lam2, lam1[k], beta0=V[:, k])
                             for k in range(K)])
        Theta = procrustes_reduit(X.T @ (X @ V))
        crit = ((X - X @ V @ Theta.T) ** 2).sum() + lam2 * (V ** 2).sum() + \
            (lam1 * np.abs(V).sum(0)).sum()
        hist.append(crit)
        if len(hist) > 1 and abs(hist[-2] - hist[-1]) < tol * hist[-1]:
            break
    nrm = np.sqrt((V ** 2).sum(0))
    return {"V": V, "directions": V / np.where(nrm > 0, nrm, 1), "Theta": Theta,
            "critere": np.array(hist)}


def nmf(X, r, n_iter=500, rng=None, W0=None, H0=None, eps=1e-12):
    """Factorisation en matrices positives ``X ≈ W H`` (Lee et Seung), (14.72)-(14.74).

    Maximise la log-vraisemblance de Poisson
    ``L(W, H) = sum_ij [x_ij log (WH)_ij - (WH)_ij]`` (14.73) par les mises à
    jour multiplicatives (14.74).

    Returns
    -------
    W : ndarray (N, r)
    H : ndarray (r, p)
    historique : ndarray des valeurs de ``L``
    """
    rng = _rng(rng)
    X = np.asarray(X, float)
    N, p = X.shape
    W = rng.uniform(0.1, 1, (N, r)) if W0 is None else np.array(W0, float)
    H = rng.uniform(0.1, 1, (r, p)) if H0 is None else np.array(H0, float)
    hist = []
    for _ in range(n_iter):
        WH = W @ H + eps
        W *= ((X / WH) @ H.T) / H.sum(1)[None, :]
        WH = W @ H + eps
        H *= (W.T @ (X / WH)) / W.sum(0)[:, None]
        WH = W @ H + eps
        hist.append((X * np.log(WH) - WH).sum())
    return W, H, np.array(hist)


def blanchir(X):
    """Centre et blanchit ``X`` (covariance empirique identité) via la SVD.

    Returns
    -------
    Xb : ndarray (N, p)
    T : ndarray (p, p)
        Matrice de blanchiment : ``Xb = (X - moyenne) T``.
    """
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    N = len(X)
    _, d, Vt = np.linalg.svd(Xc, full_matrices=False)
    T = Vt.T / d * np.sqrt(N)
    return Xc @ T, T
