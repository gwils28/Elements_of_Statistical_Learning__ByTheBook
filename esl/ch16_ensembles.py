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


# ---------------------------------------------------------------------------
# Forward stagewise (algorithme 16.1)
# ---------------------------------------------------------------------------

def forward_stagewise(X, y, eps=0.01, n_steps=200):
    """Régression linéaire « forward stagewise » incrémentale (algorithme 16.1).

    À chaque étape on choisit la fonction de base ``k*`` qui ajuste le mieux
    le résidu courant au sens des moindres carrés (c'est-à-dire qui maximise
    ``(x_k^T r)^2 / ||x_k||^2``), puis on ajoute ``eps * signe(x_k^T r)`` à son
    coefficient.

    Parameters
    ----------
    X : ndarray (N, K)
        Fonctions de base évaluées aux points d'apprentissage (en général
        centrées ; aucune constante n'est ajoutée).
    y : ndarray (N,)
        Réponse (en général centrée).
    eps : float
        Taille du pas.
    n_steps : int
        Nombre d'itérations ``M``.

    Returns
    -------
    path : ndarray (n_steps + 1, K)
        Coefficients après 0, 1, ..., ``n_steps`` itérations.
    """
    X = np.asarray(X, float)
    r = np.asarray(y, float).copy()
    N, K = X.shape
    norm2 = (X ** 2).sum(axis=0)
    norm2 = np.where(norm2 > 0, norm2, np.inf)
    alpha = np.zeros(K)
    path = np.zeros((n_steps + 1, K))
    for m in range(1, n_steps + 1):
        c = X.T @ r
        score = c ** 2 / norm2
        k = int(np.argmax(score))
        if score[k] <= 0:  # résidu orthogonal à toutes les bases : on s'arrête
            path[m:] = alpha
            break
        step = eps * np.sign(c[k])
        alpha[k] += step
        r -= step * X[:, k]
        path[m] = alpha
    return path


# ---------------------------------------------------------------------------
# Lasso par descente de coordonnées
# ---------------------------------------------------------------------------

def _soft(z, lam):
    """Seuillage doux ``S(z, lam) = signe(z) (|z| - lam)_+``."""
    return np.sign(z) * max(abs(z) - lam, 0.0)


def _cd_gram(G, c, lam, b, tol, max_sweeps):
    """Descente de coordonnées sur la forme quadratique ``b'Gb/2 - c'b + lam|b|_1``.

    Stratégie « ensemble actif » : un balayage complet, puis des balayages
    restreints aux coefficients non nuls jusqu'à convergence, et ainsi de suite
    jusqu'à ce qu'un balayage complet ne modifie plus rien.
    """
    diag = np.diag(G).copy()
    ok = diag > 0
    Gb = G @ b
    K = len(b)
    full = np.flatnonzero(ok)
    sweeps = 0
    while sweeps < max_sweeps:
        # balayage complet
        sweeps += 1
        dmax = 0.0
        for j in full:
            z = c[j] - Gb[j] + diag[j] * b[j]
            new = _soft(z, lam) / diag[j]
            d = new - b[j]
            if d != 0.0:
                Gb += G[:, j] * d
                b[j] = new
                dmax = max(dmax, abs(d) * np.sqrt(diag[j]))
        if dmax < tol:
            break
        # balayages sur l'ensemble actif
        while sweeps < max_sweeps:
            sweeps += 1
            active = np.flatnonzero(b != 0)
            dmax = 0.0
            for j in active:
                z = c[j] - Gb[j] + diag[j] * b[j]
                new = _soft(z, lam) / diag[j]
                d = new - b[j]
                if d != 0.0:
                    Gb += G[:, j] * d
                    b[j] = new
                    dmax = max(dmax, abs(d) * np.sqrt(diag[j]))
            if dmax < tol:
                break
    return b, sweeps


def lasso_cd(X, y, lam, coef_init=None, tol=1e-7, max_sweeps=10000):
    """Lasso par descente de coordonnées (avec constante non pénalisée).

    Minimise ``(1 / 2N) * ||y - b0 - X b||^2 + lam * ||b||_1``
    (même normalisation que ``sklearn.linear_model.Lasso(alpha=lam)``).

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,)
    lam : float
        Paramètre de pénalisation (>= 0).
    coef_init : ndarray (p,), optionnel
        Point de départ (« démarrage à chaud »).
    tol : float
        Critère d'arrêt sur la plus grande variation (pondérée) d'un coefficient,
        relativement à l'écart-type de ``y``.
    max_sweeps : int
        Nombre maximal de balayages.

    Returns
    -------
    b0 : float
    b : ndarray (p,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N = len(y)
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    G = Xc.T @ Xc / N
    c = Xc.T @ yc / N
    b = np.zeros(X.shape[1]) if coef_init is None else np.array(coef_init, float)
    scale = max(yc.std(), 1e-12)
    b, _ = _cd_gram(G, c, lam, b, tol * scale, max_sweeps)
    return ym - xm @ b, b


def lasso_path_cd(X, y, lambdas=None, n_lambdas=50, lambda_ratio=1e-3,
                  tol=1e-7, max_sweeps=10000):
    """Chemin complet du lasso (descente de coordonnées, démarrages à chaud).

    Parameters
    ----------
    X, y : données (une constante non pénalisée est ajustée).
    lambdas : ndarray, optionnel
        Grille décroissante de valeurs de ``lam`` ; par défaut une grille
        logarithmique de ``lam_max`` (plus petite valeur annulant tous les
        coefficients) à ``lambda_ratio * lam_max``.
    n_lambdas, lambda_ratio : paramètres de la grille par défaut.

    Returns
    -------
    lambdas : ndarray (L,)
    intercepts : ndarray (L,)
    coefs : ndarray (L, p)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, p = X.shape
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    G = Xc.T @ Xc / N
    c = Xc.T @ yc / N
    if lambdas is None:
        lmax = np.abs(c).max()
        lambdas = lmax * np.logspace(0, np.log10(lambda_ratio), n_lambdas)
    lambdas = np.asarray(lambdas, float)
    scale = max(yc.std(), 1e-12)
    b = np.zeros(p)
    coefs = np.zeros((len(lambdas), p))
    for i, lam in enumerate(lambdas):
        b, _ = _cd_gram(G, c, lam, b, tol * scale, max_sweeps)
        coefs[i] = b
    intercepts = ym - coefs @ xm
    return lambdas, intercepts, coefs


def lasso_path_fast(X, y, lambdas=None, n_lambdas=50, lambda_ratio=1e-3):
    """Même chemin que :func:`lasso_path_cd`, calculé par scikit-learn.

    La descente de coordonnées de scikit-learn est écrite en Cython : elle est
    bien plus rapide que notre boucle Python lorsque le dictionnaire compte des
    milliers de fonctions de base (post-traitement d'ensembles, règles).  Elle
    minimise exactement le même critère ``(1/2N)||y - b0 - Xb||^2 + lam ||b||_1``.

    Returns
    -------
    lambdas, intercepts, coefs : comme :func:`lasso_path_cd`.
    """
    from sklearn.linear_model import lasso_path
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    if lambdas is None:
        lmax = np.abs(Xc.T @ yc).max() / len(y)
        lambdas = lmax * np.logspace(0, np.log10(lambda_ratio), n_lambdas)
    lambdas = np.asarray(lambdas, float)
    _, coefs, _ = lasso_path(Xc, yc, alphas=lambdas, tol=1e-6, max_iter=10000)
    coefs = coefs.T
    return lambdas, ym - coefs @ xm, coefs


# ---------------------------------------------------------------------------
# Ridge paramétrée par les degrés de liberté
# ---------------------------------------------------------------------------

def ridge_lambda_for_df(d2, df):
    """Valeur de ``lam`` telle que ``sum d_j^2 / (d_j^2 + lam) = df``.

    Parameters
    ----------
    d2 : ndarray
        Carrés des valeurs singulières de la matrice (centrée) des prédicteurs.
    df : float
        Degrés de liberté visés, ``0 < df < nombre de d_j > 0``.
    """
    d2 = np.asarray(d2, float)
    f = lambda lam: np.sum(d2 / (d2 + lam)) - df
    hi = d2.max() * 1e8 + 1.0
    return brentq(f, 0.0, hi, xtol=1e-12 * hi)


def ridge_path_df(X, y, dfs):
    """Coefficients ridge (sans constante) pour une liste de degrés de liberté.

    On utilise la décomposition en valeurs singulières ``X = U D V^T`` :
    ``b(lam) = V diag(d_j / (d_j^2 + lam)) U^T y`` (éq. (3.47) du livre).

    Parameters
    ----------
    X : ndarray (N, p), supposée centrée.
    y : ndarray (N,), supposée centrée.
    dfs : liste de degrés de liberté (strictement entre 0 et rang(X)).

    Returns
    -------
    lambdas : ndarray (L,)
    coefs : ndarray (L, p)
    """
    U, d, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    keep = d > 1e-10 * d.max()
    U, d, Vt = U[:, keep], d[keep], Vt[keep]
    uty = U.T @ y
    lambdas = np.array([ridge_lambda_for_df(d ** 2, df) for df in dfs])
    coefs = np.array([Vt.T @ (d / (d ** 2 + lam) * uty) for lam in lambdas])
    return lambdas, coefs


# ---------------------------------------------------------------------------
# Données corrélées par blocs (section 16.2.3 / exercice 16.1)
# ---------------------------------------------------------------------------

def block_correlated_data(n, beta=None, n_blocks=50, block_size=20, rho=0.95,
                          nsr=0.72, rng=None):
    """Simule les données corrélées par blocs de la figure 16.3.

    Chaque variable s'écrit ``X_j = sqrt(rho) Z_b + sqrt(1 - rho) E_j`` où
    ``Z_b`` est un facteur commun à son bloc ``b`` et les ``Z_b, E_j`` sont
    i.i.d. N(0, 1) : on a ``Var X_j = 1``, ``corr(X_j, X_k) = rho`` dans un même
    bloc et 0 entre deux blocs.  Le vrai vecteur ``beta`` a un coefficient non
    nul N(0, 1) par bloc (position tirée au hasard) ; le bruit gaussien a une
    variance ``sigma^2 = nsr * Var(f(X)) = nsr * ||beta||^2``.

    Parameters
    ----------
    n : int
        Nombre d'observations.
    beta : ndarray (n_blocks * block_size,), optionnel
        Vrais coefficients (à réutiliser pour simuler un échantillon test) ;
        tirés selon la règle ci-dessus s'ils ne sont pas fournis.
    n_blocks, block_size, rho, nsr : paramètres de la simulation.
    rng : numpy.random.RandomState

    Returns
    -------
    X : ndarray (n, p)
    y : ndarray (n,)
    beta : ndarray (p,)
    sigma : float
        Écart-type du bruit.
    """
    rng = np.random.RandomState(0) if rng is None else rng
    p = n_blocks * block_size
    if beta is None:
        beta = np.zeros(p)
        for b in range(n_blocks):
            j = b * block_size + rng.randint(block_size)
            beta[j] = rng.randn()
    Z = rng.randn(n, n_blocks)
    E = rng.randn(n, p)
    X = np.sqrt(rho) * np.repeat(Z, block_size, axis=1) + np.sqrt(1 - rho) * E
    sigma = np.sqrt(nsr * np.sum(beta ** 2))
    y = X @ beta + sigma * rng.randn(n)
    return X, y, beta, sigma


# ---------------------------------------------------------------------------
# Longueur d'arc L1 (exercice 16.2)
# ---------------------------------------------------------------------------

def l1_arc_length(path):
    """Longueur d'arc L1 cumulée d'un chemin de coefficients affine par morceaux.

    Parameters
    ----------
    path : ndarray (T, p)
        Coefficients aux instants successifs (le chemin est supposé affine
        entre deux instants consécutifs).

    Returns
    -------
    ndarray (T,)
        ``Lambda(t_i) = sum_{l < i} |alpha(t_{l+1}) - alpha(t_l)|_1`` avec
        ``Lambda(t_0) = 0``.
    """
    path = np.asarray(path, float)
    inc = np.abs(np.diff(path, axis=0)).sum(axis=1)
    return np.concatenate([[0.0], np.cumsum(inc)])


# ---------------------------------------------------------------------------
# AdaBoost discret avec rétrécissement (figure 16.5)
# ---------------------------------------------------------------------------

def adaboost_shrunk(X, y, n_trees=1000, nu=0.1, max_leaf_nodes=4, X_test=None,
                    y_test=None, random_state=0):
    """AdaBoost.M1 (algorithme 10.1) avec un facteur de rétrécissement ``nu``.

    Le m-ième classifieur ``G_m(x) in {-1, +1}`` est un arbre à
    ``max_leaf_nodes`` feuilles ajusté aux poids courants ; il reçoit le poids
    ``alpha_m = nu * log((1 - err_m) / err_m)``.  On suit au fil des
    itérations la marge L1 normalisée (16.7)
    ``m(f) = min_i y_i f(x_i) / sum_m |alpha_m|``.

    Parameters
    ----------
    X, y : données d'apprentissage, ``y`` dans {-1, +1}.
    n_trees : int
    nu : float
        Rétrécissement (``nu = 1`` : AdaBoost.M1 usuel).
    max_leaf_nodes : int
    X_test, y_test : échantillon test optionnel.

    Returns
    -------
    dict
        ``alpha`` (poids), ``margin``, ``train_error`` et, si un échantillon
        test est fourni, ``test_error`` (tableaux de longueur ``n_trees``).
    """
    from sklearn.tree import DecisionTreeClassifier
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N = len(y)
    w = np.full(N, 1.0 / N)
    F = np.zeros(N)
    Ft = None if X_test is None else np.zeros(len(X_test))
    out = {k: np.zeros(n_trees) for k in ("alpha", "margin", "train_error")}
    if X_test is not None:
        out["test_error"] = np.zeros(n_trees)
    total = 0.0
    for m in range(n_trees):
        tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                                      random_state=random_state)
        tree.fit(X, y, sample_weight=w)
        g = tree.predict(X)
        miss = g != y
        err = np.clip(w[miss].sum() / w.sum(), 1e-10, 1 - 1e-10)
        a = nu * np.log((1 - err) / err)
        w = w * np.exp(a * miss)
        w /= w.sum()
        F += a * g
        total += abs(a)
        out["alpha"][m] = a
        out["margin"][m] = np.min(y * F) / total
        out["train_error"][m] = np.mean(np.sign(F) != y)
        if X_test is not None:
            Ft += a * tree.predict(X_test)
            out["test_error"][m] = np.mean(np.sign(Ft) != y_test)
    return out


# ---------------------------------------------------------------------------
# Ensembles ISLE (algorithme 16.2) et post-traitement lasso (16.9)
# ---------------------------------------------------------------------------

def isle_ensemble(X, y, n_trees=100, eta=0.5, nu=0.1, max_leaf_nodes=6,
                  max_features=None, rng=None):
    """Génère un ensemble d'arbres de régression selon l'algorithme 16.2 (ISLE).

    Perte quadratique : à l'étape ``m`` on tire sans remise un sous-échantillon
    ``S_m`` de taille ``eta * N``, on ajuste un arbre aux résidus
    ``y_i - f_{m-1}(x_i)`` sur ``S_m`` puis ``f_m = f_{m-1} + nu * arbre``.
    ``nu = 0`` donne des arbres indépendants (proche du bagging / des forêts
    aléatoires) ; ``eta = 1, nu > 0`` donne le gradient boosting usuel.

    Parameters
    ----------
    X, y : données d'apprentissage.
    n_trees : int
    eta : float dans ]0, 1]
        Fraction sous-échantillonnée.
    nu : float dans [0, 1]
        « Mémoire » (taux d'apprentissage).
    max_leaf_nodes : int ou None
        Taille des arbres (``None`` : arbres maximaux).
    max_features : paramètre ``max_features`` de scikit-learn (forêts aléatoires).
    rng : numpy.random.RandomState

    Returns
    -------
    f0 : float
        Constante initiale (moyenne de ``y``).
    trees : liste d'arbres scikit-learn ajustés.
    """
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.RandomState(0) if rng is None else rng
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N = len(y)
    size = max(2, int(round(eta * N)))
    f0 = y.mean()
    f = np.full(N, f0)
    trees = []
    for m in range(n_trees):
        idx = rng.choice(N, size=size, replace=False) if size < N else np.arange(N)
        tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                     max_features=max_features,
                                     random_state=rng.randint(2 ** 31 - 1))
        tree.fit(X[idx], (y - f)[idx])
        trees.append(tree)
        if nu > 0:
            f += nu * tree.predict(X)
    return f0, trees


def tree_predictions(trees, X):
    """Matrice ``(N, M)`` des prédictions ``T_m(x_i)`` d'une liste d'arbres."""
    X = np.asarray(X, float)
    return np.column_stack([t.predict(X) for t in trees])
