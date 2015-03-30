"""Outils du chapitre 2 : vue d'ensemble de l'apprentissage supervisé.

Ce module regroupe les implémentations « à la main » utilisées dans les
notebooks du chapitre 2 :

* moindres carrés (éq. (2.6) du livre) et classement par régression
  linéaire d'un codage 0/1 ;
* k plus proches voisins (éq. (2.8)) en régression et en classification ;
* la simulation « mixture » de la section 2.3.3 (mélanges de gaussiennes)
  et le classifieur de Bayes associé ;
* quelques formules du fléau de la dimension (section 2.5) ;
* les poids de lissage linéaires de l'exercice 2.7 et la réduction des
  ex-aequo de l'exercice 2.6.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Moindres carrés
# ---------------------------------------------------------------------------


def add_intercept(X):
    """Ajoute une colonne de 1 en tête de la matrice de design.

    Parameters
    ----------
    X : ndarray, shape (n, p)

    Returns
    -------
    ndarray, shape (n, p + 1)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(X.shape[0]), X])


def least_squares_fit(X, y, intercept=True, weights=None):
    """Estimateur des moindres carrés (éventuellement pondérés).

    Résout les équations normales :math:`X^T W X \\beta = X^T W y`
    (éq. (2.5)-(2.6) du livre lorsque :math:`W = I`).

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Matrice des entrées.
    y : ndarray, shape (n,) ou (n, K)
        Réponse(s).
    intercept : bool
        Si vrai, une constante est ajoutée au modèle (coefficient en tête).
    weights : ndarray, shape (n,), optionnel
        Poids positifs :math:`w_i` du critère :math:`\\sum_i w_i (y_i - x_i^T\\beta)^2`.

    Returns
    -------
    beta : ndarray, shape (p + intercept,) ou (p + intercept, K)
    """
    A = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=float))
        A = A * sw[:, None]
        y = y * (sw if y.ndim == 1 else sw[:, None])
    # lstsq (décomposition SVD) est numériquement plus sûr que l'inverse explicite
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def least_squares_predict(X, beta, intercept=True):
    """Prédictions :math:`\\hat y = x^T \\hat\\beta` (éq. (2.2))."""
    A = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    return A @ beta


def linear_classifier_fit(X, g):
    """Régression linéaire d'un codage 0/1 de la classe (section 2.3.1).

    Parameters
    ----------
    X : ndarray, shape (n, p)
    g : ndarray, shape (n,)
        Étiquettes 0/1.

    Returns
    -------
    beta : ndarray, shape (p + 1,)
    """
    return least_squares_fit(X, np.asarray(g, dtype=float))


def linear_classifier_predict(X, beta, threshold=0.5):
    """Classe prédite : 1 si :math:`x^T\\hat\\beta > 0.5`, sinon 0 (éq. (2.7))."""
    return (least_squares_predict(X, beta) > threshold).astype(int)


# ---------------------------------------------------------------------------
# k plus proches voisins
# ---------------------------------------------------------------------------


def squared_distances(A, B):
    """Matrice des distances euclidiennes au carré entre les lignes de A et B.

    Utilise :math:`\\|a-b\\|^2 = \\|a\\|^2 + \\|b\\|^2 - 2 a^T b` ; les petites
    valeurs négatives dues aux arrondis sont ramenées à 0.

    Returns
    -------
    ndarray, shape (len(A), len(B))
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    D = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(D, 0.0)


def knn_indices(X_train, X_test, k, chunk=2000):
    """Indices des k plus proches voisins (dans X_train) de chaque point de X_test.

    Les ex-aequo de distance sont départagés par l'ordre des observations
    (tri stable), ce qui rend le résultat déterministe.

    Returns
    -------
    ndarray d'entiers, shape (len(X_test), k)
    """
    X_test = np.asarray(X_test, dtype=float)
    if X_test.ndim == 1:
        X_test = X_test[:, None]
    out = np.empty((X_test.shape[0], k), dtype=int)
    for start in range(0, X_test.shape[0], chunk):
        D = squared_distances(X_test[start:start + chunk], X_train)
        out[start:start + chunk] = np.argsort(D, axis=1, kind="stable")[:, :k]
    return out


def knn_regress(X_train, y_train, X_test, k):
    """Moyenne des réponses des k plus proches voisins (éq. (2.8)).

    Parameters
    ----------
    X_train : ndarray, shape (n, p)
    y_train : ndarray, shape (n,)
    X_test : ndarray, shape (m, p)
    k : int

    Returns
    -------
    ndarray, shape (m,)
    """
    idx = knn_indices(X_train, X_test, k)
    return np.asarray(y_train, dtype=float)[idx].mean(axis=1)


def knn_classify(X_train, g_train, X_test, k):
    """Vote majoritaire parmi les k plus proches voisins.

    En cas d'égalité des votes, la plus petite étiquette l'emporte.

    Returns
    -------
    ndarray, shape (m,)
        Étiquettes prédites (mêmes valeurs que ``g_train``).
    """
    g_train = np.asarray(g_train)
    classes, codes = np.unique(g_train, return_inverse=True)
    idx = knn_indices(X_train, X_test, k)
    votes = np.zeros((idx.shape[0], len(classes)))
    rows = np.repeat(np.arange(idx.shape[0]), k)
    np.add.at(votes, (rows, codes[idx].ravel()), 1)
    return classes[votes.argmax(axis=1)]


def error_rate(g_true, g_pred):
    """Taux de mauvais classement :math:`\\frac1n\\sum_i 1(g_i \\ne \\hat g_i)`."""
    return float(np.mean(np.asarray(g_true) != np.asarray(g_pred)))


# ---------------------------------------------------------------------------
# Simulation « mixture » (section 2.3.3) et classifieur de Bayes
# ---------------------------------------------------------------------------


def mixture_centers(rng, n_centers=10):
    """Tire les centres des deux mélanges comme dans le livre.

    Classe 0 (BLEU) : centres :math:`m_k \\sim N((1,0)^T, I)` ;
    classe 1 (ORANGE) : centres :math:`m_k \\sim N((0,1)^T, I)`.

    Returns
    -------
    ndarray, shape (2 * n_centers, 2)
        Les ``n_centers`` premières lignes sont les centres de la classe 0.
    """
    blue = rng.normal(size=(n_centers, 2)) + np.array([1.0, 0.0])
    orange = rng.normal(size=(n_centers, 2)) + np.array([0.0, 1.0])
    return np.vstack([blue, orange])


def sample_mixture(means, n_per_class, rng, sigma2=0.2):
    """Simule des observations des deux classes du mélange.

    Pour chaque observation de la classe :math:`c`, on choisit un centre
    :math:`m_k` de cette classe avec probabilité uniforme, puis on tire
    :math:`x \\sim N(m_k, \\sigma^2 I)` avec :math:`\\sigma^2 = 1/5`.

    Parameters
    ----------
    means : ndarray, shape (2 * K, 2)
        Centres (les K premiers pour la classe 0).
    n_per_class : int
    rng : numpy.random.RandomState
    sigma2 : float

    Returns
    -------
    X : ndarray, shape (2 * n_per_class, 2)
    g : ndarray, shape (2 * n_per_class,)
    """
    means = np.asarray(means, dtype=float)
    K = means.shape[0] // 2
    X, g = [], []
    for c in (0, 1):
        k = rng.randint(0, K, size=n_per_class)
        X.append(means[c * K + k] + np.sqrt(sigma2) * rng.normal(size=(n_per_class, 2)))
        g.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(g)


def mixture_density(X, centers, sigma2=0.2):
    """Densité d'un mélange équipondéré de gaussiennes :math:`N(m_k, \\sigma^2 I_2)`."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    D = squared_distances(X, centers)
    return np.exp(-D / (2 * sigma2)).mean(axis=1) / (2 * np.pi * sigma2)


def bayes_posterior(X, means, sigma2=0.2):
    """Probabilité a posteriori :math:`P(G = 1 \\mid X = x)` (classe ORANGE).

    Les deux classes ont la même probabilité a priori 1/2, donc par la formule
    de Bayes :math:`P(G=1\\mid x) = p_1(x) / (p_0(x) + p_1(x))`.
    """
    means = np.asarray(means, dtype=float)
    K = means.shape[0] // 2
    p0 = mixture_density(X, means[:K], sigma2)
    p1 = mixture_density(X, means[K:], sigma2)
    return p1 / (p0 + p1)


def bayes_classify(X, means, sigma2=0.2):
    """Classifieur de Bayes (éq. (2.23)) pour la simulation mixture."""
    return (bayes_posterior(X, means, sigma2) > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Fléau de la dimension (section 2.5)
# ---------------------------------------------------------------------------


def edge_length(r, p):
    """Côté :math:`e_p(r) = r^{1/p}` d'un sous-cube contenant une fraction r du volume."""
    return np.asarray(r, dtype=float) ** (1.0 / p)


def median_nn_distance(p, N):
    """Distance médiane entre l'origine et le plus proche de N points uniformes
    dans la boule unité de dimension p (éq. (2.24)) :
    :math:`d(p, N) = (1 - (1/2)^{1/N})^{1/p}`."""
    return (1.0 - 0.5 ** (1.0 / N)) ** (1.0 / p)


def sample_unit_ball(n, p, rng):
    """Tire n points uniformément dans la boule unité de :math:`\\mathbb R^p`.

    Direction : gaussienne normalisée ; rayon : :math:`U^{1/p}` avec
    :math:`U` uniforme sur [0, 1] (car :math:`P(\\|X\\|\\le r) = r^p`).
    """
    Z = rng.normal(size=(n, p))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    R = rng.uniform(size=(n, 1)) ** (1.0 / p)
    return Z * R


# ---------------------------------------------------------------------------
# Exercices 2.6 et 2.7
# ---------------------------------------------------------------------------


def ols_weights(X, x0, intercept=True):
    """Poids :math:`\\ell_i(x_0; \\mathcal X)` de la régression linéaire.

    :math:`\\hat f(x_0) = x_0^T (X^TX)^{-1} X^T y = \\sum_i \\ell_i(x_0) y_i`
    avec :math:`\\ell(x_0) = X (X^TX)^{-1} x_0`.

    Returns
    -------
    ndarray, shape (n,)
    """
    A = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    a0 = np.concatenate([[1.0], np.atleast_1d(x0)]) if intercept else np.atleast_1d(x0)
    return A @ np.linalg.solve(A.T @ A, a0)


def knn_weights(X, x0, k):
    """Poids :math:`\\ell_i(x_0;\\mathcal X) = 1/k` si :math:`x_i \\in N_k(x_0)`, 0 sinon."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    idx = knn_indices(X, np.atleast_2d(np.asarray(x0, dtype=float)).reshape(1, -1), k)[0]
    w = np.zeros(X.shape[0])
    w[idx] = 1.0 / k
    return w


def reduce_ties(X, y):
    """Regroupe les observations ayant la même entrée (exercice 2.6).

    Returns
    -------
    X_unique : ndarray, shape (m, p)
        Valeurs distinctes de x.
    y_bar : ndarray, shape (m,)
        Moyenne des réponses pour chaque valeur distincte.
    counts : ndarray, shape (m,)
        Nombre d'observations :math:`N_j` partageant cette valeur (poids).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X_unique, inverse, counts = np.unique(X, axis=0, return_inverse=True,
                                          return_counts=True)
    inverse = np.ravel(inverse)
    y_bar = np.bincount(inverse, weights=np.asarray(y, dtype=float)) / counts
    return X_unique, y_bar, counts


# ---------------------------------------------------------------------------
# Estimateurs « restreints » de la section 2.8
# ---------------------------------------------------------------------------


def gaussian_kernel(x0, x, lam):
    """Noyau gaussien :math:`K_\\lambda(x_0, x) = \\frac{1}{\\lambda}\\exp(-\\|x-x_0\\|^2 / (2\\lambda))`
    (éq. (2.40) du livre).

    Parameters
    ----------
    x0 : ndarray, shape (m, p) ou (m,)
        Points cibles.
    x : ndarray, shape (n, p) ou (n,)
        Points d'apprentissage.
    lam : float
        Largeur (variance) du noyau.

    Returns
    -------
    ndarray, shape (m, n)
    """
    return np.exp(-squared_distances(x0, x) / (2 * lam)) / lam
