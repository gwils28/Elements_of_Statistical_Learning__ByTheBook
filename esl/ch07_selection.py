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


def zero_one_loss(y, yhat):
    """Perte 0-1 ``I(y != yhat)``, élément par élément."""
    return (np.asarray(y) != np.asarray(yhat)).astype(float)


# ---------------------------------------------------------------------------
# Lisseurs linéaires  yhat = S y
# ---------------------------------------------------------------------------

def _design(X, intercept=True):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return X


def hat_matrix(X, intercept=True):
    """Matrice chapeau ``H = X (X^T X)^{-1} X^T`` des moindres carrés.

    Parameters
    ----------
    X : array (N, p)
        Prédicteurs.
    intercept : bool
        Ajoute une colonne de 1 si vrai.

    Returns
    -------
    H : array (N, N)
        Projecteur orthogonal sur l'espace des colonnes de X.
    """
    Z = _design(X, intercept)
    Q, _ = np.linalg.qr(Z)
    return Q @ Q.T


def ols_fit_predict(Xtr, ytr, Xte, intercept=True):
    """Ajuste les moindres carrés sur ``(Xtr, ytr)`` et prédit en ``Xte``."""
    Ztr = _design(Xtr, intercept)
    Zte = _design(Xte, intercept)
    beta, *_ = np.linalg.lstsq(Ztr, np.asarray(ytr, float), rcond=None)
    return Zte @ beta


def knn_regression(Xtr, ytr, Xte, k):
    """Régression par les k plus proches voisins (distance euclidienne).

    Le point d'apprentissage lui-même fait partie de son voisinage lorsqu'on
    prédit en un point d'apprentissage.
    """
    Xtr = _design(Xtr, False)
    Xte = _design(Xte, False)
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1, kind="stable")[:, :k]
    return np.asarray(ytr, float)[idx].mean(axis=1)


def knn_smoother_matrix(X, k):
    """Matrice de lissage S des k plus proches voisins aux points d'apprentissage.

    ``S[i, j] = 1/k`` si ``x_j`` est l'un des k plus proches voisins de ``x_i``
    (``x_i`` compris), 0 sinon. Sa trace vaut ``N/k`` (exercice 7.6).
    """
    X = _design(X, False)
    n = len(X)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1, kind="stable")[:, :k]
    S = np.zeros((n, n))
    np.put_along_axis(S, idx, 1.0 / k, axis=1)
    return S


def _reinsch_QR(x):
    """Matrices Q (n, n-2) et R (n-2, n-2) de la forme de Reinsch."""
    x = np.asarray(x, float)
    n = len(x)
    h = np.diff(x)
    Q = np.zeros((n, n - 2))
    R = np.zeros((n - 2, n - 2))
    for j in range(1, n - 1):
        c = j - 1
        Q[j - 1, c] = 1.0 / h[j - 1]
        Q[j, c] = -1.0 / h[j - 1] - 1.0 / h[j]
        Q[j + 1, c] = 1.0 / h[j]
        R[c, c] = (h[j - 1] + h[j]) / 3.0
        if c + 1 < n - 2:
            R[c, c + 1] = R[c + 1, c] = h[j] / 6.0
    return Q, R


def smoothing_spline_matrix(x, lam):
    """Matrice de lissage d'une spline cubique de lissage.

    Minimise ``sum (y_i - g(x_i))^2 + lam * int g''(t)^2 dt`` ; la solution
    aux nœuds vaut ``S y`` avec ``S = (I + lam K)^{-1}``, ``K = Q R^{-1} Q^T``
    (forme de Reinsch), calculée sous la forme équivalente
    ``S = I - lam Q (R + lam Q^T Q)^{-1} Q^T``.

    Parameters
    ----------
    x : array (N,)
        Abscisses **triées et distinctes**.
    lam : float
        Paramètre de lissage (>= 0).
    """
    x = np.asarray(x, float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x doit être strictement croissant")
    Q, R = _reinsch_QR(x)
    # forme équivalente numériquement stable (Green et Silverman) :
    # S = I - lam Q (R + lam Q^T Q)^{-1} Q^T
    S = np.eye(len(x)) - lam * Q @ np.linalg.solve(R + lam * Q.T @ Q, Q.T)
    return (S + S.T) / 2


def smoothing_spline_predict(x, y, lam, t):
    """Ajuste une spline de lissage sur ``(x, y)`` et l'évalue en ``t``.

    Utilise la représentation par valeurs ``g`` et dérivées secondes ``gamma``
    aux nœuds ; prolongement linéaire hors de ``[x_1, x_n]``.
    """
    x = np.asarray(x, float)
    order = np.argsort(x)
    x, y = x[order], np.asarray(y, float)[order]
    Q, R = _reinsch_QR(x)
    g = smoothing_spline_matrix(x, lam) @ y
    gamma = np.zeros(len(x))
    gamma[1:-1] = np.linalg.solve(R, Q.T @ g)
    t = np.atleast_1d(np.asarray(t, float))
    out = np.empty_like(t)
    h = np.diff(x)
    for m, tt in enumerate(t):
        if tt <= x[0]:
            slope = (g[1] - g[0]) / h[0] - h[0] * gamma[1] / 6.0
            out[m] = g[0] + slope * (tt - x[0])
        elif tt >= x[-1]:
            slope = (g[-1] - g[-2]) / h[-1] + h[-1] * gamma[-2] / 6.0
            out[m] = g[-1] + slope * (tt - x[-1])
        else:
            i = np.searchsorted(x, tt) - 1
            a, b = tt - x[i], x[i + 1] - tt
            out[m] = (a * g[i + 1] + b * g[i]) / h[i] - a * b / 6.0 * (
                (1 + a / h[i]) * gamma[i + 1] + (1 + b / h[i]) * gamma[i])
    return out


def df_trace(S):
    """Nombre effectif de paramètres d'un lisseur linéaire : ``trace(S)`` (7.32)."""
    return float(np.trace(S))


def loo_residuals(y, S):
    """Résidus de validation croisée « leave-one-out » par le raccourci (7.64).

    ``y_i - f_{-i}(x_i) = (y_i - yhat_i) / (1 - S_ii)``.
    """
    y = np.asarray(y, float)
    return (y - S @ y) / (1.0 - np.diag(S))


def gcv_score(y, S):
    """Validation croisée généralisée (7.52)."""
    y = np.asarray(y, float)
    n = len(y)
    return float(np.mean(((y - S @ y) / (1.0 - np.trace(S) / n)) ** 2))


def df_monte_carlo(fit, f, sigma, n_rep=2000, rng=None):
    """Estime ``df = sum_i Cov(yhat_i, y_i) / sigma^2`` (7.33) par simulation.

    Parameters
    ----------
    fit : callable
        ``fit(y) -> yhat`` (x fixés).
    f : array (N,)
        Vraie fonction de régression aux points d'apprentissage.
    sigma : float
        Écart-type du bruit gaussien additif.
    n_rep : int
        Nombre de répétitions.
    rng : np.random.RandomState
    """
    rng = np.random.RandomState(0) if rng is None else rng
    f = np.asarray(f, float)
    Y = f + sigma * rng.randn(n_rep, len(f))
    Yhat = np.array([fit(y) for y in Y])
    cov = ((Y - Y.mean(0)) * (Yhat - Yhat.mean(0))).sum(0) / (n_rep - 1)
    return float(cov.sum() / sigma ** 2)


# ---------------------------------------------------------------------------
# Critères analytiques
# ---------------------------------------------------------------------------

def cp_statistic(err, d, sigma2, n):
    """Statistique Cp (7.26) : ``err + 2 d sigma2 / N``."""
    return err + 2.0 * d * sigma2 / n


def aic_loglik(loglik, d, n):
    """AIC (7.29) : ``-2/N * loglik + 2 d / N``."""
    return -2.0 * loglik / n + 2.0 * d / n


def bic_gaussian(err, d, sigma2, n):
    """BIC gaussien (7.36) : ``N / sigma2 * (err + log(N) d sigma2 / N)``."""
    return n / sigma2 * (err + np.log(n) * d * sigma2 / n)


# ---------------------------------------------------------------------------
# Validation croisée et bootstrap
# ---------------------------------------------------------------------------

def kfold_indices(n, K, rng=None):
    """Partition aléatoire de ``{0, ..., n-1}`` en K blocs de tailles ~ égales."""
    rng = np.random.RandomState(0) if rng is None else rng
    perm = rng.permutation(n)
    return [np.sort(b) for b in np.array_split(perm, K)]


def cross_val_error(fit_predict, X, y, K=10, rng=None, loss=squared_loss,
                    folds=None):
    """Erreur de validation croisée à K blocs (7.48).

    Parameters
    ----------
    fit_predict : callable
        ``fit_predict(Xtr, ytr, Xte) -> prédictions en Xte``. Toute la
        procédure (y compris une éventuelle sélection de variables) doit être
        refaite à l'intérieur : c'est la « bonne manière » (section 7.10.2).
    X, y : arrays
    K : int
        Nombre de blocs (``K = N`` : leave-one-out).
    loss : callable
        Perte élément par élément.
    folds : list of arrays, optional
        Blocs imposés (sinon tirés par :func:`kfold_indices`).

    Returns
    -------
    cv : float
        Moyenne des pertes sur les N observations.
    se : float
        Écart-type des K erreurs de blocs divisé par sqrt(K).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    if folds is None:
        folds = kfold_indices(n, K, rng)
    losses = np.empty(n)
    fold_err = []
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        pred = fit_predict(X[tr], y[tr], X[te])
        losses[te] = loss(y[te], pred)
        fold_err.append(losses[te].mean())
    fold_err = np.array(fold_err)
    return float(losses.mean()), float(fold_err.std(ddof=1) / np.sqrt(len(folds)))


def no_information_rate(y, yhat, loss=zero_one_loss):
    """Taux d'erreur sans information (7.58) : moyenne de ``L(y_i, yhat_i')``
    sur tous les couples ``(i, i')``."""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(loss(y[:, None], yhat[None, :]).mean())


def bootstrap_error_estimates(fit_predict, X, y, B=100, rng=None,
                              loss=squared_loss):
    """Estimateurs bootstrap de l'erreur de prédiction (section 7.11).

    Returns
    -------
    dict
        ``err`` : erreur d'apprentissage ;
        ``boot`` : bootstrap naïf (7.54) ;
        ``loo1`` : bootstrap leave-one-out (7.56) ;
        ``632`` : estimateur .632 (7.57) ;
        ``632plus`` : estimateur .632+ (7.61) ;
        ``gamma`` : taux sans information (7.58) ; ``R`` : (7.60).
    """
    rng = np.random.RandomState(0) if rng is None else rng
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    fitted = fit_predict(X, y, X)
    err = float(loss(y, fitted).mean())
    sum_out = np.zeros(n)
    cnt_out = np.zeros(n)
    boot_total = 0.0
    for _ in range(B):
        idx = rng.randint(0, n, n)
        pred = fit_predict(X[idx], y[idx], X)
        L = loss(y, pred)
        boot_total += L.mean()
        out = np.ones(n, bool)
        out[idx] = False
        sum_out[out] += L[out]
        cnt_out[out] += 1
    ok = cnt_out > 0
    loo1 = float(np.mean(sum_out[ok] / cnt_out[ok]))
    gamma = no_information_rate(y, fitted, loss)
    loo1_c = min(loo1, gamma)
    if loo1_c > err and gamma > err:
        R = (loo1_c - err) / (gamma - err)
    else:
        R = 0.0
    w = 0.632 / (1.0 - 0.368 * R)
    return {
        "err": err,
        "boot": boot_total / B,
        "loo1": loo1,
        "632": 0.368 * err + 0.632 * loo1,
        "632plus": (1 - w) * err + w * loo1_c,
        "gamma": gamma,
        "R": R,
    }


# ---------------------------------------------------------------------------
# Sélection de sous-ensembles
# ---------------------------------------------------------------------------

def rss(X, y, cols):
    """Somme des carrés résiduels de la régression (avec constante) sur ``cols``."""
    Z = _design(np.asarray(X, float)[:, list(cols)], True) if len(cols) else \
        np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    r = y - Z @ beta
    return float(r @ r)


def best_subset(X, y, max_size=None):
    """Meilleur sous-ensemble de chaque taille (recherche exhaustive).

    Returns
    -------
    list of tuple
        Élément k : ``(cols, rss)`` pour le meilleur sous-ensemble de taille k
        (k = 0, ..., max_size).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    max_size = p if max_size is None else max_size
    out = []
    for k in range(max_size + 1):
        best = min(((c, rss(X, y, c)) for c in combinations(range(p), k)),
                   key=lambda t: t[1])
        out.append(best)
    return out


def forward_stepwise(X, y, max_size=None):
    """Ordre d'entrée des variables par sélection pas à pas ascendante.

    À chaque étape on ajoute la variable qui diminue le plus la RSS.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    max_size = p if max_size is None else max_size
    active, rest = [], list(range(p))
    for _ in range(max_size):
        j = min(rest, key=lambda c: rss(X, y, active + [c]))
        active.append(j)
        rest.remove(j)
    return active


# ---------------------------------------------------------------------------
# Régression logistique et splines naturelles (exemple des phonèmes)
# ---------------------------------------------------------------------------

def natural_spline_basis(x, knots):
    """Base de splines cubiques naturelles (éq. (5.4)-(5.5) du livre).

    Pour K nœuds, renvoie les K fonctions ``N_1 = 1, N_2 = x,
    N_{k+2} = d_k - d_{K-1}`` évaluées en ``x`` (tableau (len(x), K)).
    """
    x = np.asarray(x, float)
    xi = np.asarray(knots, float)
    K = len(xi)
    cols = [np.ones_like(x), x]

    def d(k):
        return (np.maximum(x - xi[k], 0) ** 3 - np.maximum(x - xi[-1], 0) ** 3) \
            / (xi[-1] - xi[k])

    for k in range(K - 2):
        cols.append(d(k) - d(K - 2))
    return np.column_stack(cols)


def logistic_regression(X, y, ridge=1e-8, max_iter=100, tol=1e-10):
    """Régression logistique binaire par Newton-Raphson (IRLS).

    Parameters
    ----------
    X : array (N, p)
        Matrice de dessin (inclure soi-même la constante si voulu).
    y : array (N,)
        Réponses 0/1.
    ridge : float
        Petite pénalité quadratique pour la stabilité numérique.

    Returns
    -------
    beta : array (p,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        grad = X.T @ (y - p) - ridge * beta
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta
