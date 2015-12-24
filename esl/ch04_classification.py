"""Méthodes linéaires de classification - chapitre 4 d'ESL.

Implémentations explicites (numpy / scipy) de :

* la régression linéaire sur matrice indicatrice (section 4.2) ;
* l'analyse discriminante linéaire (LDA), quadratique (QDA) et régularisée
  (RDA) (section 4.3) ;
* la LDA de rang réduit / variables canoniques de Fisher (section 4.3.3) ;
* la régression logistique binaire par Newton-Raphson / IRLS (section 4.4.1),
  multinomiale par Newton-Raphson (exercice 4.4) et pénalisée L1 par
  descente par coordonnées (section 4.4.4) ;
* le perceptron de Rosenblatt et l'hyperplan séparateur optimal
  (section 4.5).

Convention : les étiquettes de classe ``y`` peuvent être quelconques ; elles
sont ordonnées par ``np.unique`` et stockées dans ``model["classes"]``.
"""

import numpy as np
from scipy import linalg, optimize


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

def indicator_matrix(y, classes=None):
    """Matrice indicatrice (codage « un parmi K ») des réponses.

    Parameters
    ----------
    y : array (N,)
        Étiquettes.
    classes : array (K,), optional
        Ordre des classes (par défaut ``np.unique(y)``).

    Returns
    -------
    Y : array (N, K)
        ``Y[i, k] = 1`` si ``y[i] == classes[k]``, 0 sinon.
    """
    classes = np.unique(y) if classes is None else np.asarray(classes)
    return (np.asarray(y)[:, None] == classes[None, :]).astype(float)


def error_rate(y_true, y_pred):
    """Taux d'erreur de classification (proportion de prédictions fausses)."""
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def _add_intercept(X):
    X = np.asarray(X, float)
    return np.column_stack([np.ones(X.shape[0]), X])


# ---------------------------------------------------------------------------
# 4.2  Régression sur matrice indicatrice
# ---------------------------------------------------------------------------

def indicator_regression_fit(X, y):
    """Régression linéaire (avec constante) de la matrice indicatrice sur X.

    Returns
    -------
    model : dict
        ``B`` (p+1, K) coefficients (première ligne = constantes) et
        ``classes``.
    """
    classes = np.unique(y)
    Y = indicator_matrix(y, classes)
    Xa = _add_intercept(X)
    B, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    return {"B": B, "classes": classes}


def indicator_regression_predict(model, X, return_fitted=False):
    """Classe de plus grande composante ajustée, règle (4.4) du livre."""
    F = _add_intercept(X) @ model["B"]
    pred = model["classes"][np.argmax(F, axis=1)]
    return (pred, F) if return_fitted else pred


# ---------------------------------------------------------------------------
# 4.3  LDA, QDA, RDA
# ---------------------------------------------------------------------------

def _class_stats(X, y):
    X = np.asarray(X, float)
    classes = np.unique(y)
    N, p = X.shape
    counts = np.array([np.sum(y == k) for k in classes])
    means = np.array([X[y == k].mean(axis=0) for k in classes])
    return X, classes, counts, means


def lda_fit(X, y, priors=None):
    """Analyse discriminante linéaire (section 4.3).

    Estimations (4.10) du livre : proportions ``pi_k = N_k / N``, moyennes de
    classe, covariance intra-classe commune ``sum_k sum_{g_i=k}
    (x_i - mu_k)(x_i - mu_k)^T / (N - K)``.

    Returns
    -------
    model : dict
        ``classes``, ``priors``, ``means`` (K, p), ``cov`` (p, p).
    """
    X, classes, counts, means = _class_stats(X, y)
    N, K = X.shape[0], len(classes)
    R = X - means[np.searchsorted(classes, y)]
    cov = R.T @ R / (N - K)
    pri = counts / N if priors is None else np.asarray(priors, float)
    return {"classes": classes, "priors": pri, "means": means, "cov": cov,
            "counts": counts}


def lda_decision_function(model, X):
    """Fonctions discriminantes linéaires (4.10) :
    ``delta_k(x) = x^T S^{-1} mu_k - mu_k^T S^{-1} mu_k / 2 + log pi_k``.

    Returns
    -------
    D : array (N, K)
    """
    A = linalg.solve(model["cov"], model["means"].T, assume_a="pos")  # (p, K)
    const = -0.5 * np.sum(model["means"].T * A, axis=0) + np.log(model["priors"])
    return np.asarray(X, float) @ A + const


def lda_predict(model, X):
    """Classe maximisant la fonction discriminante linéaire."""
    return model["classes"][np.argmax(lda_decision_function(model, X), axis=1)]


def qda_fit(X, y, alpha=1.0, priors=None):
    """Analyse discriminante quadratique, éventuellement régularisée (RDA).

    Chaque classe a sa propre covariance ``S_k`` (dénominateur ``N_k - 1``).
    Avec ``alpha < 1`` on utilise la covariance régularisée (4.13) du livre
    ``S_k(alpha) = alpha S_k + (1 - alpha) S`` où ``S`` est la covariance
    commune de la LDA (``alpha = 0`` redonne la LDA, ``alpha = 1`` la QDA).

    Returns
    -------
    model : dict
        ``classes``, ``priors``, ``means``, ``covs`` (K, p, p).
    """
    X, classes, counts, means = _class_stats(X, y)
    pooled = lda_fit(X, y)["cov"]
    covs = []
    for k, c in enumerate(classes):
        R = X[y == c] - means[k]
        Sk = R.T @ R / (counts[k] - 1)
        covs.append(alpha * Sk + (1 - alpha) * pooled)
    pri = counts / X.shape[0] if priors is None else np.asarray(priors, float)
    return {"classes": classes, "priors": pri, "means": means,
            "covs": np.array(covs)}


def qda_decision_function(model, X):
    """Fonctions discriminantes quadratiques (4.12) :
    ``-log|S_k|/2 - (x - mu_k)^T S_k^{-1} (x - mu_k)/2 + log pi_k``."""
    X = np.asarray(X, float)
    D = np.empty((X.shape[0], len(model["classes"])))
    for k in range(len(model["classes"])):
        L = linalg.cholesky(model["covs"][k], lower=True)
        Zk = linalg.solve_triangular(L, (X - model["means"][k]).T, lower=True)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        D[:, k] = -0.5 * logdet - 0.5 * np.sum(Zk ** 2, axis=0) + np.log(model["priors"][k])
    return D
