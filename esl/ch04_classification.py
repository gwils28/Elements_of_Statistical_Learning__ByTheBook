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


def qda_predict(model, X):
    """Classe maximisant la fonction discriminante quadratique."""
    return model["classes"][np.argmax(qda_decision_function(model, X), axis=1)]


# ---------------------------------------------------------------------------
# 4.3.3  LDA de rang réduit (variables canoniques)
# ---------------------------------------------------------------------------

def reduced_rank_lda_fit(X, y):
    """Directions discriminantes de Fisher (section 4.3.3).

    On résout le problème aux valeurs propres généralisé ``B v = lambda W v``
    où ``W`` est la covariance intra-classe (celle de la LDA) et
    ``B = sum_k pi_k (mu_k - mu)(mu_k - mu)^T`` la covariance inter-classes.
    Les vecteurs sont normalisés par ``v^T W v = 1`` et rangés par valeurs
    propres décroissantes ; seuls les ``min(p, K-1)`` premiers sont gardés.

    Returns
    -------
    model : dict
        Celui de :func:`lda_fit` complété de ``center`` (moyenne globale
        pondérée), ``scalings`` (p, r) et ``eigvals`` (r,).
    """
    model = lda_fit(X, y)
    pi, M = model["priors"], model["means"]
    center = pi @ M
    Mc = M - center
    B = (Mc * pi[:, None]).T @ Mc
    vals, vecs = linalg.eigh(B, model["cov"])
    order = np.argsort(vals)[::-1]
    r = min(M.shape[1], len(pi) - 1)
    vecs = vecs[:, order[:r]]
    # orientation reproductible : première composante non nulle positive
    signs = np.sign(vecs[np.argmax(np.abs(vecs) > 1e-12, axis=0), np.arange(r)])
    model.update(center=center, scalings=vecs * signs, eigvals=vals[order[:r]])
    return model


def reduced_rank_lda_transform(model, X, L=None):
    """Coordonnées discriminantes (variables canoniques) ``Z = (X - mu) V_L``."""
    V = model["scalings"] if L is None else model["scalings"][:, :L]
    return (np.asarray(X, float) - model["center"]) @ V


def reduced_rank_lda_predict(model, X, L):
    """Classification au centroïde le plus proche dans le sous-espace de
    dimension ``L`` avec la correction ``log pi_k`` :
    ``argmax_k  -||z - zbar_k||^2 / 2 + log pi_k``."""
    Z = reduced_rank_lda_transform(model, X, L)
    C = reduced_rank_lda_transform(model, model["means"], L)
    d2 = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return model["classes"][np.argmax(-0.5 * d2 + np.log(model["priors"]), axis=1)]


# ---------------------------------------------------------------------------
# 4.4  Régression logistique
# ---------------------------------------------------------------------------

def _sigmoid(t):
    return 0.5 * (1 + np.tanh(0.5 * t))


def logistic_irls(X, y, max_iter=50, tol=1e-10, beta_init=None):
    """Régression logistique binaire par Newton-Raphson (= IRLS), éq. (4.26)-(4.28).

    À chaque itération : ``p = sigma(X beta)``, ``W = diag(p(1-p))``,
    réponse ajustée ``z = X beta + W^{-1}(y - p)`` puis
    ``beta <- (X^T W X)^{-1} X^T W z`` (moindres carrés pondérés).

    Parameters
    ----------
    X : array (N, p)
        Prédicteurs (une colonne de 1 est ajoutée).
    y : array (N,)
        Réponses 0/1.

    Returns
    -------
    res : dict
        ``coef`` (p+1,) (constante en premier), ``se`` erreurs standard
        (racine de la diagonale de ``(X^T W X)^{-1}``), ``z`` scores de Wald,
        ``deviance`` (= -2 log-vraisemblance), ``n_iter``, ``history`` (suite
        des log-vraisemblances), ``coef_path`` (itérés successifs).
    """
    Xa = _add_intercept(X)
    y = np.asarray(y, float)
    beta = np.zeros(Xa.shape[1]) if beta_init is None else np.array(beta_init, float)
    history, path = [], [beta.copy()]
    for it in range(1, max_iter + 1):
        eta = Xa @ beta
        p = _sigmoid(eta)
        history.append(float(np.sum(y * eta - np.logaddexp(0, eta))))
        w = np.clip(p * (1 - p), 1e-12, None)
        z = eta + (y - p) / w
        XtW = Xa.T * w
        new = linalg.solve(XtW @ Xa, XtW @ z, assume_a="pos")
        path.append(new.copy())
        done = np.max(np.abs(new - beta)) < tol * (1 + np.max(np.abs(beta)))
        beta = new
        if done:
            break
    eta = Xa @ beta
    p = _sigmoid(eta)
    loglik = float(np.sum(y * eta - np.logaddexp(0, eta)))
    history.append(loglik)
    w = p * (1 - p)
    cov = linalg.inv((Xa.T * w) @ Xa)
    se = np.sqrt(np.diag(cov))
    return {"coef": beta, "se": se, "z": beta / se, "deviance": -2 * loglik,
            "loglik": loglik, "n_iter": it, "history": np.array(history),
            "coef_path": np.array(path), "cov": cov}


def logistic_predict_proba(coef, X):
    """Probabilités ``Pr(Y = 1 | x)`` pour des coefficients (constante en tête)."""
    return _sigmoid(_add_intercept(X) @ coef)


def multilogit_newton(X, y, max_iter=100, tol=1e-9):
    """Régression logistique multinomiale par Newton-Raphson (exercice 4.4).

    Modèle (4.17) : la dernière classe sert de référence,
    ``log Pr(G=k|x)/Pr(G=K|x) = beta_k0 + beta_k^T x`` pour ``k < K``.
    Le vecteur des paramètres ``theta`` (de taille ``(p+1)(K-1)``) empile les
    ``beta_k`` ; gradient et hessienne sont calculés par blocs, et un pas de
    Newton est divisé par deux tant que la log-vraisemblance ne croît pas.

    Returns
    -------
    res : dict
        ``coef`` (K-1, p+1), ``classes``, ``loglik``, ``n_iter``,
        ``history``.
    """
    classes = np.unique(y)
    Xa = _add_intercept(X)
    N, q = Xa.shape
    K = len(classes)
    Y = indicator_matrix(y, classes)[:, :K - 1]

    def loglik(theta):
        E = np.column_stack([Xa @ theta.reshape(K - 1, q).T, np.zeros(N)])
        lse = np.logaddexp.reduce(E, axis=1)
        return float(np.sum(Y * E[:, :K - 1]) - np.sum(lse)), np.exp(E - lse[:, None])[:, :K - 1]

    theta = np.zeros((K - 1) * q)
    ll, P = loglik(theta)
    history = [ll]
    for it in range(1, max_iter + 1):
        grad = ((Y - P).T @ Xa).ravel()                       # blocs X^T (y_k - p_k)
        H = np.empty(((K - 1) * q, (K - 1) * q))
        for k in range(K - 1):
            for l in range(K - 1):
                w = P[:, k] * ((k == l) - P[:, l])
                H[k * q:(k + 1) * q, l * q:(l + 1) * q] = -(Xa.T * w) @ Xa
        step = linalg.solve(-H, grad, assume_a="pos")
        t = 1.0
        while True:
            new_ll, new_P = loglik(theta + t * step)
            if new_ll >= ll - 1e-12 or t < 1e-8:
                break
            t /= 2
        theta = theta + t * step
        done = abs(new_ll - ll) < tol * (1 + abs(ll))
        ll, P = new_ll, new_P
        history.append(ll)
        if done:
            break
    return {"coef": theta.reshape(K - 1, q), "classes": classes, "loglik": ll,
            "n_iter": it, "history": np.array(history)}


def multilogit_predict_proba(res, X):
    """Probabilités a posteriori (N, K) du modèle multinomial."""
    Xa = _add_intercept(X)
    E = np.column_stack([Xa @ res["coef"].T, np.zeros(Xa.shape[0])])
    E -= E.max(axis=1, keepdims=True)
    P = np.exp(E)
    return P / P.sum(axis=1, keepdims=True)


def multilogit_predict(res, X):
    """Classe de plus grande probabilité a posteriori."""
    return res["classes"][np.argmax(multilogit_predict_proba(res, X), axis=1)]


def soft_threshold(z, g):
    """Seuillage doux ``S(z, g) = sign(z) (|z| - g)_+``."""
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)
