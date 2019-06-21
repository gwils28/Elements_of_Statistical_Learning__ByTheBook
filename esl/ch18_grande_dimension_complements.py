"""Chapitre 18 (p >> N) - outils pour les exercices 18.11 à 18.20.

Contenu :

* régression logistique (avec ou sans pénalité quadratique) et données séparables ;
* réduction « SVD » X = R V^T et validation croisée de la régression ridge ;
* régression logistique ridge à partir d'une matrice de noyau K ;
* 1-plus-proche-voisin pondéré par les distances (score continu) et courbe ROC ;
* ACP à noyau et projection d'un nouveau point ;
* tests multiples : t-statistiques, permutations, Bonferroni, Benjamini-Hochberg,
  estimation « plug-in » du FDR, estimation de pi0.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Régression logistique
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Fonction logistique 1 / (1 + exp(-z)), calculée de façon stable."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logistic_loglik(beta0, beta, X, y):
    """Log-vraisemblance du modèle logit Pr(Y=1|x) = beta0 + x^T beta.

    Parameters
    ----------
    beta0 : float
        Ordonnée à l'origine.
    beta : ndarray (p,)
        Coefficients.
    X : ndarray (N, p)
    y : ndarray (N,)  valeurs 0/1.

    Returns
    -------
    float
        sum_i [ y_i eta_i - log(1 + exp(eta_i)) ]  (toujours < 0).
    """
    eta = beta0 + X @ np.atleast_1d(beta)
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def logistic_gradient_ascent(X, y, n_iter=10000, lr=0.5, record=None):
    """Montée de gradient sur la log-vraisemblance logistique (sans pénalité).

    Sert à visualiser la divergence des coefficients quand les classes sont
    séparables (exercice 18.11).

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,) 0/1
    n_iter : int
    lr : float
        Pas (appliqué au gradient moyen).
    record : array-like d'entiers, optionnel
        Itérations auxquelles mémoriser (beta0, beta, loglik).

    Returns
    -------
    iters : ndarray
    path : ndarray (len(iters), p + 1)  - colonnes (beta0, beta_1..beta_p)
    loglik : ndarray (len(iters),)
    """
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    if record is None:
        record = np.unique(np.logspace(0, np.log10(n_iter), 60).astype(int))
    record = set(int(r) for r in record)
    b0, b = 0.0, np.zeros(p)
    iters, path, ll = [], [], []
    for t in range(1, n_iter + 1):
        r = y - sigmoid(b0 + X @ b)
        b0 += lr * r.mean()
        b += lr * (X.T @ r) / N
        if t in record:
            iters.append(t)
            path.append(np.r_[b0, b])
            ll.append(logistic_loglik(b0, b, X, y))
    return np.array(iters), np.array(path), np.array(ll)


def logistic_ridge(X, y, lam, max_iter=100, tol=1e-10):
    """Régression logistique avec pénalité quadratique (Newton-Raphson).

    Minimise  - sum_i log-vraisemblance_i + lam * ||beta||^2  ; l'ordonnée à
    l'origine beta0 n'est pas pénalisée (forme (18.16) du livre).

    Parameters
    ----------
    X : ndarray (N, p)
    y : ndarray (N,) 0/1
    lam : float > 0

    Returns
    -------
    beta0 : float
    beta : ndarray (p,)
    """
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    Xa = np.c_[np.ones(N), X]
    P = 2.0 * lam * np.eye(p + 1)
    P[0, 0] = 0.0
    w = np.zeros(p + 1)
    for _ in range(max_iter):
        pr = sigmoid(Xa @ w)
        grad = -Xa.T @ (y - pr) + P @ w
        H = Xa.T @ (Xa * (pr * (1 - pr))[:, None]) + P
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w[0], w[1:]


# ---------------------------------------------------------------------------
# Réduction SVD et régression ridge
# ---------------------------------------------------------------------------

def svd_reduce(X):
    """Écrit X = R V^T avec R = U D (N x N) et V (p x N) à colonnes orthonormées.

    Returns
    -------
    R : ndarray (N, min(N,p))
    V : ndarray (p, min(N,p))
    """
    U, d, Vt = np.linalg.svd(np.asarray(X, dtype=float), full_matrices=False)
    return U * d, Vt.T


def ridge_fit(X, y, lam):
    """Régression ridge avec ordonnée à l'origine non pénalisée.

    Minimise ||y - beta0 - X beta||^2 + lam ||beta||^2. Le calcul passe par la
    forme « duale » beta = Xc^T (Xc Xc^T + lam I)^{-1} yc (Xc, yc centrés), qui ne
    coûte que O(N^2 p).

    Returns
    -------
    beta0 : float
    beta : ndarray (p,)
    """
    X = np.asarray(X, dtype=float)
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    N = X.shape[0]
    beta = Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(N), yc)
    return ym - xm @ beta, beta


def kfold_indices(N, n_folds, rng):
    """Attribue aléatoirement chacune des N observations à l'un des n_folds blocs."""
    return rng.permutation(np.arange(N) % n_folds)
