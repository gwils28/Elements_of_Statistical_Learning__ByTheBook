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
