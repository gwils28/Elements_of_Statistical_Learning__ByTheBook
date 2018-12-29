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
