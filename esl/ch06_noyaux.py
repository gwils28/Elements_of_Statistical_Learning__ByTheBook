"""Méthodes de lissage par noyaux - chapitre 6 d'ESL.

Ce module regroupe les outils implémentés « à la main » dans les notebooks du
chapitre 6 :

* noyaux de lissage (Epanechnikov, tri-cube, gaussien) et poids
  :math:`K_\\lambda(x_0, x_i)` à fenêtre métrique ou à k plus proches voisins ;
* moyenne mobile des k-ppv, estimateur de Nadaraya-Watson ;
* régression polynomiale locale en dimension quelconque (monômes au choix),
  noyaux équivalents, matrice de lissage, critères LOO / GCV / :math:`C_p` ;
* régression logistique (globale et locale) par IRLS ;
* estimation de densité par noyau, classifieur par densités, Bayes naïf ;
* fonctions de base radiales, mélanges gaussiens (algorithme EM) ;
* analyse discriminante linéaire locale (exercice 6.12).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Noyaux
# ---------------------------------------------------------------------------


def epanechnikov(t):
    """Noyau d'Epanechnikov :math:`D(t) = \\tfrac34 (1 - t^2)` sur [-1, 1] (éq. 6.4).

    Parameters
    ----------
    t : array_like
        Distances réduites :math:`|x - x_0| / \\lambda`.

    Returns
    -------
    ndarray
        Valeurs de :math:`D(t)` (nulles hors de [-1, 1]). Intègre à 1.
    """
    t = np.asarray(t, dtype=float)
    return np.where(np.abs(t) <= 1, 0.75 * (1 - t ** 2), 0.0)


def tricube(t):
    """Noyau tri-cube :math:`D(t) = (1 - |t|^3)^3` sur [-1, 1] (éq. 6.6).

    Non normalisé : son intégrale vaut 81/70.
    """
    a = np.abs(np.asarray(t, dtype=float))
    return np.where(a <= 1, (1 - a ** 3) ** 3, 0.0)


def gaussian(t):
    """Noyau gaussien :math:`D(t) = \\varphi(t)` (densité de la loi N(0, 1))."""
    t = np.asarray(t, dtype=float)
    return np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)


KERNELS = {"epanechnikov": epanechnikov, "tricube": tricube,
           "gaussian": gaussian}


def _kernel(kernel):
    """Renvoie la fonction noyau associée à un nom (ou la fonction elle-même)."""
    if callable(kernel):
        return kernel
    try:
        return KERNELS[kernel]
    except KeyError:
        raise ValueError("Noyau inconnu : %r" % (kernel,))


def _as_2d(x):
    """Met un tableau de points sous la forme (n, p)."""
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def distances(x0, x, A=None):
    """Distances entre points cibles et points d'apprentissage.

    Parameters
    ----------
    x0 : array_like, shape (m,) ou (m, p)
        Points cibles.
    x : array_like, shape (n,) ou (n, p)
        Points d'apprentissage.
    A : ndarray (p, p), optional
        Matrice semi-définie positive de la métrique
        :math:`\\sqrt{(x - x_0)^T A (x - x_0)}` (cf. éq. 6.14). Par défaut,
        l'identité (distance euclidienne).

    Returns
    -------
    ndarray, shape (m, n)
    """
    x0, x = _as_2d(x0), _as_2d(x)
    diff = x[None, :, :] - x0[:, None, :]
    if A is None:
        d2 = np.einsum("mnp,mnp->mn", diff, diff)
    else:
        d2 = np.einsum("mnp,pq,mnq->mn", diff, np.asarray(A, float), diff)
    return np.sqrt(np.maximum(d2, 0.0))


def kernel_weights(x0, x, lam=None, k=None, kernel="epanechnikov", A=None):
    """Poids :math:`K_\\lambda(x_0, x_i) = D(\\|x_i - x_0\\| / h(x_0))` (éq. 6.5).

    Parameters
    ----------
    x0, x : array_like
        Points cibles (m,) / (m, p) et points d'apprentissage (n,) / (n, p).
    lam : float, optional
        Largeur métrique constante :math:`h(x_0) = \\lambda`.
    k : int, optional
        Fenêtre adaptative : :math:`h(x_0)` est la distance au k-ième plus
        proche voisin de :math:`x_0` (exclusif avec ``lam``).
    kernel : str ou callable
        ``"epanechnikov"``, ``"tricube"``, ``"gaussian"`` ou une fonction D.
    A : ndarray, optional
        Métrique (voir :func:`distances`).

    Returns
    -------
    ndarray, shape (m, n)
    """
    if (lam is None) == (k is None):
        raise ValueError("Préciser exactement un des deux paramètres lam ou k.")
    d = distances(x0, x, A)
    if k is not None:
        h = np.partition(d, k - 1, axis=1)[:, k - 1]
        h = np.where(h > 0, h, 1e-12)
    else:
        h = np.full(d.shape[0], float(lam))
    return _kernel(kernel)(d / h[:, None])


# ---------------------------------------------------------------------------
# Moyennes locales
# ---------------------------------------------------------------------------


def knn_mean(x0, x, y, k):
    """Moyenne des k plus proches voisins (éq. 6.1)."""
    d = distances(x0, x)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.asarray(y, float)[idx].mean(axis=1)


def nadaraya_watson(x0, x, y, lam=None, k=None, kernel="epanechnikov", A=None):
    """Moyenne pondérée de Nadaraya-Watson (éq. 6.2).

    Renvoie ``nan`` aux points cibles dont la fenêtre ne contient aucun point.
    """
    W = kernel_weights(x0, x, lam=lam, k=k, kernel=kernel, A=A)
    s = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, W @ np.asarray(y, float) / np.where(s > 0, s, 1), np.nan)


# ---------------------------------------------------------------------------
# Régression polynomiale locale
# ---------------------------------------------------------------------------


def monomials_of_degree(p, degree):
    """Liste des exposants (tuples de longueur p) de degré total <= ``degree``.

    Le monôme constant ``(0, ..., 0)`` est toujours en tête.
    """
    out = [(0,) * p]
    if degree >= 1:
        for j in range(p):
            e = [0] * p
            e[j] = 1
            out.append(tuple(e))
    if degree >= 2:
        for j in range(p):
            for l in range(j, p):
                e = [0] * p
                e[j] += 1
                e[l] += 1
                out.append(tuple(e))
    if degree >= 3:
        if p != 1:
            raise ValueError("Degré > 2 seulement disponible en dimension 1.")
        out += [(d,) for d in range(3, degree + 1)]
    return out
