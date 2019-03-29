"""Chapitre 17 - Modèles graphiques non orientés.

Ce module regroupe les outils du chapitre :

* vocabulaire des graphes (matrice d'adjacence, cliques maximales,
  séparation) et dessin d'un graphe avec matplotlib ;
* modèles graphiques gaussiens : estimation à graphe connu
  (algorithme 17.1 du livre), lasso par descente de coordonnées
  (éq. 17.26), graphical lasso (algorithme 17.2), algorithme EM avec
  données manquantes (exercice 17.9) ;
* modèles de Ising (machines de Boltzmann entièrement visibles) pour de
  petites valeurs de ``p`` : énumération des ``2^p`` états, fonction de
  partition, maximum de vraisemblance par Newton, pseudo-vraisemblance,
  régressions logistiques nœud par nœud.

Conventions pour le modèle de Ising : les variables valent 0 ou 1 ; les
paramètres sont un vecteur ``theta0`` (arêtes vers le nœud constant
``X_0 = 1``) et une matrice symétrique ``Theta`` de diagonale nulle, de sorte
que ::

    log p(x) = sum_j theta0[j] x_j + sum_{j<k} Theta[j, k] x_j x_k - Phi.
"""

import itertools

import numpy as np

# ---------------------------------------------------------------------------
# 1. Graphes
# ---------------------------------------------------------------------------


def adjacency_from_edges(p, edges):
    """Matrice d'adjacence booléenne (p, p) à partir d'une liste d'arêtes.

    Parameters
    ----------
    p : int
        Nombre de sommets (numérotés 0, ..., p-1).
    edges : iterable de couples (j, k)
        Arêtes non orientées.

    Returns
    -------
    A : ndarray (p, p) de booléens, symétrique, diagonale nulle.
    """
    A = np.zeros((p, p), dtype=bool)
    for j, k in edges:
        if j != k:
            A[j, k] = A[k, j] = True
    return A


def edges_from_adjacency(A):
    """Liste des arêtes ``(j, k)`` avec ``j < k`` d'une matrice d'adjacence."""
    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    return [(j, k) for j in range(p) for k in range(j + 1, p) if A[j, k]]


def support_graph(Theta, tol=1e-8):
    """Graphe (adjacence) des coefficients hors diagonale non nuls de ``Theta``."""
    A = np.abs(np.asarray(Theta)) > tol
    np.fill_diagonal(A, False)
    return A | A.T


def maximal_cliques(A):
    """Cliques maximales d'un graphe (algorithme de Bron–Kerbosch avec pivot).

    Returns
    -------
    list de tuples triés, eux-mêmes triés par ordre lexicographique.
    """
    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    nbrs = [set(np.flatnonzero(A[j])) for j in range(p)]
    cliques = []

    def bk(R, P, X):
        if not P and not X:
            cliques.append(tuple(sorted(int(r) for r in R)))
            return
        pivot = max(P | X, key=lambda u: len(nbrs[u] & P))
        for v in list(P - nbrs[pivot]):
            bk(R | {v}, P & nbrs[v], X & nbrs[v])
            P = P - {v}
            X = X | {v}

    bk(set(), set(range(p)), set())
    return sorted(cliques)


def separates(A, a, b, c):
    """Teste si l'ensemble ``c`` sépare les ensembles ``a`` et ``b``.

    ``c`` sépare ``a`` et ``b`` si tout chemin d'un sommet de ``a`` à un
    sommet de ``b`` passe par un sommet de ``c`` ; autrement dit, après
    suppression des sommets de ``c``, aucun sommet de ``b`` n'est atteignable
    depuis ``a`` (parcours en largeur).
    """
    A = np.asarray(A, dtype=bool)
    a, b, c = set(a), set(b), set(c)
    if (a & b) - c:
        return False
    start = a - c
    seen = set(start)
    stack = list(start)
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(A[u]):
            v = int(v)
            if v in c or v in seen:
                continue
            if v in b:
                return False
            seen.add(v)
            stack.append(v)
    return True


def connected_components(A):
    """Composantes connexes (liste de listes triées de sommets)."""
    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    label = -np.ones(p, dtype=int)
    comps = []
    for s in range(p):
        if label[s] >= 0:
            continue
        stack, comp = [s], []
        label[s] = len(comps)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(A[u]):
                if label[v] < 0:
                    label[v] = len(comps)
                    stack.append(int(v))
        comps.append(sorted(comp))
    return comps


def circle_layout(p, start=np.pi / 2):
    """Positions de ``p`` sommets régulièrement répartis sur le cercle unité
    (sens horaire, premier sommet en haut)."""
    ang = start - 2 * np.pi * np.arange(p) / p
    return np.c_[np.cos(ang), np.sin(ang)]


def draw_graph(A, labels=None, pos=None, weights=None, ax=None, title=None,
               node_color="#f0efec", edge_color="#2a78d6", node_size=0.11,
               fontsize=10, cmap=None, vmax=None):
    """Dessine un graphe non orienté avec matplotlib.

    Parameters
    ----------
    A : ndarray (p, p) booléen
        Matrice d'adjacence.
    labels : liste de str, optionnelle
        Étiquettes des sommets (par défaut ``X1, ..., Xp``).
    pos : ndarray (p, 2), optionnel
        Positions des sommets (par défaut disposition circulaire).
    weights : ndarray (p, p), optionnel
        Poids des arêtes (par ex. corrélations partielles). L'épaisseur est
        proportionnelle à ``|poids|`` ; si ``cmap`` est donné, la couleur
        code le signe et l'intensité.
    ax : axes matplotlib, optionnel.
    title : str, optionnel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    if labels is None:
        labels = ["$X_{%d}$" % (j + 1) for j in range(p)]
    if pos is None:
        pos = circle_layout(p)
    pos = np.asarray(pos, dtype=float)
    if ax is None:
        ax = plt.gca()
    if weights is not None and vmax is None:
        w = np.abs(np.asarray(weights)[A])
        vmax = w.max() if w.size else 1.0
    for j, k in edges_from_adjacency(A):
        lw, col = 1.6, edge_color
        if weights is not None:
            wjk = weights[j, k]
            lw = 0.6 + 4.0 * abs(wjk) / vmax
            if cmap is not None:
                col = cmap(0.5 + 0.5 * np.clip(wjk / vmax, -1, 1))
        ax.plot(pos[[j, k], 0], pos[[j, k], 1], color=col, lw=lw,
                solid_capstyle="round", zorder=1)
    for j in range(p):
        ax.add_patch(Circle(pos[j], node_size, facecolor=node_color,
                            edgecolor="#52514e", lw=1.0, zorder=2))
        ax.text(pos[j, 0], pos[j, 1], labels[j], ha="center", va="center",
                fontsize=fontsize, zorder=3)
    pad = node_size + 0.08
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# 2. Modèles graphiques gaussiens
# ---------------------------------------------------------------------------


def soft_threshold(x, t):
    """Opérateur de seuillage doux ``S(x, t) = sign(x) (|x| - t)_+`` (éq. 17.27)."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def partial_correlations(Theta):
    """Corrélations partielles ``rho_{jk|reste} = -theta_jk / sqrt(theta_jj theta_kk)``.

    La diagonale est fixée à 1 (convention).
    """
    Theta = np.asarray(Theta, dtype=float)
    d = 1.0 / np.sqrt(np.diag(Theta))
    R = -Theta * np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def gaussian_loglik(Theta, S):
    """Log-vraisemblance profilée (éq. 17.11) : ``log det Theta - tr(S Theta)``."""
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return -np.inf
    return logdet - np.trace(S @ Theta)


def empirical_covariance(X):
    """Covariance empirique ``S`` normalisée par ``N`` (éq. 17.10)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / X.shape[0]


def _theta_from_w(W, B):
    """Reconstruit Theta = W^{-1} à partir des coefficients de régression.

    ``B[:, j]`` contient le vecteur beta (de longueur p, composante j
    ignorée) obtenu pour la variable j. On applique l'étape 3 des
    algorithmes 17.1 et 17.2 : 1/theta_jj = w_jj - w_12^T beta et
    theta_12 = -beta theta_jj.
    """
    p = W.shape[0]
    Theta = np.zeros_like(W)
    for j in range(p):
        idx = np.arange(p) != j
        beta = B[idx, j]
        theta_jj = 1.0 / (W[j, j] - W[idx, j] @ beta)
        Theta[j, j] = theta_jj
        Theta[idx, j] = -beta * theta_jj
    return (Theta + Theta.T) / 2


def fit_known_graph(S, A, tol=1e-10, max_iter=1000, return_n_iter=False):
    """Algorithme 17.1 : estimation d'un modèle graphique gaussien à
    structure connue par régressions modifiées.

    Maximise ``log det Theta - tr(S Theta)`` sous la contrainte
    ``theta_jk = 0`` pour toute paire ``(j, k)`` non reliée dans ``A``.

    Parameters
    ----------
    S : ndarray (p, p)
        Covariance empirique.
    A : ndarray (p, p) booléen
        Graphe imposé (``A[j, k]`` vrai si l'arête est présente).
    tol : float
        Critère d'arrêt sur la variation maximale de ``W`` lors d'un cycle.
    max_iter : int
        Nombre maximal de cycles sur les p variables.

    Returns
    -------
    W : ndarray (p, p)
        Covariance estimée ``Sigma_hat``.
    Theta : ndarray (p, p)
        Matrice de précision estimée ``Sigma_hat^{-1}``.
    n_iter : int (si ``return_n_iter``)
    """
    S = np.asarray(S, dtype=float)
    A = np.asarray(A, dtype=bool)
    p = S.shape[0]
    W = S.copy()
    B = np.zeros((p, p))
    for it in range(1, max_iter + 1):
        W_old = W.copy()
        for j in range(p):
            idx = np.flatnonzero(np.arange(p) != j)
            nb = idx[A[j, idx]]  # voisins de j (arêtes libres)
            beta = np.zeros(p)
            if nb.size:
                beta[nb] = np.linalg.solve(W[np.ix_(nb, nb)], S[nb, j])
            B[:, j] = beta
            w12 = W[np.ix_(idx, idx)] @ beta[idx]
            W[idx, j] = w12
            W[j, idx] = w12
        if np.max(np.abs(W - W_old)) < tol:
            break
    Theta = _theta_from_w(W, B)
    if return_n_iter:
        return W, Theta, it
    return W, Theta


def lasso_cd(V, s, lam, beta=None, tol=1e-10, max_iter=5000):
    """Lasso « modifié » par descente de coordonnées cyclique (éq. 17.26).

    Minimise ``1/2 beta^T V beta - s^T beta + lam * ||beta||_1``, dont les
    équations d'estimation sont ``V beta - s + lam Sign(beta) = 0``
    (éq. 17.23 et 17.25). Avec ``V = Z^T Z / N`` et ``s = Z^T y / N`` on
    retrouve le lasso usuel ``(1/2N) ||y - Z beta||^2 + lam ||beta||_1``.

    Parameters
    ----------
    V : ndarray (q, q) symétrique semi-définie positive, diagonale > 0.
    s : ndarray (q,)
    lam : float >= 0
    beta : ndarray (q,), optionnel - point de départ (démarrage à chaud).

    Returns
    -------
    beta : ndarray (q,)
    """
    V = np.asarray(V, dtype=float)
    s = np.asarray(s, dtype=float)
    q = s.size
    beta = np.zeros(q) if beta is None else np.array(beta, dtype=float)
    for _ in range(max_iter):
        delta = 0.0
        for j in range(q):
            r = s[j] - V[j] @ beta + V[j, j] * beta[j]
            new = soft_threshold(r, lam) / V[j, j]
            delta = max(delta, abs(new - beta[j]))
            beta[j] = new
        if delta < tol:
            break
    return beta


def graphical_lasso(S, lam, tol=1e-6, max_iter=500, penalize_diagonal=True,
                    W_init=None, B_init=None, return_n_iter=False,
                    return_beta=False):
    """Algorithme 17.2 : graphical lasso (Friedman, Hastie, Tibshirani 2008).

    Maximise ``log det Theta - tr(S Theta) - lam * ||Theta||_1`` (éq. 17.21).
    Si ``penalize_diagonal=False``, seuls les coefficients hors diagonale
    sont pénalisés (variante de la note 3 du livre, utilisée par
    ``sklearn.covariance.graphical_lasso``).

    Parameters
    ----------
    S : ndarray (p, p)
        Covariance empirique.
    lam : float >= 0, ou ndarray (p, p) de pénalités par arête
        (``np.inf`` force l'absence d'arête, cf. algorithme 17.1).
    tol : float
        Critère d'arrêt : variation moyenne de ``W`` relative à la moyenne
        des ``|s_jk|`` hors diagonale.
    W_init, B_init : démarrage à chaud (sortie d'un appel précédent).

    Returns
    -------
    W : ndarray (p, p) - covariance estimée.
    Theta : ndarray (p, p) - précision estimée.
    """
    S = np.asarray(S, dtype=float)
    p = S.shape[0]
    Lam = np.broadcast_to(np.asarray(lam, dtype=float), (p, p))
    diag_shift = np.diag(Lam) if penalize_diagonal else np.zeros(p)
    if W_init is None:
        W = S.copy()
        W[np.diag_indices(p)] += diag_shift
    else:
        W = np.array(W_init, dtype=float)
        W[np.diag_indices(p)] = np.diag(S) + diag_shift
    B = np.zeros((p, p)) if B_init is None else np.array(B_init, dtype=float)
    off = np.abs(S[~np.eye(p, dtype=bool)]).mean() if p > 1 else 1.0
    scale = off if off > 0 else 1.0
    for it in range(1, max_iter + 1):
        W_old = W.copy()
        for j in range(p):
            idx = np.flatnonzero(np.arange(p) != j)
            V = W[np.ix_(idx, idx)]
            lj = Lam[idx, j]
            free = np.isfinite(lj)
            beta = np.zeros(p - 1)
            if free.any():
                f = np.flatnonzero(free)
                Vf = V[np.ix_(f, f)]
                # lasso à pénalités éventuellement différentes : on remet à
                # l'échelle chaque coordonnée (pénalité commune si lam scalaire)
                bstart = B[idx, j][f]
                beta[f] = _lasso_cd_weighted(Vf, S[idx, j][f], lj[f], bstart)
            B[idx, j] = beta
            B[j, j] = 0.0
            w12 = V @ beta
            W[idx, j] = w12
            W[j, idx] = w12
        if np.abs(W - W_old).mean() < tol * scale:
            break
    Theta = _theta_from_w(W, B)
    out = [W, Theta]
    if return_beta:
        out.append(B)
    if return_n_iter:
        out.append(it)
    return tuple(out)


def _lasso_cd_weighted(V, s, lams, beta, tol=1e-10, max_iter=5000):
    """Lasso modifié avec une pénalité propre à chaque coordonnée."""
    beta = np.array(beta, dtype=float)
    q = s.size
    for _ in range(max_iter):
        delta = 0.0
        for j in range(q):
            r = s[j] - V[j] @ beta + V[j, j] * beta[j]
            new = soft_threshold(r, lams[j]) / V[j, j]
            delta = max(delta, abs(new - beta[j]))
            beta[j] = new
        if delta < tol:
            break
    return beta


def graphical_lasso_path(S, lams, **kwargs):
    """Chemin du graphical lasso pour une suite de pénalités ``lams``.

    Les pénalités sont parcourues dans l'ordre décroissant avec démarrage à
    chaud. Retourne une liste de matrices ``Theta`` dans l'ordre de ``lams``.
    """
    lams = np.asarray(lams, dtype=float)
    order = np.argsort(-lams)
    out = [None] * len(lams)
    W = B = None
    for i in order:
        W, Theta, B = graphical_lasso(S, lams[i], W_init=W, B_init=B,
                                      return_beta=True, **kwargs)
        out[i] = Theta
    return out


def meinshausen_buhlmann(S, lam, rule="or"):
    """Sélection de graphe de Meinshausen–Bühlmann (section 17.3.2).

    Pour chaque variable j, lasso de ``X_j`` sur les autres (forme
    covariance : ``V = S_{-j,-j}``, ``s = S_{-j,j}``). L'arête (j, k) est
    retenue si le coefficient de k dans la régression de j **ou** (``"or"``)
    **et** (``"and"``) celui de j dans la régression de k est non nul.

    Returns
    -------
    A : ndarray (p, p) booléen.
    """
    S = np.asarray(S, dtype=float)
    p = S.shape[0]
    NZ = np.zeros((p, p), dtype=bool)
    for j in range(p):
        idx = np.arange(p) != j
        b = lasso_cd(S[np.ix_(idx, idx)], S[idx, j], lam)
        NZ[idx, j] = np.abs(b) > 1e-12
    return (NZ | NZ.T) if rule == "or" else (NZ & NZ.T)


def make_sparse_precision(A, rng, low=0.3, high=0.6, min_eig=0.2):
    """Matrice de précision aléatoire de support ``A`` (pour simulations).

    Les coefficients hors diagonale des arêtes sont tirés uniformément dans
    ``[low, high]`` avec un signe aléatoire ; la diagonale est ensuite
    choisie pour que la plus petite valeur propre vaille ``min_eig``, puis la
    matrice est remise à l'échelle pour que ``Sigma`` ait une diagonale
    unité.

    Returns
    -------
    Theta : ndarray (p, p) définie positive.
    """
    A = np.asarray(A, dtype=bool)
    p = A.shape[0]
    T = np.zeros((p, p))
    for j, k in edges_from_adjacency(A):
        T[j, k] = T[k, j] = rng.uniform(low, high) * rng.choice([-1.0, 1.0])
    lmin = np.linalg.eigvalsh(T).min()
    T[np.diag_indices(p)] = min_eig - lmin
    Sigma = np.linalg.inv(T)
    d = np.sqrt(np.diag(Sigma))
    return T * np.outer(d, d)  # Theta pour la covariance D^{-1} Sigma D^{-1}


def em_gaussian_graph(X, A, n_iter=100, tol=1e-8):
    """Algorithme EM pour un modèle graphique gaussien à graphe connu, avec
    valeurs manquantes codées ``np.nan`` (exercice 17.9).

    Étape E : imputation par l'espérance conditionnelle (éq. 17.43) et
    accumulation de la covariance conditionnelle des composantes manquantes
    (terme correctif de l'éq. 17.44). Étape M : moyenne empirique, puis
    algorithme 17.1 appliqué à la covariance « complétée ».

    Returns
    -------
    mu : ndarray (p,)
    W : ndarray (p, p) - covariance estimée.
    Theta : ndarray (p, p) - précision estimée.
    X_hat : ndarray (N, p) - données imputées (dernière étape E).
    """
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    miss = np.isnan(X)
    mu = np.nanmean(X, axis=0)
    X_hat = np.where(miss, mu, X)
    W = empirical_covariance(X_hat)
    W[np.diag_indices(p)] = np.nanvar(X, axis=0)
    patterns = {}
    for i in range(N):
        patterns.setdefault(miss[i].tobytes(), []).append(i)
    for _ in range(n_iter):
        C = np.zeros((p, p))
        for key, rows in patterns.items():
            m = miss[rows[0]]
            if not m.any():
                continue
            o = ~m
            rows = np.asarray(rows)
            K = np.linalg.solve(W[np.ix_(o, o)], W[np.ix_(o, m)])  # Soo^-1 Som
            X_hat[np.ix_(rows, np.flatnonzero(m))] = (
                mu[m] + (X[np.ix_(rows, np.flatnonzero(o))] - mu[o]) @ K)
            Cmm = W[np.ix_(m, m)] - W[np.ix_(m, o)] @ K
            C[np.ix_(m, m)] += len(rows) * Cmm
        mu_new = X_hat.mean(axis=0)
        S = empirical_covariance(X_hat) + C / N
        W_new, Theta = fit_known_graph(S, A)
        change = np.max(np.abs(W_new - W)) + np.max(np.abs(mu_new - mu))
        W, mu = W_new, mu_new
        if change < tol:
            break
    return mu, W, Theta, X_hat


# ---------------------------------------------------------------------------
# 3. Modèles de Ising (petit p)
# ---------------------------------------------------------------------------


def ising_states(p):
    """Les ``2^p`` états de ``{0, 1}^p`` (tableau (2^p, p) d'entiers)."""
    return np.array(list(itertools.product([0, 1], repeat=p)), dtype=float)


def ising_energy(X, theta0, Theta):
    """« Énergie » (logit non normalisé)
    ``sum_j theta0_j x_j + sum_{j<k} Theta_jk x_j x_k`` pour chaque ligne de X."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Theta = np.asarray(Theta, dtype=float)
    Tu = np.triu(Theta, 1)
    return X @ np.asarray(theta0, dtype=float) + np.einsum("ij,jk,ik->i", X, Tu, X)


def ising_log_partition(theta0, Theta):
    """Logarithme de la fonction de partition ``Phi(Theta)`` (éq. 17.29),
    calculé par énumération des ``2^p`` états (log-sum-exp stable)."""
    p = len(theta0)
    e = ising_energy(ising_states(p), theta0, Theta)
    m = e.max()
    return m + np.log(np.exp(e - m).sum())


def ising_probs(theta0, Theta):
    """Probabilités des ``2^p`` états. Retourne ``(states, probs)``."""
    p = len(theta0)
    X = ising_states(p)
    e = ising_energy(X, theta0, Theta)
    w = np.exp(e - e.max())
    return X, w / w.sum()


def ising_moments(theta0, Theta):
    """Moments exacts par énumération : ``E[X_j]`` et ``E[X_j X_k]``.

    Returns
    -------
    m1 : ndarray (p,)
    M2 : ndarray (p, p) avec ``M2[j, k] = E[X_j X_k]`` (``M2[j, j] = E[X_j]``).
    """
    X, pr = ising_probs(theta0, Theta)
    m1 = pr @ X
    M2 = (X * pr[:, None]).T @ X
    return m1, M2


def ising_sample(theta0, Theta, N, rng):
    """Échantillon exact de taille N (tirage multinomial sur les 2^p états)."""
    X, pr = ising_probs(theta0, Theta)
    idx = rng.choice(len(pr), size=N, p=pr)
    return X[idx]


def ising_conditional(X, j, theta0, Theta):
    """``Pr(X_j = 1 | X_{-j})`` (forme logistique, éq. 17.30) pour chaque ligne."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Theta = np.asarray(Theta, dtype=float)
    eta = theta0[j] + X @ Theta[:, j] - X[:, j] * Theta[j, j]
    return 1.0 / (1.0 + np.exp(-eta))


def ising_gibbs(theta0, Theta, n_sweeps, rng, x0=None, burn=0):
    """Échantillonneur de Gibbs : tire successivement chaque ``X_j`` selon
    sa loi conditionnelle (17.30). Retourne les états après chaque balayage
    (après ``burn`` balayages de chauffe)."""
    p = len(theta0)
    x = rng.randint(0, 2, size=p).astype(float) if x0 is None else np.array(x0, float)
    out = []
    for t in range(burn + n_sweeps):
        for j in range(p):
            pj = 1.0 / (1.0 + np.exp(-(theta0[j] + Theta[j] @ x - Theta[j, j] * x[j])))
            x[j] = float(rng.rand() < pj)
        if t >= burn:
            out.append(x.copy())
    return np.array(out)


def _ising_design(p, A):
    """Liste des arêtes libres et matrice des statistiques suffisantes."""
    edges = edges_from_adjacency(A)
    X = ising_states(p)
    T = np.c_[X, np.array([X[:, j] * X[:, k] for j, k in edges]).T.reshape(len(X), -1)]
    return edges, X, T


def _ising_unpack(par, p, edges):
    theta0 = par[:p].copy()
    Theta = np.zeros((p, p))
    for v, (j, k) in zip(par[p:], edges):
        Theta[j, k] = Theta[k, j] = v
    return theta0, Theta


def ising_loglik(X, theta0, Theta):
    """Log-vraisemblance moyenne ``(1/N) sum_i log p(x_i)`` (éq. 17.31 / N)."""
    return ising_energy(X, theta0, Theta).mean() - ising_log_partition(theta0, Theta)


def ising_fit_mle(X, A, max_iter=100, tol=1e-10, return_n_iter=False,
                  sample_weight=None):
    """Maximum de vraisemblance exact d'un modèle de Ising à graphe ``A``
    par méthode de Newton, en énumérant les 2^p états.

    Le gradient de la log-vraisemblance moyenne est ``E_hat[t] - E_theta[t]``
    (éq. 17.34) où ``t(x) = (x_j, x_j x_k)`` ; la hessienne est
    ``-Cov_theta(t)``.

    ``sample_weight`` (poids positifs, normalisés en interne) permet de
    remplacer la loi empirique par une loi quelconque sur les observations,
    par exemple une loi exacte sur les ``2^p`` états.

    Returns
    -------
    theta0 : ndarray (p,)
    Theta : ndarray (p, p) symétrique, nulle hors des arêtes de ``A``.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    edges, states, T = _ising_design(p, A)
    w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
    w = w / w.sum()
    tbar = w @ np.c_[X, np.array([X[:, j] * X[:, k] for j, k in edges]).T.reshape(len(X), -1)]
    def loglik(par):
        e = T @ par
        m = e.max()
        return par @ tbar - (m + np.log(np.exp(e - m).sum()))

    par = np.zeros(T.shape[1])
    ll = loglik(par)
    for it in range(1, max_iter + 1):
        e = T @ par
        w = np.exp(e - e.max())
        pr = w / w.sum()
        Et = pr @ T
        grad = tbar - Et
        if np.max(np.abs(grad)) < tol:
            break
        Tc = T - Et
        H = (Tc * pr[:, None]).T @ Tc + 1e-12 * np.eye(len(par))
        step = np.linalg.solve(H, grad)
        # recherche linéaire (Newton amorti) : on divise le pas par 2 tant
        # que la log-vraisemblance n'augmente pas suffisamment
        t = 1.0
        while t > 1e-10:
            new = par + t * step
            ll_new = loglik(new)
            if ll_new >= ll + 0.25 * t * (grad @ step):
                break
            t /= 2
        par, ll = new, ll_new
    theta0, Theta = _ising_unpack(par, p, edges)
    if return_n_iter:
        return theta0, Theta, it
    return theta0, Theta


def ising_pseudo_loglik(X, theta0, Theta):
    """Pseudo-log-vraisemblance moyenne (Besag) :
    ``(1/N) sum_i sum_j log Pr(X_j = x_ij | x_{i,-j})``."""
    X = np.asarray(X, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    Eta = theta0 + X @ Theta - X * np.diag(Theta)
    return np.mean(np.sum(X * Eta - np.logaddexp(0.0, Eta), axis=1))


def ising_fit_pseudolikelihood(X, A):
    """Maximum de pseudo-vraisemblance (paramètres symétriques) par L-BFGS.

    Ne nécessite pas la fonction de partition : coût ``O(N p^2)`` par
    évaluation, utilisable pour ``p`` grand.
    """
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=float)
    N, p = X.shape
    edges = edges_from_adjacency(A)

    def f(par):
        theta0, Theta = _ising_unpack(par, p, edges)
        Eta = theta0 + X @ Theta
        P = 1.0 / (1.0 + np.exp(-Eta))
        val = -np.mean(np.sum(X * Eta - np.logaddexp(0.0, Eta), axis=1))
        R = X - P  # résidus
        g0 = -R.mean(0)
        G = R.T @ X / N  # G[j, k] = mean (x_j - p_j) x_k
        ge = np.array([-(G[j, k] + G[k, j]) for j, k in edges])
        return val, np.r_[g0, ge]

    res = minimize(f, np.zeros(p + len(edges)), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-10})
    return _ising_unpack(res.x, p, edges)


def logistic_l1(Z, y, lam, max_iter=5000, tol=1e-9):
    """Régression logistique avec pénalité L1 (ordonnée à l'origine non
    pénalisée) par gradient proximal (ISTA).

    Minimise ``-(1/N) sum_i [y_i eta_i - log(1 + e^{eta_i})] + lam ||b||_1``
    avec ``eta_i = b0 + z_i^T b``.

    Returns
    -------
    b0 : float, b : ndarray
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    N, q = Z.shape
    Z1 = np.c_[np.ones(N), Z]
    L = 0.25 * np.linalg.eigvalsh(Z1.T @ Z1 / N).max()
    step = 1.0 / L
    par = np.zeros(q + 1)
    for _ in range(max_iter):
        eta = Z1 @ par
        g = Z1.T @ (1.0 / (1.0 + np.exp(-eta)) - y) / N
        new = par - step * g
        new[1:] = soft_threshold(new[1:], step * lam)
        if np.max(np.abs(new - par)) < tol:
            par = new
            break
        par = new
    return par[0], par[1:]


def nodewise_logistic(X, lam=0.0, rule="min"):
    """Approche de Wainwright et al. (2007) : régression logistique L1 de
    chaque nœud sur les autres, puis symétrisation.

    Parameters
    ----------
    X : ndarray (N, p) binaire.
    lam : float - pénalité L1.
    rule : ``"min"``, ``"max"`` ou ``"mean"`` - règle de symétrisation
        (``"min"`` garde le coefficient de plus petite valeur absolue).

    Returns
    -------
    theta0 : ndarray (p,) - ordonnées à l'origine.
    Theta : ndarray (p, p) symétrique.
    Theta_tilde : ndarray (p, p) non symétrique (ligne j = régression de j).
    """
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    Tt = np.zeros((p, p))
    theta0 = np.zeros(p)
    for j in range(p):
        idx = np.arange(p) != j
        b0, b = logistic_l1(X[:, idx], X[:, j], lam)
        theta0[j] = b0
        Tt[j, idx] = b
    if rule == "mean":
        Theta = (Tt + Tt.T) / 2
    else:
        pick = np.abs(Tt) <= np.abs(Tt.T) if rule == "min" else np.abs(Tt) >= np.abs(Tt.T)
        Theta = np.where(pick, Tt, Tt.T)
    return theta0, Theta, Tt


# ---------------------------------------------------------------------------
# 4. Données simulées « façon cytométrie » (graphe de la figure 17.1)
# ---------------------------------------------------------------------------

#: Noms des 11 protéines (ordre des colonnes du jeu de Sachs et al.).
CYTO_NAMES = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA",
              "PKC", "P38", "Jnk"]

#: Arêtes lues sur la figure 17.1 du livre (graphical lasso, lambda = 14).
CYTO_EDGES = [("Raf", "Mek"), ("Mek", "P38"), ("Mek", "PIP2"), ("Mek", "Akt"),
              ("Mek", "PKA"), ("Jnk", "P38"), ("Jnk", "PIP2"), ("Plcg", "P38"),
              ("Plcg", "PIP2"), ("Plcg", "PKA"), ("PIP2", "P38"),
              ("PIP2", "PKA"), ("PIP2", "Akt"), ("P38", "PKC"), ("P38", "PKA"),
              ("P38", "Akt")]

#: Ordre (sens horaire, à partir du haut) des sommets sur la figure 17.1.
_CYTO_CIRCLE = ["Raf", "Jnk", "P38", "PKC", "PKA", "Akt", "Erk", "PIP3",
                "PIP2", "Plcg", "Mek"]


def cytometry_like_graph():
    """Graphe de la figure 17.1 : retourne ``(names, A, pos)``.

    ``pos`` reproduit la disposition circulaire du livre.
    """
    idx = {n: i for i, n in enumerate(CYTO_NAMES)}
    A = adjacency_from_edges(len(CYTO_NAMES), [(idx[a], idx[b]) for a, b in CYTO_EDGES])
    circ = circle_layout(len(CYTO_NAMES))
    pos = np.array([circ[_CYTO_CIRCLE.index(n)] for n in CYTO_NAMES])
    return list(CYTO_NAMES), A, pos


def simulate_cytometry_like(N=7466, seed=0):
    """Données gaussiennes simulées dont le graphe vrai est celui de la
    figure 17.1 (les vraies données de cytométrie ne sont pas disponibles).

    Returns
    -------
    X : ndarray (N, 11) - observations (variables de variance 1).
    Theta : ndarray (11, 11) - vraie matrice de précision.
    A : ndarray (11, 11) - vrai graphe.
    """
    rng = np.random.RandomState(seed)
    names, A, _ = cytometry_like_graph()
    Theta = make_sparse_precision(A, rng, low=0.25, high=0.5, min_eig=0.25)
    Sigma = np.linalg.inv(Theta)
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(N, len(names)) @ L.T
    return X, Theta, A


# ---------------------------------------------------------------------------
# 5. Machine de Boltzmann restreinte (RBM)
# ---------------------------------------------------------------------------


def sigmoid(z):
    """Fonction logistique ``1 / (1 + e^{-z})``."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def rbm_free_energy(V, W, b, c):
    """Énergie libre ``F(v) = -b^T v - sum_h log(1 + exp(c_h + v^T W_h))``.

    On a ``Pr(v) = exp(-F(v)) / Z`` : la somme sur les unités cachées
    (binaires, indépendantes sachant v) se calcule en forme fermée.
    """
    V = np.atleast_2d(V)
    return -V @ b - np.logaddexp(0.0, V @ W + c).sum(axis=1)


def rbm_fit_cd(V, n_hidden, rng, n_epochs=20, lr=0.05, batch_size=50,
               momentum=0.5, weight_decay=1e-4, softmax_slice=None):
    """Apprentissage d'une RBM binaire par divergence contrastive CD-1.

    Le gradient (17.37) ``E_data[v h^T] - E_model[v h^T]`` est approché en
    remplaçant le second terme par une seule étape de Gibbs partant des
    données (Hinton, 2002).

    Parameters
    ----------
    V : ndarray (N, n_visible) à valeurs dans {0, 1}.
    n_hidden : int
    rng : np.random.RandomState
    softmax_slice : slice, optionnel
        Bloc d'unités visibles formant une unité multinomiale (une seule
        vaut 1, par ex. l'étiquette de classe) : leur reconstruction se fait
        par softmax.

    Returns
    -------
    W : ndarray (n_visible, n_hidden), b : ndarray (n_visible,),
    c : ndarray (n_hidden,)
    """
    try:  # petits produits matriciels : le multi-threading BLAS ralentit
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return _rbm_fit_cd(V, n_hidden, rng, n_epochs, lr, batch_size,
                               momentum, weight_decay, softmax_slice)
    except ImportError:
        return _rbm_fit_cd(V, n_hidden, rng, n_epochs, lr, batch_size,
                           momentum, weight_decay, softmax_slice)


def _rbm_fit_cd(V, n_hidden, rng, n_epochs, lr, batch_size, momentum,
                weight_decay, softmax_slice):
    V = np.asarray(V, dtype=float)
    N, nv = V.shape
    W = 0.01 * rng.randn(nv, n_hidden)
    b = np.log((V.mean(0) + 1e-3) / (1 - V.mean(0) + 1e-3))
    c = np.zeros(n_hidden)
    dW, db, dc = np.zeros_like(W), np.zeros_like(b), np.zeros_like(c)
    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            v0 = V[perm[start:start + batch_size]]
            ph0 = sigmoid(v0 @ W + c)
            h0 = (rng.rand(*ph0.shape) < ph0).astype(float)
            a = h0 @ W.T + b
            v1 = sigmoid(a)
            if softmax_slice is not None:
                z = a[:, softmax_slice]
                z = np.exp(z - z.max(axis=1, keepdims=True))
                v1[:, softmax_slice] = z / z.sum(axis=1, keepdims=True)
            ph1 = sigmoid(v1 @ W + c)
            n = len(v0)
            gW = (v0.T @ ph0 - v1.T @ ph1) / n - weight_decay * W
            dW = momentum * dW + lr * gW
            db = momentum * db + lr * (v0 - v1).mean(0)
            dc = momentum * dc + lr * (ph0 - ph1).mean(0)
            W += dW
            b += db
            c += dc
    return W, b, c
