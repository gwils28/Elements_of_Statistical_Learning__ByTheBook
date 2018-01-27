"""Méthodes à prototypes et plus proches voisins - chapitre 13 d'ESL.

Ce module regroupe des implémentations explicites (numpy) de :

* K-means (algorithme de Lloyd) et son utilisation comme classifieur
  (R prototypes par classe) ;
* LVQ1 (Learning Vector Quantization, algorithme 13.1 du livre) ;
* un classifieur par mélanges gaussiens à covariance commune, ajusté par EM
  (section 13.2.3) ;
* les k plus proches voisins (avec plusieurs valeurs de k d'un coup et
  validation croisée) ;
* la métrique adaptative DANN (éq. (13.8)-(13.9)) ;
* les vecteurs tangents d'images (rotation, translations, homothéties,
  cisaillement, épaisseur) et la distance tangente (section 13.3.3) ;
* quelques outils : rayon médian du 1-voisinage (éq. (13.7)), simulations
  du chapitre, condensation de Hart, multi-édition.
"""

import numpy as np
from scipy.special import gammaln, logsumexp


# ---------------------------------------------------------------------------
# Outils de base
# ---------------------------------------------------------------------------

def sq_dists(A, B):
    """Distances euclidiennes au carré entre les lignes de ``A`` et de ``B``.

    Parameters
    ----------
    A : ndarray (n, p)
    B : ndarray (m, p)

    Returns
    -------
    D : ndarray (n, m), ``D[i, j] = ||A_i - B_j||^2`` (tronqué à 0).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    D = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(D, 0.0)


def kfold_indices(n, n_folds, rng=None):
    """Découpe aléatoire de ``range(n)`` en ``n_folds`` blocs (liste d'indices)."""
    rng = np.random.RandomState(0) if rng is None else rng
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, n_folds)]


# ---------------------------------------------------------------------------
# K-means et classifieur K-means
# ---------------------------------------------------------------------------

def kmeans(X, R, n_init=5, max_iter=100, rng=None, init=None):
    """Algorithme de Lloyd : minimise l'inertie intra-groupes.

    Critère : :math:`\\sum_i \\min_r \\|x_i - m_r\\|^2`.

    Parameters
    ----------
    X : ndarray (N, p)
    R : int
        Nombre de centres.
    n_init : int
        Nombre de redémarrages aléatoires (on garde le meilleur).
    max_iter : int
    rng : RandomState
    init : ndarray (R, p), optionnel
        Centres initiaux (dans ce cas un seul démarrage).

    Returns
    -------
    centers : ndarray (R, p)
    labels : ndarray (N,)
    inertia : float
    history : list
        Valeurs successives du critère pour le meilleur démarrage.
    """
    X = np.asarray(X, float)
    rng = np.random.RandomState(0) if rng is None else rng
    N = X.shape[0]
    R = min(R, N)
    best = None
    starts = [np.array(init, float)] if init is not None else [None] * n_init
    for start in starts:
        C = X[rng.choice(N, R, replace=False)].copy() if start is None else start.copy()
        hist = []
        labels = None
        for _ in range(max_iter):
            D = sq_dists(X, C)
            new_labels = D.argmin(1)
            hist.append(D[np.arange(N), new_labels].sum())
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for r in range(R):
                mask = labels == r
                if mask.any():
                    C[r] = X[mask].mean(0)
                else:  # groupe vide : on le relance sur un point au hasard
                    C[r] = X[rng.randint(N)]
        inertia = hist[-1]
        if best is None or inertia < best[2]:
            best = (C, labels, inertia, hist)
    return best


class KMeansClassifier:
    """Classifieur à prototypes obtenus par K-means dans chaque classe.

    Parameters
    ----------
    R : int
        Nombre de prototypes par classe.
    n_init : int
    rng : RandomState
    """

    def __init__(self, R=5, n_init=5, rng=None):
        self.R = R
        self.n_init = n_init
        self.rng = np.random.RandomState(0) if rng is None else rng

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        protos, labs = [], []
        for k in self.classes_:
            C, _, _, _ = kmeans(X[y == k], self.R, n_init=self.n_init, rng=self.rng)
            protos.append(C)
            labs.append(np.full(len(C), k))
        self.prototypes_ = np.vstack(protos)
        self.proto_labels_ = np.concatenate(labs)
        return self

    def predict(self, X):
        return nearest_prototype(X, self.prototypes_, self.proto_labels_)


def nearest_prototype(X, prototypes, proto_labels):
    """Classe du prototype le plus proche (distance euclidienne)."""
    D = sq_dists(X, prototypes)
    return np.asarray(proto_labels)[D.argmin(1)]


# ---------------------------------------------------------------------------
# LVQ1
# ---------------------------------------------------------------------------

def lvq1(X, y, prototypes, proto_labels, n_iter=None, eps0=0.05, rng=None):
    """Algorithme LVQ1 (algorithme 13.1 du livre).

    À chaque itération on tire un point d'apprentissage ``x_i`` au hasard ;
    le prototype le plus proche ``m`` est rapproché de ``x_i``
    (``m <- m + eps (x_i - m)``) s'il est de la même classe, éloigné
    (``m <- m - eps (x_i - m)``) sinon. Le pas décroît linéairement de
    ``eps0`` à 0.

    Parameters
    ----------
    X, y : données d'apprentissage.
    prototypes : ndarray (M, p)
        Prototypes initiaux (non modifiés : on travaille sur une copie).
    proto_labels : ndarray (M,)
    n_iter : int, par défaut ``20 * N``.
    eps0 : float
        Pas initial.
    rng : RandomState

    Returns
    -------
    M : ndarray (M, p), prototypes finaux.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    rng = np.random.RandomState(0) if rng is None else rng
    M = np.array(prototypes, float, copy=True)
    proto_labels = np.asarray(proto_labels)
    N = X.shape[0]
    n_iter = 20 * N if n_iter is None else n_iter
    idx = rng.randint(N, size=n_iter)
    for t, i in enumerate(idx):
        eps = eps0 * (1.0 - t / n_iter)
        d = ((M - X[i]) ** 2).sum(1)
        j = d.argmin()
        if proto_labels[j] == y[i]:
            M[j] += eps * (X[i] - M[j])
        else:
            M[j] -= eps * (X[i] - M[j])
    return M


class LVQClassifier:
    """Classifieur LVQ1 initialisé par la solution K-means (R par classe)."""

    def __init__(self, R=5, n_iter=None, eps0=0.05, n_init=5, rng=None):
        self.R = R
        self.n_iter = n_iter
        self.eps0 = eps0
        self.n_init = n_init
        self.rng = np.random.RandomState(0) if rng is None else rng

    def fit(self, X, y):
        km = KMeansClassifier(self.R, self.n_init, self.rng).fit(X, y)
        self.classes_ = km.classes_
        self.init_prototypes_ = km.prototypes_
        self.proto_labels_ = km.proto_labels_
        self.prototypes_ = lvq1(X, y, km.prototypes_, km.proto_labels_,
                                self.n_iter, self.eps0, self.rng)
        return self

    def predict(self, X):
        return nearest_prototype(X, self.prototypes_, self.proto_labels_)


# ---------------------------------------------------------------------------
# Mélanges gaussiens (covariance commune) - EM
# ---------------------------------------------------------------------------

def _log_gauss_common(X, means, cov):
    """log phi(x ; mu_r, cov) pour tous les points et toutes les moyennes."""
    p = X.shape[1]
    L = np.linalg.cholesky(cov)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    out = np.empty((X.shape[0], len(means)))
    for r, m in enumerate(means):
        z = np.linalg.solve(L, (X - m).T)
        out[:, r] = -0.5 * (z ** 2).sum(0)
    return out - 0.5 * (p * np.log(2 * np.pi) + logdet)


class GaussianMixtureClassifier:
    """Mélange de R gaussiennes par classe, covariance commune à toutes.

    Chaque classe ``k`` a pour densité
    :math:`f_k(x) = \\sum_r w_{kr} \\phi(x; \\mu_{kr}, \\Sigma)` et l'on classe
    par :math:`\\arg\\max_k \\pi_k f_k(x)`. Les paramètres sont estimés par EM,
    initialisé par la solution K-means de chaque classe (comme dans la
    figure 13.2 du livre).
    """

    def __init__(self, R=5, n_iter=100, tol=1e-8, rng=None):
        self.R = R
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(0) if rng is None else rng

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        N, p = X.shape
        km = KMeansClassifier(self.R, rng=self.rng).fit(X, y)
        self.classes_ = km.classes_
        means = km.prototypes_.copy()
        lab = km.proto_labels_
        # initialisation : covariance intra-groupes K-means
        D = sq_dists(X, means)
        D[:, :] = np.where(lab[None, :] == y[:, None], D, np.inf)
        a = D.argmin(1)
        res = X - means[a]
        cov = res.T @ res / N + 1e-6 * np.eye(p)
        w = np.array([np.mean(a[y == lab[r]] == r) for r in range(len(means))])
        same = lab[None, :] == y[:, None]           # composantes autorisées
        self.loglik_ = []
        for _ in range(self.n_iter):
            # Étape E : responsabilités restreintes aux composantes de la classe
            logp = _log_gauss_common(X, means, cov) + np.log(np.maximum(w, 1e-300))
            logp = np.where(same, logp, -np.inf)
            lse = logsumexp(logp, axis=1, keepdims=True)
            G = np.exp(logp - lse)
            self.loglik_.append(lse.sum())
            # Étape M
            Nr = G.sum(0)
            means = (G.T @ X) / np.maximum(Nr, 1e-12)[:, None]
            cov = np.zeros((p, p))
            for r in range(len(means)):
                Z = X - means[r]
                cov += (G[:, r, None] * Z).T @ Z
            cov = cov / N + 1e-9 * np.eye(p)
            for k in self.classes_:
                m = lab == k
                w[m] = Nr[m] / Nr[m].sum()
            if len(self.loglik_) > 1 and abs(self.loglik_[-1] - self.loglik_[-2]) < self.tol * abs(self.loglik_[-1]):
                break
        self.means_, self.cov_, self.weights_, self.labels_ = means, cov, w, lab
        self.priors_ = np.array([np.mean(y == k) for k in self.classes_])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        logp = _log_gauss_common(X, self.means_, self.cov_) + np.log(np.maximum(self.weights_, 1e-300))
        cols = []
        for j, k in enumerate(self.classes_):
            cols.append(np.log(self.priors_[j]) + logsumexp(logp[:, self.labels_ == k], axis=1))
        L = np.column_stack(cols)
        return np.exp(L - logsumexp(L, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ---------------------------------------------------------------------------
# k plus proches voisins
# ---------------------------------------------------------------------------

def knn_from_dists(D, y_train, ks, rng=None, classes=None):
    """Prédictions k-ppv pour plusieurs k à partir d'une matrice de distances.

    Parameters
    ----------
    D : ndarray (n_test, n_train)
        Distances (ou distances au carré : seul l'ordre compte).
    y_train : ndarray (n_train,)
    ks : int ou liste d'entiers
    rng : RandomState
        Sert à départager les égalités de votes au hasard (comme dans le livre).
    classes : ndarray, optionnel

    Returns
    -------
    pred : ndarray (len(ks), n_test) (ou (n_test,) si ``ks`` est un entier).
    """
    rng = np.random.RandomState(0) if rng is None else rng
    scalar = np.isscalar(ks)
    ks = np.atleast_1d(ks).astype(int)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train) if classes is None else np.asarray(classes)
    kmax = min(ks.max(), D.shape[1])
    if kmax < D.shape[1]:
        part = np.argpartition(D, kmax - 1, axis=1)[:, :kmax]
    else:
        part = np.tile(np.arange(D.shape[1]), (D.shape[0], 1))
    order = np.take_along_axis(D, part, 1).argsort(1, kind="stable")
    nbr = np.take_along_axis(part, order, 1)           # voisins triés
    lab = np.searchsorted(classes, y_train[nbr])       # (n_test, kmax)
    onehot = np.zeros(lab.shape + (len(classes),))
    np.put_along_axis(onehot, lab[..., None], 1.0, axis=2)
    counts = onehot.cumsum(1)                          # votes pour chaque k
    jitter = rng.uniform(0, 0.5, size=(D.shape[0], len(classes)))
    out = np.empty((len(ks), D.shape[0]), dtype=classes.dtype)
    for i, k in enumerate(ks):
        out[i] = classes[(counts[:, min(k, kmax) - 1, :] + jitter).argmax(1)]
    return out[0] if scalar else out


def knn_predict(X_train, y_train, X_test, k=1, rng=None):
    """Classifieur des k plus proches voisins (distance euclidienne)."""
    return knn_from_dists(sq_dists(X_test, X_train), y_train, k, rng)


def knn_cv_error(X, y, ks, n_folds=10, rng=None, groups=None):
    """Erreur de validation croisée du k-ppv pour une liste de ``ks``.

    Parameters
    ----------
    groups : ndarray (N,), optionnel
        Si fourni, les observations d'un même groupe restent dans le même
        bloc (utile quand on a ajouté des copies transformées d'un point).

    Returns
    -------
    mean, se : ndarray (len(ks),)
        Moyenne des erreurs des blocs et erreur standard
        (écart-type / sqrt(n_folds)).
    """
    rng = np.random.RandomState(0) if rng is None else rng
    ks = np.atleast_1d(ks)
    if groups is None:
        groups = np.arange(len(y))
    ug = np.unique(groups)
    folds = kfold_indices(len(ug), n_folds, rng)
    errs = np.empty((n_folds, len(ks)))
    for f, fg in enumerate(folds):
        test = np.isin(groups, ug[fg])
        pred = knn_from_dists(sq_dists(X[test], X[~test]), y[~test], ks, rng,
                              classes=np.unique(y))
        errs[f] = (pred != y[test]).mean(1)
    return errs.mean(0), errs.std(0, ddof=1) / np.sqrt(n_folds)


# ---------------------------------------------------------------------------
# DANN : métrique discriminante adaptative
# ---------------------------------------------------------------------------

def _inv_sqrt_sym(W, ridge=1e-8):
    vals, vecs = np.linalg.eigh(W)
    vals = np.maximum(vals, ridge * max(vals.max(), 1e-12))
    return (vecs / np.sqrt(vals)) @ vecs.T


def dann_metric(Xn, yn, eps=1.0):
    """Matrice Sigma de la métrique DANN (éq. (13.9) du livre).

    :math:`\\Sigma = W^{-1/2}[W^{-1/2} B W^{-1/2} + \\epsilon I] W^{-1/2}`,
    où ``W`` est la covariance intra-classe poolée et ``B`` la covariance
    inter-classes, calculées sur le voisinage ``(Xn, yn)``.
    """
    Xn = np.asarray(Xn, float)
    yn = np.asarray(yn)
    p = Xn.shape[1]
    xbar = Xn.mean(0)
    W = np.zeros((p, p))
    B = np.zeros((p, p))
    for k in np.unique(yn):
        Z = Xn[yn == k]
        pik = len(Z) / len(Xn)
        mk = Z.mean(0)
        W += pik * np.cov(Z.T, bias=True).reshape(p, p)
        B += pik * np.outer(mk - xbar, mk - xbar)
    Wm = _inv_sqrt_sym(W)
    Bstar = Wm @ B @ Wm
    return Wm @ (Bstar + eps * np.eye(p)) @ Wm


def dann_predict(X_train, y_train, X_test, k=5, n_metric=50, eps=1.0, rng=None):
    """Classifieur k-ppv avec la métrique DANN estimée en chaque point test.

    Pour chaque point test ``x0`` : (1) on prend ses ``n_metric`` voisins
    euclidiens, (2) on y calcule ``Sigma`` (éq. (13.9)), (3) on applique le
    k-ppv avec la distance ``(x - x0)^T Sigma (x - x0)`` (éq. (13.8)).
    """
    rng = np.random.RandomState(0) if rng is None else rng
    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)
    classes = np.unique(y_train)
    D0 = sq_dists(X_test, X_train)
    nbrs = np.argpartition(D0, n_metric - 1, axis=1)[:, :n_metric]
    pred = np.empty(len(X_test), dtype=classes.dtype)
    for i, x0 in enumerate(X_test):
        S = dann_metric(X_train[nbrs[i]], y_train[nbrs[i]], eps)
        Z = X_train - x0
        d = np.einsum("ij,jk,ik->i", Z, S, Z)
        pred[i] = knn_from_dists(d[None, :], y_train, k, rng, classes)[0]
    return pred


# ---------------------------------------------------------------------------
# Images : vecteurs tangents et distance tangente
# ---------------------------------------------------------------------------

def pixel_coords(side=16):
    """Coordonnées (u, v) centrées des pixels d'une image ``side x side``.

    ``u`` croît vers la droite, ``v`` vers le haut ; l'ordre est celui de la
    lecture ligne par ligne (comme les 256 colonnes de ``load_zip``).
    """
    c = (side - 1) / 2.0
    i, j = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.column_stack([(j - c).ravel(), (c - i).ravel()])


def smoother_matrices(side=16, h=0.9, points=None):
    """Matrices du lissage à noyau gaussien (Nadaraya-Watson) d'une image.

    L'image discrète (valeurs ``F_j`` aux pixels ``u_j``) est prolongée en une
    fonction lisse :math:`\\hat F(u) = \\sum_j K_h(u - u_j) F_j / \\sum_j K_h(u - u_j)`.
    ``hat F`` et ses dérivées partielles sont **linéaires** en l'image : on
    renvoie les matrices ``S, Su, Sv`` telles que ``hat F(points) = S @ F``,
    ``d hat F / du = Su @ F``, ``d hat F / dv = Sv @ F``.

    Parameters
    ----------
    side : int
    h : float
        Largeur de bande du noyau (en pixels).
    points : ndarray (n, 2), optionnel
        Points d'évaluation (par défaut les centres des pixels).
    """
    U = pixel_coords(side)
    P = U if points is None else np.asarray(points, float)
    du = P[:, None, 0] - U[None, :, 0]
    dv = P[:, None, 1] - U[None, :, 1]
    K = np.exp(-(du ** 2 + dv ** 2) / (2 * h * h))
    s = K.sum(1, keepdims=True)
    Ku = -du / (h * h) * K           # dérivée de K par rapport à u (point d'évaluation)
    Kv = -dv / (h * h) * K
    S = K / s
    Su = Ku / s - K * Ku.sum(1, keepdims=True) / s ** 2
    Sv = Kv / s - K * Kv.sum(1, keepdims=True) / s ** 2
    return S, Su, Sv


TANGENT_NAMES = ("translation x", "translation y", "rotation",
                 "homothétie x", "homothétie y", "cisaillement", "épaisseur")


def image_tangents(X, which=TANGENT_NAMES, side=16, h=0.9):
    """Vecteurs tangents des images ``X`` (une image par ligne).

    Pour une transformation ``F(u) -> F(c + A u)`` proche de l'identité, la
    dérivée par rapport à chaque paramètre s'écrit avec les deux dérivées
    spatiales ``F_u, F_v`` (exercice 13.4) :

    * translations : ``F_u``, ``F_v`` ;
    * rotation : ``-v F_u + u F_v`` ;
    * homothéties : ``u F_u`` et ``v F_v`` ;
    * cisaillement : ``v F_u`` ;
    * épaisseur du trait : ``F_u^2 + F_v^2`` (Simard et al.).

    Returns
    -------
    T : ndarray (n, len(which), side*side)
    """
    X = np.atleast_2d(np.asarray(X, float))
    _, Su, Sv = smoother_matrices(side, h)
    Fu = X @ Su.T
    Fv = X @ Sv.T
    U = pixel_coords(side)
    u, v = U[:, 0], U[:, 1]
    table = {
        "translation x": Fu,
        "translation y": Fv,
        "rotation": -v * Fu + u * Fv,
        "homothétie x": u * Fu,
        "homothétie y": v * Fv,
        "cisaillement": v * Fu,
        "épaisseur": Fu ** 2 + Fv ** 2,
    }
    return np.stack([table[w] for w in which], axis=1)


def transform_image(x, A=np.eye(2), c=(0.0, 0.0), side=16, h=0.6):
    """Image transformée ``u -> F(c + A u)`` (évaluée par lissage à noyau).

    Le centre de l'image sert d'origine ``x0``.
    """
    U = pixel_coords(side)
    P = U @ np.asarray(A, float).T + np.asarray(c, float)
    S, _, _ = smoother_matrices(side, h, P)
    return S @ np.asarray(x, float)
