"""Chapitre 11 - Réseaux de neurones et projection pursuit regression.

Ce module rassemble les outils utilisés dans les notebooks du chapitre 11 :

* un réseau de neurones à **une couche cachée** (modèle (11.5) du livre),
  écrit entièrement en numpy, avec la **rétropropagation** du gradient
  (équations (11.12)–(11.15)), le *weight decay* (11.16) et plusieurs
  algorithmes d'entraînement (descente de gradient par lots, descente de
  gradient stochastique par mini-lots, L-BFGS) ;
* une **vérification numérique du gradient** par différences finies
  centrées ;
* un **lisseur spline cubique** (smoothing spline) à nombre de degrés de
  liberté fixé, et la **projection pursuit regression** (PPR, section 11.2)
  ajustée par l'algorithme alterné « lissage / Gauss–Newton ».

Notations (celles du livre)
---------------------------
* ``X`` : matrice (N, p) des entrées ; ``Y`` : matrice (N, K) des sorties
  (codage 0/1 « one-hot » en classification) ;
* ``alpha`` : matrice (M, p+1) ; la ligne m contient
  :math:`(\\alpha_{0m}, \\alpha_m^T)` ;
* ``beta`` : matrice (K, M+1) ; la ligne k contient
  :math:`(\\beta_{0k}, \\beta_k^T)` ;
* :math:`Z_m = \\sigma(\\alpha_{0m} + \\alpha_m^T X)`,
  :math:`T_k = \\beta_{0k} + \\beta_k^T Z`, :math:`f_k(X) = g_k(T)`.

Quand ``M = 0`` il n'y a pas de couche cachée : ``alpha`` vaut ``None`` et
``beta`` est de taille (K, p+1) (modèle linéaire / logistique multinomial).
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline

try:  # dépendance de scikit-learn
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover
    _threadpool_limits = None


def limit_blas_threads(n=1):
    """Limite le nombre de fils d'exécution de BLAS/LAPACK.

    Les calculs de ce chapitre portent sur de **petites** matrices (quelques
    centaines de lignes) répétées des milliers de fois : la parallélisation
    de BLAS y coûte plus cher qu'elle ne rapporte (facteur 10 à 100 observé
    sur une machine à nombreux cœurs). Cette fonction est appelée à
    l'import du module.
    """
    if _threadpool_limits is not None:
        return _threadpool_limits(n)
    return None


limit_blas_threads(1)

# ---------------------------------------------------------------------------
# Fonctions d'activation
# ---------------------------------------------------------------------------


def sigmoid(v):
    """Sigmoïde :math:`\\sigma(v) = 1/(1+e^{-v})`.

    On utilise l'identité :math:`\\sigma(v) = (1 + \\tanh(v/2))/2`, qui est
    numériquement stable pour les grandes valeurs de :math:`|v|`.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(v, dtype=float)))


def softmax(T):
    """Softmax ligne par ligne, éq. (11.6) : :math:`e^{T_k}/\\sum_l e^{T_l}`.

    On retranche le maximum de chaque ligne avant l'exponentielle, ce qui ne
    change pas le résultat (invariance par translation) mais évite les
    dépassements numériques.
    """
    T = np.asarray(T, dtype=float)
    E = np.exp(T - T.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def one_hot(y, K=None):
    """Codage 0/1 d'un vecteur de classes ``y`` à valeurs dans {0, ..., K-1}."""
    y = np.asarray(y, dtype=int)
    if K is None:
        K = y.max() + 1
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Paramètres : initialisation, empaquetage
# ---------------------------------------------------------------------------


def init_params(p, M, K, rng, scale=0.7):
    """Poids initiaux uniformes sur [-scale, scale] (section 11.5.3).

    Parameters
    ----------
    p, M, K : int
        Nombre d'entrées, d'unités cachées (éventuellement 0) et de sorties.
    rng : numpy.random.RandomState
        Générateur aléatoire.
    scale : float
        Demi-largeur de l'intervalle des poids initiaux.

    Returns
    -------
    dict
        ``{"alpha": (M, p+1) ou None, "beta": (K, M+1)}``.
    """
    if M == 0:
        return {"alpha": None, "beta": rng.uniform(-scale, scale, (K, p + 1))}
    return {"alpha": rng.uniform(-scale, scale, (M, p + 1)),
            "beta": rng.uniform(-scale, scale, (K, M + 1))}


def pack(params):
    """Concatène ``alpha`` puis ``beta`` en un seul vecteur :math:`\\theta`."""
    parts = [params["beta"].ravel()]
    if params["alpha"] is not None:
        parts.insert(0, params["alpha"].ravel())
    return np.concatenate(parts)


def unpack(theta, p, M, K):
    """Opération inverse de :func:`pack`."""
    theta = np.asarray(theta, dtype=float)
    if M == 0:
        return {"alpha": None, "beta": theta.reshape(K, p + 1)}
    na = M * (p + 1)
    return {"alpha": theta[:na].reshape(M, p + 1),
            "beta": theta[na:].reshape(K, M + 1)}


def n_weights(p, M, K):
    """Nombre total de poids, éq. (11.8) : M(p+1) + K(M+1)."""
    if M == 0:
        return K * (p + 1)
    return M * (p + 1) + K * (M + 1)


# ---------------------------------------------------------------------------
# Propagation avant et rétropropagation
# ---------------------------------------------------------------------------


def output_function(T, output):
    """Fonction de sortie :math:`g` : ``identity``, ``sigmoid`` ou ``softmax``."""
    if output == "identity":
        return T
    if output == "sigmoid":
        return sigmoid(T)
    if output == "softmax":
        return softmax(T)
    raise ValueError("sortie inconnue : %s" % output)


def forward(params, X, output="identity"):
    """Passe avant : calcule toutes les quantités intermédiaires.

    Returns
    -------
    dict
        ``Z`` (N, M) unités cachées (``X`` si M = 0), ``T`` (N, K)
        combinaisons linéaires de sortie et ``F`` (N, K) sorties
        :math:`f_k(x_i)`.
    """
    X = np.asarray(X, dtype=float)
    if params["alpha"] is None:
        Z = X
    else:
        alpha = params["alpha"]
        Z = sigmoid(X @ alpha[:, 1:].T + alpha[:, 0])  # alpha_0m + alpha_m^T x_i
    beta = params["beta"]
    T = Z @ beta[:, 1:].T + beta[:, 0]                 # beta_0k + beta_k^T z_i
    return {"Z": Z, "T": T, "F": output_function(T, output)}


def data_loss(F, Y, loss="squared"):
    """Erreur d'ajustement R : somme des carrés (11.9) ou entropie croisée (11.10)."""
    if loss == "squared":
        return float(np.sum((Y - F) ** 2))
    if loss == "cross_entropy":
        return float(-np.sum(Y * np.log(np.clip(F, 1e-300, None))))
    raise ValueError("perte inconnue : %s" % loss)


def penalty(params):
    """Pénalité de weight decay (11.16), **hors termes constants** (biais).

    :math:`J(\\theta) = \\sum_{k,m\\ge 1} \\beta_{km}^2 + \\sum_{m,\\ell\\ge 1} \\alpha_{m\\ell}^2`.
    """
    J = np.sum(params["beta"][:, 1:] ** 2)
    if params["alpha"] is not None:
        J += np.sum(params["alpha"][:, 1:] ** 2)
    return float(J)


def backprop(params, X, Y, output="identity", loss="squared", weight_decay=0.0):
    """Erreur pénalisée et son gradient par rétropropagation.

    Calcule :math:`R(\\theta) + \\lambda J(\\theta)` et son gradient.
    La passe arrière suit les éq. (11.14)–(11.15) du livre :

    * erreurs de sortie :math:`\\delta_{ki} = \\partial R_i / \\partial T_{ki}` ;
    * erreurs cachées :math:`s_{mi} = \\sigma'(\\alpha_{0m}+\\alpha_m^T x_i)
      \\sum_k \\beta_{km} \\delta_{ki}` ;
    * :math:`\\partial R / \\partial \\beta_{km} = \\sum_i \\delta_{ki} z_{mi}`,
      :math:`\\partial R / \\partial \\alpha_{m\\ell} = \\sum_i s_{mi} x_{i\\ell}`.

    Parameters
    ----------
    params : dict
        Poids ``alpha`` et ``beta`` (voir :func:`init_params`).
    X, Y : ndarray
        Entrées (N, p) et sorties (N, K).
    output : {"identity", "sigmoid", "softmax"}
        Fonction de sortie :math:`g`.
    loss : {"squared", "cross_entropy"}
        Erreur (11.9) ou (11.10).
    weight_decay : float
        Paramètre :math:`\\lambda \\ge 0`.

    Returns
    -------
    R : float
        Valeur de l'erreur pénalisée.
    grads : dict
        Gradients de même forme que ``params``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    fw = forward(params, X, output)
    Z, F = fw["Z"], fw["F"]
    R = data_loss(F, Y, loss) + weight_decay * penalty(params)

    # --- dérivée par rapport aux sorties T (les "delta") ---------------------
    if output == "softmax" and loss == "cross_entropy":
        delta = F - Y                                  # simplification (Ex. 11.3)
    else:
        if loss == "squared":
            dF = -2.0 * (Y - F)
        else:
            dF = -Y / np.clip(F, 1e-300, None)
        if output == "identity":
            delta = dF
        elif output == "sigmoid":
            delta = dF * F * (1.0 - F)
        else:  # softmax : jacobienne diag(f) - f f^T
            delta = F * (dF - np.sum(dF * F, axis=1, keepdims=True))

    # --- gradient de la couche de sortie -------------------------------------
    g_beta = np.empty_like(params["beta"])             # (K, M+1)
    g_beta[:, 0] = delta.sum(axis=0)
    g_beta[:, 1:] = delta.T @ Z
    g_beta[:, 1:] += 2.0 * weight_decay * params["beta"][:, 1:]
    grads = {"alpha": None, "beta": g_beta}

    # --- rétropropagation vers la couche cachée ------------------------------
    if params["alpha"] is not None:
        s = (delta @ params["beta"][:, 1:]) * Z * (1.0 - Z)   # sigma' = sigma(1-sigma)
        g_alpha = np.empty_like(params["alpha"])       # (M, p+1)
        g_alpha[:, 0] = s.sum(axis=0)
        g_alpha[:, 1:] = s.T @ X
        g_alpha[:, 1:] += 2.0 * weight_decay * params["alpha"][:, 1:]
        grads["alpha"] = g_alpha
    return R, grads


# ---------------------------------------------------------------------------
# Vérification numérique du gradient
# ---------------------------------------------------------------------------


def numerical_gradient(fun, theta, eps=1e-6):
    """Gradient de ``fun`` en ``theta`` par différences finies centrées.

    :math:`\\partial_j f(\\theta) \\approx
    [f(\\theta + \\varepsilon e_j) - f(\\theta - \\varepsilon e_j)] / (2\\varepsilon)`,
    avec une erreur en :math:`O(\\varepsilon^2)`.
    """
    theta = np.asarray(theta, dtype=float)
    g = np.zeros_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        g[j] = (fun(theta + e) - fun(theta - e)) / (2 * eps)
    return g


def gradient_check(params, X, Y, output="identity", loss="squared",
                   weight_decay=0.0, eps=1e-6):
    """Compare le gradient de :func:`backprop` à :func:`numerical_gradient`.

    Returns
    -------
    float
        Erreur relative :math:`\\|g - g_{num}\\| / \\max(\\|g\\| + \\|g_{num}\\|, 10^{-12})`.
    """
    p = X.shape[1]
    K = params["beta"].shape[0]
    M = 0 if params["alpha"] is None else params["alpha"].shape[0]

    def fun(th):
        return backprop(unpack(th, p, M, K), X, Y, output, loss, weight_decay)[0]

    g = pack(backprop(params, X, Y, output, loss, weight_decay)[1])
    gn = numerical_gradient(fun, pack(params), eps)
    return float(np.linalg.norm(g - gn) / max(np.linalg.norm(g) + np.linalg.norm(gn), 1e-12))


# ---------------------------------------------------------------------------
# Le réseau de neurones
# ---------------------------------------------------------------------------


class MLP:
    """Réseau de neurones à une couche cachée (modèle (11.5) du livre).

    Parameters
    ----------
    n_hidden : int
        Nombre M d'unités cachées (0 = pas de couche cachée).
    output : {"identity", "sigmoid", "softmax"}
        Fonction de sortie.
    loss : {"squared", "cross_entropy"}
        Erreur minimisée.
    weight_decay : float
        Paramètre :math:`\\lambda` de la pénalité (11.16).
    init_scale : float
        Poids initiaux uniformes sur [-init_scale, init_scale].
    random_state : int ou RandomState
        Graine des poids initiaux.

    Attributes
    ----------
    params_ : dict
        Poids ajustés.
    history_ : list of float
        Erreur pénalisée au cours des itérations / époques.
    """

    def __init__(self, n_hidden=10, output="identity", loss="squared",
                 weight_decay=0.0, init_scale=0.7, random_state=0):
        self.n_hidden = n_hidden
        self.output = output
        self.loss = loss
        self.weight_decay = weight_decay
        self.init_scale = init_scale
        self.random_state = random_state

    def _rng(self):
        if isinstance(self.random_state, np.random.RandomState):
            return self.random_state
        return np.random.RandomState(self.random_state)

    def init(self, p, K):
        """Initialise ``params_`` (appelé automatiquement par :meth:`fit`)."""
        self.p_, self.K_ = p, K
        self.params_ = init_params(p, self.n_hidden, K, self._rng(), self.init_scale)
        self.history_ = []
        return self

    def objective(self, X, Y):
        """Erreur pénalisée :math:`R(\\theta) + \\lambda J(\\theta)` actuelle."""
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        return backprop(self.params_, X, Y, self.output, self.loss, self.weight_decay)[0]

    def fit(self, X, Y, method="lbfgs", max_iter=500, lr=0.1, momentum=0.0,
            n_epochs=100, batch_size=None, callback=None, warm_start=False, tol=1e-10):
        """Ajuste les poids.

        Parameters
        ----------
        X : ndarray (N, p)
        Y : ndarray (N, K) ou (N,)
            Un vecteur est transformé en colonne.
        method : {"lbfgs", "gd", "sgd"}
            * ``"lbfgs"`` : quasi-Newton à mémoire limitée (``scipy``), une
              méthode « à métrique variable » recommandée en fin de 11.4 ;
            * ``"gd"`` : descente de gradient par lots (11.13), une époque =
              une mise à jour ; le pas est appliqué au gradient **moyen**
              (gradient de R/N) ;
            * ``"sgd"`` : descente de gradient par mini-lots de taille
              ``batch_size`` (1 = apprentissage « en ligne »), une époque =
              un passage sur toutes les données (ordre aléatoire).
        max_iter : int
            Nombre maximal d'itérations de L-BFGS.
        lr, momentum : float
            Pas d'apprentissage et coefficient d'inertie (``gd``/``sgd``).
        n_epochs : int
            Nombre d'époques (``gd``/``sgd``).
        batch_size : int ou None
            Taille des mini-lots pour ``sgd``.
        callback : callable ou None
            ``callback(model, epoch)`` appelé après chaque époque
            (``gd``/``sgd``) ou itération (``lbfgs``).
        warm_start : bool
            Si vrai, repart des poids actuels.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        N, p = X.shape
        K = Y.shape[1]
        if not (warm_start and hasattr(self, "params_")):
            self.init(p, K)
        M = self.n_hidden
        args = (self.output, self.loss, self.weight_decay)

        if method == "lbfgs":
            last = {}

            def fun(th):
                R, g = backprop(unpack(th, p, M, K), X, Y, *args)
                last["R"] = R
                return R, pack(g)

            it = [0]

            def cb(th):
                self.params_ = unpack(th.copy(), p, M, K)
                self.history_.append(last["R"])
                it[0] += 1
                if callback is not None:
                    callback(self, it[0])

            res = optimize.minimize(fun, pack(self.params_), jac=True, method="L-BFGS-B",
                                    callback=cb, options={"maxiter": max_iter,
                                                          "ftol": tol, "gtol": 1e-8})
            self.params_ = unpack(res.x, p, M, K)
            self.n_iter_ = res.nit
            return self

        rng = self._rng()
        theta = pack(self.params_)
        vel = np.zeros_like(theta)
        bs = N if (method == "gd" or batch_size is None) else batch_size
        for epoch in range(1, n_epochs + 1):
            order = np.arange(N) if method == "gd" else rng.permutation(N)
            for start in range(0, N, bs):
                idx = order[start:start + bs]
                # la pénalité est répartie au prorata de la taille du lot
                _, g = backprop(unpack(theta, p, M, K), X[idx], Y[idx], self.output,
                                self.loss, self.weight_decay * len(idx) / N)
                vel = momentum * vel - lr * pack(g) / len(idx)
                theta = theta + vel
            self.params_ = unpack(theta, p, M, K)
            self.history_.append(self.objective(X, Y))
            if callback is not None:
                callback(self, epoch)
        return self

    def predict(self, X):
        """Sorties :math:`f_k(x)` (matrice (N, K))."""
        return forward(self.params_, X, self.output)["F"]

    def hidden(self, X):
        """Valeurs des unités cachées :math:`z_{mi}` (matrice (N, M))."""
        return forward(self.params_, X, self.output)["Z"]

    def predict_class(self, X):
        """Classe prédite :math:`G(x) = \\arg\\max_k f_k(x)`."""
        return np.argmax(self.predict(X), axis=1)


# ---------------------------------------------------------------------------
# Splines de lissage (pour la PPR)
# ---------------------------------------------------------------------------


def _reinsch_QR(x):
    """Matrices Q (N, N-2) et R (N-2, N-2) de Reinsch pour des x triés distincts.

    Pour une spline cubique naturelle de valeurs f aux nœuds x, on a
    :math:`\\int f''(t)^2 dt = f^T Q R^{-1} Q^T f`.
    """
    h = np.diff(x)
    n = len(x)
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


class SmoothingSpline:
    """Spline de lissage cubique (chapitre 5), paramétrée par λ ou par ddl.

    Minimise :math:`\\sum_i (y_i - g(x_i))^2 + \\lambda \\int g''(t)^2 dt`.
    La solution est une spline cubique naturelle à nœuds aux :math:`x_i` dont
    les valeurs ajustées sont :math:`\\hat g = (I + \\lambda K)^{-1} y` avec
    :math:`K = Q R^{-1} Q^T`. Le nombre de degrés de liberté effectifs est
    :math:`\\mathrm{tr}(I+\\lambda K)^{-1} = \\sum_j 1/(1+\\lambda d_j)` où les
    :math:`d_j` sont les valeurs propres de K.

    Parameters
    ----------
    df : float ou None
        Degrés de liberté visés (entre 2 et N) ; prioritaire sur ``lam``.
    lam : float ou None
        Paramètre de lissage λ.
    """

    def __init__(self, df=5.0, lam=None):
        self.df = df
        self.lam = lam

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        if np.any(np.diff(xs) <= 1e-12 * max(1.0, np.ptp(xs))):
            raise ValueError("les abscisses doivent être distinctes")
        Q, R = _reinsch_QR(xs)
        Kmat = Q @ np.linalg.solve(R, Q.T)
        Kmat = (Kmat + Kmat.T) / 2
        d, U = np.linalg.eigh(Kmat)
        d = np.clip(d, 0.0, None)
        lam = self.lam if self.df is None else df_to_lambda(d, self.df)
        self.lam_ = lam
        shrink = 1.0 / (1.0 + lam * d)
        self.df_ = float(shrink.sum())
        fitted = U @ (shrink * (U.T @ ys))
        self.x_, self.fitted_ = xs, fitted
        self._spline = CubicSpline(xs, fitted, bc_type="natural")
        return self

    def predict(self, x):
        """Valeurs de la spline ; extrapolation linéaire hors de [min x, max x]."""
        x = np.asarray(x, dtype=float)
        out = self._spline(np.clip(x, self.x_[0], self.x_[-1]))
        lo, hi = x < self.x_[0], x > self.x_[-1]
        d1 = self._spline.derivative()
        out[lo] += d1(self.x_[0]) * (x[lo] - self.x_[0])
        out[hi] += d1(self.x_[-1]) * (x[hi] - self.x_[-1])
        return out

    def derivative(self, x):
        """Dérivée g'(x) (constante hors du support des données)."""
        x = np.clip(np.asarray(x, dtype=float), self.x_[0], self.x_[-1])
        return self._spline.derivative()(x)


def df_to_lambda(d, df):
    """λ tel que :math:`\\sum_j 1/(1+\\lambda d_j) = df` (recherche dichotomique en log λ).

    Les deux valeurs propres nulles (fonctions affines, non pénalisées)
    imposent :math:`df > 2`.
    """
    n = len(d)
    if not 2.0 < df < n:
        raise ValueError("df doit être strictement entre 2 et N")

    def f(loglam):
        return np.sum(1.0 / (1.0 + np.exp(loglam) * d)) - df

    lo, hi = -40.0, 40.0
    return float(np.exp(optimize.brentq(f, lo, hi, xtol=1e-10)))


# ---------------------------------------------------------------------------
# Projection pursuit regression
# ---------------------------------------------------------------------------


class PPR:
    """Projection pursuit regression, modèle (11.1) : :math:`f(X)=\\sum_m g_m(\\omega_m^T X)`.

    Ajustement progressif (*forward stagewise*) : on ajoute un terme
    :math:`(\\omega_m, g_m)` à la fois, en alternant

    1. lissage : :math:`g` = spline de lissage (ddl fixés) des résidus en
       fonction de :math:`v_i = \\omega^T x_i` ;
    2. mise à jour de ω par Gauss–Newton (éq. (11.3)–(11.4)) : moindres
       carrés pondérés de la cible
       :math:`\\omega_{old}^T x_i + (r_i - g(\\omega_{old}^T x_i))/g'(\\omega_{old}^T x_i)`
       sur :math:`x_i`, poids :math:`g'(\\omega_{old}^T x_i)^2`, sans
       constante, puis normalisation :math:`\\|\\omega\\|=1`.

    Parameters
    ----------
    n_terms : int
        Nombre M de fonctions ridge.
    df : float
        Degrés de liberté de chaque spline de lissage.
    n_iter : int
        Nombre maximal d'alternances par terme.
    n_starts : int
        Nombre de directions initiales essayées pour chaque terme (la
        première est la direction des moindres carrés, les autres sont
        aléatoires) ; on garde la meilleure.
    backfit : bool
        Réajuster les :math:`g_m` précédents par *backfitting* après l'ajout
        de chaque terme (les :math:`\\omega_m` restent fixés).
    refit_directions : bool
        Si vrai, après l'ajout de chaque terme, on effectue quelques cycles
        où **chaque** couple :math:`(\\omega_l, g_l)` est réajusté (par
        l'algorithme alterné, en partant du :math:`\\omega_l` courant) aux
        résidus partiels :math:`y - \\bar y - \\sum_{l' \\ne l} g_{l'}`.
        Le livre note que c'est possible « en principe » mais rarement fait.
    random_state : int
        Graine.
    """

    def __init__(self, n_terms=2, df=5.0, n_iter=30, n_starts=4, backfit=True,
                 refit_directions=False, random_state=0):
        self.n_terms = n_terms
        self.df = df
        self.n_iter = n_iter
        self.n_starts = n_starts
        self.backfit = backfit
        self.refit_directions = refit_directions
        self.random_state = random_state

    def _fit_term(self, X, r, w):
        """Ajuste un seul terme (ω, g) aux résidus r à partir de la direction w."""
        w = w / np.linalg.norm(w)
        g = SmoothingSpline(df=self.df).fit(X @ w, r)
        rss = np.sum((r - g.predict(X @ w)) ** 2)
        for _ in range(self.n_iter):
            v = X @ w
            gv, dg = g.predict(v), g.derivative(v)
            # moindres carrés sans constante : lignes g'(v_i) x_i,
            # réponse g'(v_i) v_i + (r_i - g(v_i))  (forme (11.4) sans division)
            A = dg[:, None] * X
            b = dg * v + (r - gv)
            w_gn = np.linalg.lstsq(A, b, rcond=None)[0]
            if not np.all(np.isfinite(w_gn)) or np.linalg.norm(w_gn) == 0:
                break
            step, improved = 1.0, False
            for _ in range(8):                         # demi-pas si nécessaire
                w_new = w + step * (w_gn - w)
                w_new /= np.linalg.norm(w_new)
                g_new = SmoothingSpline(df=self.df).fit(X @ w_new, r)
                rss_new = np.sum((r - g_new.predict(X @ w_new)) ** 2)
                if rss_new < rss:
                    improved = True
                    break
                step /= 2
            if not improved:
                break
            converged = rss - rss_new < 1e-8 * max(rss, 1e-12)
            w, g, rss = w_new, g_new, rss_new
            if converged:
                break
        return w, g, rss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.RandomState(self.random_state)
        N, p = X.shape
        self.mean_ = y.mean()
        self.omegas_, self.smoothers_ = [], []
        r = y - self.mean_
        self.rss_path_ = [float(np.sum(r ** 2))]
        for m in range(self.n_terms):
            starts = [np.linalg.lstsq(X, r, rcond=None)[0]]
            starts += [rng.randn(p) for _ in range(self.n_starts - 1)]
            best = None
            for w0 in starts:
                if np.linalg.norm(w0) == 0:
                    continue
                cand = self._fit_term(X, r, w0)
                if best is None or cand[2] < best[2]:
                    best = cand
            w, g, _ = best
            self.omegas_.append(w)
            self.smoothers_.append(g)
            if self.refit_directions and m > 0:
                self._refit(X, y)
            elif self.backfit and m > 0:
                self._backfit(X, y)
            r = y - self.predict(X)
            self.rss_path_.append(float(np.sum(r ** 2)))
        return self

    def _backfit(self, X, y, n_cycles=5):
        """Backfitting des g_m à directions fixées (chapitre 9)."""
        V = np.column_stack([X @ w for w in self.omegas_])
        for _ in range(n_cycles):
            for m in range(len(self.omegas_)):
                partial = y - self.mean_ - sum(self.smoothers_[l].predict(V[:, l])
                                               for l in range(len(self.omegas_)) if l != m)
                self.smoothers_[m] = SmoothingSpline(df=self.df).fit(V[:, m], partial)

    def _refit(self, X, y, n_cycles=5):
        """Réajustement cyclique des couples (ω_l, g_l) sur les résidus partiels."""
        for _ in range(n_cycles):
            for m in range(len(self.omegas_)):
                partial = y - self.mean_ - sum(
                    self.smoothers_[l].predict(X @ self.omegas_[l])
                    for l in range(len(self.omegas_)) if l != m)
                w, g, _ = self._fit_term(X, partial, self.omegas_[m])
                self.omegas_[m], self.smoothers_[m] = w, g

    def ridge_functions(self, X):
        """Matrice (N, M) des contributions :math:`g_m(\\omega_m^T x_i)`."""
        X = np.asarray(X, dtype=float)
        return np.column_stack([g.predict(X @ w) for w, g in zip(self.omegas_, self.smoothers_)])

    def predict(self, X):
        """Prédiction :math:`\\bar y + \\sum_m g_m(\\omega_m^T x)`."""
        return self.mean_ + self.ridge_functions(X).sum(axis=1)
