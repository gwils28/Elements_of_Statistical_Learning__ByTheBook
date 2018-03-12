"""Tests du module esl.ch13_prototypes (chapitre 13)."""

import numpy as np
import pytest

from esl import ch13_prototypes as cp


def _blobs(rng, n=60):
    X = np.vstack([rng.randn(n, 2) * 0.3 + [-2, 0], rng.randn(n, 2) * 0.3 + [2, 0]])
    y = np.repeat([0, 1], n)
    return X, y


def test_sq_dists():
    rng = np.random.RandomState(0)
    A, B = rng.randn(5, 3), rng.randn(4, 3)
    ref = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    assert np.allclose(cp.sq_dists(A, B), ref)


def test_kmeans_monotone_and_sklearn():
    from sklearn.cluster import KMeans
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(50, 2) + c for c in ([0, 0], [5, 5], [0, 6])])
    C, lab, inertia, hist = cp.kmeans(X, 3, rng=rng)
    assert np.all(np.diff(hist) <= 1e-9)            # le critère décroît
    sk = KMeans(3, n_init=10, random_state=0).fit(X)
    assert inertia == pytest.approx(sk.inertia_, rel=1e-6)
    # chaque centre est la moyenne de son groupe
    for r in range(3):
        assert np.allclose(C[r], X[lab == r].mean(0))


def test_kmeans_classifier_separable():
    rng = np.random.RandomState(1)
    X, y = _blobs(rng)
    clf = cp.KMeansClassifier(R=3, rng=rng).fit(X, y)
    assert clf.prototypes_.shape == (6, 2)
    assert np.mean(clf.predict(X) == y) == 1.0


def test_lvq1_single_step():
    X = np.array([[0.0, 0.0]])
    protos = np.array([[1.0, 0.0], [5.0, 5.0]])
    # même classe : le prototype se rapproche
    M = cp.lvq1(X, np.array([0]), protos, np.array([0, 1]), n_iter=1, eps0=0.5)
    assert np.allclose(M[0], [0.5, 0.0]) and np.allclose(M[1], protos[1])
    # classe différente : il s'éloigne
    M = cp.lvq1(X, np.array([1]), protos, np.array([0, 1]), n_iter=1, eps0=0.5)
    assert np.allclose(M[0], [1.5, 0.0])


def test_lvq_classifier_runs():
    rng = np.random.RandomState(2)
    X, y = _blobs(rng)
    clf = cp.LVQClassifier(R=2, rng=rng).fit(X, y)
    assert np.mean(clf.predict(X) == y) == 1.0


def test_gmm_classifier_probabilities():
    rng = np.random.RandomState(3)
    X, y = _blobs(rng)
    clf = cp.GaussianMixtureClassifier(R=2, rng=rng).fit(X, y)
    P = clf.predict_proba(X)
    assert np.allclose(P.sum(1), 1.0)
    assert np.all(np.diff(clf.loglik_) > -1e-6)     # EM : vraisemblance croissante
    assert np.mean(clf.predict(X) == y) == 1.0


def test_knn_matches_sklearn():
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.RandomState(4)
    X = rng.randn(200, 3)
    y = (X[:, 0] + 0.5 * rng.randn(200) > 0).astype(int)
    Xt = rng.randn(100, 3)
    for k in (1, 5, 15):          # k impair : pas d'égalité à deux classes
        ref = KNeighborsClassifier(k).fit(X, y).predict(Xt)
        assert np.array_equal(cp.knn_predict(X, y, Xt, k), ref)


def test_knn_multi_k_consistent_and_1nn_train():
    rng = np.random.RandomState(5)
    X = rng.randn(80, 2)
    y = rng.randint(3, size=80)
    D = cp.sq_dists(X, X)
    P = cp.knn_from_dists(D, y, [1, 3, 7])
    assert np.array_equal(P[0], y)                  # 1-ppv : erreur d'apprentissage nulle
    assert np.array_equal(P[2], cp.knn_from_dists(D, y, 7))


def test_knn_cv_shapes():
    rng = np.random.RandomState(6)
    X, y = _blobs(rng, 40)
    m, se = cp.knn_cv_error(X, y, [1, 3, 5], n_folds=5, rng=rng)
    assert m.shape == (3,) and np.all(m <= 0.05) and np.all(se >= 0)


def test_dann_metric_properties():
    rng = np.random.RandomState(7)
    # classes séparées selon la 1re coordonnée seulement
    X = rng.uniform(-1, 1, size=(400, 2))
    y = (X[:, 0] > 0).astype(int)
    S = cp.dann_metric(X, y, eps=1.0)
    assert np.allclose(S, S.T)
    vals, vecs = np.linalg.eigh(S)
    assert np.all(vals > 0)
    # la direction la plus « pénalisée » est l'axe discriminant e1
    assert abs(vecs[0, -1]) > 0.99
    # une seule classe : B = 0 et Sigma = eps W^{-1}
    S1 = cp.dann_metric(X, np.zeros(400, int), eps=2.0)
    W = np.cov(X.T, bias=True)
    assert np.allclose(S1, 2.0 * np.linalg.inv(W), atol=1e-8)


def test_dann_predict_beats_chance():
    rng = np.random.RandomState(8)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = (X[:, 0] > 0).astype(int)
    Xt = rng.uniform(-1, 1, size=(100, 2))
    yt = (Xt[:, 0] > 0).astype(int)
    assert np.mean(cp.dann_predict(X, y, Xt, k=5, n_metric=30) == yt) > 0.9


def test_tangent_distance_properties():
    rng = np.random.RandomState(9)
    p = 20
    x, z = rng.randn(p), rng.randn(p)
    Tx, Tz = rng.randn(2, p), rng.randn(3, p)
    d_two = cp.tangent_distance(x, Tx, z, Tz, ridge=0)
    d_one = cp.tangent_distance(x, Tx, z, ridge=0)
    eu = ((x - z) ** 2).sum()
    assert d_two <= d_one + 1e-10 <= eu + 1e-10
    # point situé sur le plan tangent : distance nulle
    z2 = x + Tx.T @ np.array([0.3, -1.2])
    assert cp.tangent_distance(x, Tx, z2, ridge=0) == pytest.approx(0, abs=1e-18)
    # comparaison à un moindre carré direct
    A = np.vstack([Tx, -Tz]).T
    coef = np.linalg.lstsq(A, z - x, rcond=None)[0]
    assert d_two == pytest.approx(((z - x - A @ coef) ** 2).sum())
