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
