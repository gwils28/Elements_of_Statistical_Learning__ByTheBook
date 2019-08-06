import numpy as np
import pytest
from scipy import optimize, stats

from esl import ch18_grande_dimension as g


def _classif_data(rng, N=40, p=60, K=3):
    y = np.repeat(np.arange(K), N // K)
    X = rng.randn(len(y), p)
    X[:, :5] += y[:, None]
    return X, y


def test_soft_threshold():
    x = np.array([-3.0, -0.5, 0.0, 0.2, 2.0])
    np.testing.assert_allclose(g.soft_threshold(x, 1.0), [-2, 0, 0, 0, 1])


def test_nsc_delta_zero_is_diagonal_lda():
    rng = np.random.RandomState(0)
    X, y = _classif_data(rng)
    m = g.NearestShrunkenCentroids(delta=0.0).fit(X, y)
    np.testing.assert_allclose(m.shrunken_centroids_, m.centroids_)
    # LDA diagonale écrite à la main
    s = m.s_ + m.s0_
    sc = np.stack([-(((X - m.centroids_[k]) / s) ** 2).sum(1) + 2 * np.log(np.mean(y == k))
                   for k in range(3)], axis=1)
    np.testing.assert_allclose(m.discriminant(X), sc)
    np.testing.assert_allclose(m.predict_proba(X).sum(1), 1.0)


def test_nsc_large_delta_removes_all_features():
    rng = np.random.RandomState(1)
    X, y = _classif_data(rng)
    m = g.NearestShrunkenCentroids(delta=1e6).fit(X, y)
    assert len(m.active_features_) == 0
    np.testing.assert_allclose(m.shrunken_centroids_, np.tile(X.mean(0), (3, 1)))
    # le nombre de gènes actifs décroît avec delta
    counts = [len(m.set_delta(d).active_features_) for d in np.linspace(0, 6, 13)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
