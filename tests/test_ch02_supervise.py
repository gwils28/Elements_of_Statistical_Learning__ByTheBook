import numpy as np
import pytest

from esl import ch02_supervise as c2
from esl import datasets


def test_least_squares_matches_normal_equations():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 3))
    y = 1 + X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=50)
    beta = c2.least_squares_fit(X, y)
    A = c2.add_intercept(X)
    np.testing.assert_allclose(beta, np.linalg.solve(A.T @ A, A.T @ y))
    # résidus orthogonaux aux colonnes (éq. (2.5))
    np.testing.assert_allclose(A.T @ (y - c2.least_squares_predict(X, beta)), 0, atol=1e-10)


def test_weighted_least_squares():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    w = rng.uniform(0.5, 2, size=30)
    beta = c2.least_squares_fit(X, y, weights=w)
    A = c2.add_intercept(X)
    W = np.diag(w)
    np.testing.assert_allclose(beta, np.linalg.solve(A.T @ W @ A, A.T @ W @ y))


def test_knn_regress_limits():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=40)
    # k = 1 sur l'échantillon d'apprentissage : on retrouve y
    np.testing.assert_allclose(c2.knn_regress(X, y, X, 1), y)
    # k = N : moyenne globale
    np.testing.assert_allclose(c2.knn_regress(X, y, X[:5], 40), np.full(5, y.mean()))


def test_knn_classify_matches_sklearn():
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.RandomState(3)
    X = rng.normal(size=(100, 3))
    g = rng.randint(0, 3, size=100)
    Xt = rng.normal(size=(60, 3))
    for k in (1, 5, 15):
        ref = KNeighborsClassifier(n_neighbors=k).fit(X, g).predict(Xt)
        np.testing.assert_array_equal(c2.knn_classify(X, g, Xt, k), ref)


def test_bayes_posterior_matches_book_data():
    m = datasets.load_mixture_example()
    p = c2.bayes_posterior(m["xnew"], m["means"])
    np.testing.assert_allclose(p, m["prob"], atol=1e-10)


def test_sample_mixture_shapes():
    rng = np.random.RandomState(0)
    means = c2.mixture_centers(rng)
    X, g = c2.sample_mixture(means, 100, rng)
    assert X.shape == (200, 2)
    assert (g[:100] == 0).all() and (g[100:] == 1).all()


def test_median_nn_distance_formula_and_simulation():
    assert c2.median_nn_distance(10, 500) == pytest.approx(0.518, abs=1e-3)
    rng = np.random.RandomState(0)
    p, N = 3, 20
    d = np.array([np.linalg.norm(c2.sample_unit_ball(N, p, rng), axis=1).min()
                  for _ in range(4000)])
    assert np.median(d) == pytest.approx(c2.median_nn_distance(p, N), abs=0.01)


def test_edge_length():
    assert c2.edge_length(0.01, 10) == pytest.approx(0.631, abs=1e-3)
    assert c2.edge_length(0.1, 10) == pytest.approx(0.794, abs=1e-3)


def test_smoother_weights():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    x0 = np.array([0.3, -0.2])
    l = c2.ols_weights(X, x0)
    beta = c2.least_squares_fit(X, y)
    assert l @ y == pytest.approx(c2.least_squares_predict(x0[None, :], beta)[0])
    assert l.sum() == pytest.approx(1.0)  # avec constante, les poids somment à 1
    w = c2.knn_weights(X, x0, 7)
    assert w.sum() == pytest.approx(1.0)
    assert w @ y == pytest.approx(c2.knn_regress(X, y, x0[None, :], 7)[0])


def test_reduce_ties_gives_same_fit():
    rng = np.random.RandomState(5)
    x = rng.randint(0, 6, size=80).astype(float)
    y = 2 - x + rng.normal(size=80)
    xu, yb, n = c2.reduce_ties(x, y)
    assert n.sum() == 80 and len(xu) == len(np.unique(x))
    np.testing.assert_allclose(c2.least_squares_fit(xu, yb, weights=n),
                               c2.least_squares_fit(x, y))
