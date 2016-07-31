import numpy as np
import pytest
from scipy import integrate, stats

from esl import ch06_noyaux as nk


def _data(n=80, seed=0):
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(4 * x) + rng.normal(0, 1 / 3, n)
    return x, y


def test_kernels_integrals():
    assert integrate.quad(nk.epanechnikov, -1, 1)[0] == pytest.approx(1.0)
    assert integrate.quad(nk.gaussian, -np.inf, np.inf)[0] == pytest.approx(1.0)
    assert integrate.quad(nk.tricube, -1, 1)[0] == pytest.approx(81 / 70)
    assert nk.epanechnikov(1.5) == 0 and nk.tricube(-1.2) == 0


def test_nw_equals_local_constant():
    x, y = _data()
    x0 = np.linspace(0.05, 0.95, 15)
    a = nk.nadaraya_watson(x0, x, y, lam=0.2, kernel="tricube")
    b = nk.local_regression(x0, x, y, lam=0.2, degree=0, kernel="tricube")
    assert np.allclose(a, b)


def test_knn_mean_brute_force():
    x, y = _data()
    x0 = np.array([0.3, 0.71])
    for j, t in enumerate(x0):
        idx = np.argsort(np.abs(x - t))[:10]
        assert nk.knn_mean(x0, x, y, 10)[j] == pytest.approx(y[idx].mean())


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_moments_of_equivalent_kernel(degree):
    # b_j(x0) = sum_i (x_i - x0)^j l_i(x0) = 1 si j = 0, 0 pour 1 <= j <= degré
    x, _ = _data()
    x0 = np.array([0.0, 0.37, 0.93])
    L = nk.local_weights_matrix(x0, x, lam=0.3, degree=degree, kernel="tricube")
    for a, t in enumerate(x0):
        for j in range(degree + 1):
            bj = np.sum((x - t) ** j * L[a])
            assert bj == pytest.approx(1.0 if j == 0 else 0.0, abs=1e-8)


def test_local_linear_reproduces_lines_in_2d():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(60, 2))
    y = 1.0 - 2 * X[:, 0] + 0.5 * X[:, 1]
    X0 = rng.normal(size=(5, 2))
    f = nk.local_regression(X0, X, y, k=20, degree=1)
    assert np.allclose(f, 1.0 - 2 * X0[:, 0] + 0.5 * X0[:, 1])


def test_loocv_shortcut_matches_brute_force():
    x, y = _data(50)
    S = nk.smoother_matrix(x, lam=0.25, degree=1, kernel="tricube")
    brute = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        brute.append(nk.local_regression(x[i:i + 1], x[m], y[m], lam=0.25,
                                         degree=1, kernel="tricube")[0])
    assert nk.loocv_error(S, y) == pytest.approx(np.mean((y - np.array(brute)) ** 2))


def test_varying_coefficient_through_monomials():
    rng = np.random.RandomState(2)
    n = 200
    z = rng.uniform(0, 1, n)
    xx = rng.normal(size=n)
    y = 1 + 2 * z + (1 + z) * xx  # alpha(z), beta(z) affines : reproduits exactement
    X = np.column_stack([xx, z])
    A = np.diag([0.0, 1.0])  # le noyau ne regarde que z
    monos = [(0, 0), (1, 0), (0, 1), (1, 1)]
    X0 = np.array([[0.7, 0.4], [-1.0, 0.6]])
    f = nk.local_regression(X0, X, y, lam=0.3, kernel="tricube", A=A, monomials=monos)
    truth = 1 + 2 * X0[:, 1] + (1 + X0[:, 1]) * X0[:, 0]
    assert np.allclose(f, truth)


def test_logistic_irls_matches_known_optimum():
    rng = np.random.RandomState(3)
    X = np.column_stack([np.ones(300), rng.normal(size=300)])
    y = (rng.uniform(size=300) < 1 / (1 + np.exp(-(0.5 + X[:, 1])))).astype(float)
    b = nk.logistic_irls(X, y)
    p = 1 / (1 + np.exp(-X @ b))
    assert np.allclose(X.T @ (y - p), 0, atol=1e-8)  # équations du score


def test_local_constant_logistic_is_nw():
    # exercice 6.5 : logit localement constant = moyenne de Nadaraya-Watson
    rng = np.random.RandomState(4)
    x = rng.uniform(0, 1, 150)
    y = (rng.uniform(size=150) < x).astype(float)
    x0 = np.array([0.2, 0.5, 0.8])
    p, _ = nk.local_logistic(x0, x, y, lam=0.3, degree=0, ridge=0.0)
    assert np.allclose(p, nk.nadaraya_watson(x0, x, y, lam=0.3, kernel="tricube"))


def test_kde_matches_scipy_and_integrates():
    rng = np.random.RandomState(5)
    x = rng.normal(size=100)
    lam = 0.4
    grid = np.linspace(-5, 5, 2001)
    f = nk.kde(grid, x, lam)
    ref = stats.gaussian_kde(x, bw_method=lam / np.std(x, ddof=1))(grid)
    assert np.allclose(f, ref)
    assert np.trapezoid(f, grid) == pytest.approx(1.0, abs=1e-4)
    fe = nk.kde(grid, x, lam, kernel="epanechnikov")
    assert np.trapezoid(fe, grid) == pytest.approx(1.0, abs=1e-3)


def test_kde_posterior_rows_sum_to_one():
    rng = np.random.RandomState(6)
    x = np.r_[rng.normal(0, 1, 50), rng.normal(2, 1, 30)]
    g = np.r_[np.zeros(50), np.ones(30)]
    post, cl = nk.kde_posterior(np.linspace(-2, 4, 7), x, g, 0.5)
    assert np.allclose(post.sum(axis=1), 1)
    assert post[0, 0] > 0.9 and post[-1, 1] > 0.9


def test_naive_bayes_kde():
    rng = np.random.RandomState(7)
    n = 300
    y = rng.randint(0, 2, n)
    X = np.column_stack([rng.normal(2 * y, 1), rng.binomial(1, 0.2 + 0.6 * y)])
    nb = nk.NaiveBayesKDE(discrete=[False, True]).fit(X, y)
    P = nb.predict_proba(X)
    assert np.allclose(P.sum(axis=1), 1)
    assert np.mean(nb.predict(X) == y) > 0.8
