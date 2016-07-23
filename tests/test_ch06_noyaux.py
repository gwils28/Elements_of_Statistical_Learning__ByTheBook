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
