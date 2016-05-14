import numpy as np
import pytest
from scipy.interpolate import BSpline, make_interp_spline

from esl.ch05_splines import (
    truncated_power_basis, piecewise_polynomial_basis, bspline_basis,
    augmented_knots, natural_spline_basis, bspline_penalty_matrix,
    SmoothingSpline, smoother_matrix, effective_df, lambda_for_df, loocv_score,
    haar_transform, inverse_haar_transform, haar_matrix, soft_threshold,
    logistic_irls,
)

rng = np.random.RandomState(0)
X = np.sort(rng.uniform(0, 1, 60))
KNOTS = np.array([0.2, 0.45, 0.7])


def test_truncated_power_shape_and_span():
    H = truncated_power_basis(X, KNOTS)
    assert H.shape == (60, 4 + 3)
    B = bspline_basis(X, KNOTS, 0, 1)
    # même espace : projection de B sur H sans résidu
    coef, *_ = np.linalg.lstsq(H, B, rcond=None)
    assert np.allclose(H @ coef, B, atol=1e-8)


def test_piecewise_dimensions():
    # (K+1)(d+1) - K(c+1) fonctions
    for c in (-1, 0, 1, 2):
        H = piecewise_polynomial_basis(X, KNOTS, degree=3, continuity=c)
        assert H.shape[1] == 4 * 4 - 3 * (c + 1)
        assert np.linalg.matrix_rank(H) == H.shape[1]


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_bspline_matches_scipy(order):
    t = augmented_knots(KNOTS, 0, 1, order)
    x = np.linspace(0, 0.999, 200)
    B = bspline_basis(x, KNOTS, 0, 1, order)
    ref = BSpline.design_matrix(x, t, order - 1).toarray()
    assert np.allclose(B, ref, atol=1e-12)


def test_bspline_partition_of_unity_and_positivity():
    x = np.linspace(0, 1, 301)  # bord droit inclus
    B = bspline_basis(x, KNOTS, 0, 1, 4)
    assert np.allclose(B.sum(1), 1)
    assert (B >= -1e-14).all()


def test_bspline_derivatives():
    x = np.linspace(0.01, 0.99, 50)
    t = augmented_knots(KNOTS, 0, 1, 4)
    c = rng.randn(len(KNOTS) + 4)
    spl = BSpline(t, c, 3)
    for d in (1, 2, 3):
        Bd = bspline_basis(x, KNOTS, 0, 1, 4, deriv=d)
        assert np.allclose(Bd @ c, spl.derivative(d)(x), atol=1e-8)


def test_penalty_matrix_exact():
    t = augmented_knots(KNOTS, 0, 1, 4)
    c = rng.randn(len(KNOTS) + 4)
    Om = bspline_penalty_matrix(KNOTS, 0, 1)
    f2 = BSpline(t, c, 3).derivative(2)
    grid = np.linspace(0, 1, 200001)
    num = np.trapezoid(f2(grid) ** 2, grid)
    assert np.isclose(c @ Om @ c, num, rtol=1e-6)
    # le noyau de Omega = fonctions linéaires
    ev = np.linalg.eigvalsh(Om)
    assert np.sum(ev < 1e-8 * ev.max()) == 2


def test_natural_spline_linear_outside():
    xi = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    N = natural_spline_basis(np.array([0.92, 0.95, 0.98, 0.0, 0.03, 0.06]), xi)
    # différences secondes nulles de chaque côté
    assert np.allclose(N[0] - 2 * N[1] + N[2], 0, atol=1e-12)
    assert np.allclose(N[3] - 2 * N[4] + N[5], 0, atol=1e-12)
    # c'est bien une spline cubique de nœuds xi
    H = truncated_power_basis(X, xi)
    Nx = natural_spline_basis(X, xi)
    coef, *_ = np.linalg.lstsq(H, Nx, rcond=None)
    assert np.allclose(H @ coef, Nx, atol=1e-10)


def test_smoothing_spline_limits():
    y = np.sin(6 * X) + 0.1 * rng.randn(60)
    # lambda tres grand : droite des moindres carres
    big = SmoothingSpline(lam=1e8).fit(X, y)
    line = np.polyval(np.polyfit(X, y, 1), X)
    assert np.allclose(big.predict(X), line, atol=1e-4)
    # lambda tres petit : interpolation (spline naturelle interpolante)
    small = SmoothingSpline(lam=1e-16).fit(X, y)
    assert np.allclose(small.predict(X), y, atol=1e-4)
    nat = make_interp_spline(X, y, k=3, bc_type="natural")
    xx = np.linspace(X[0], X[-1], 50)
    assert np.allclose(small.predict(xx), nat(xx), atol=1e-4)


def test_smoother_matrix_properties():
    S = smoother_matrix(X, 1e-4)
    assert np.allclose(S, S.T, atol=1e-10)
    ev = np.linalg.eigvalsh(S)
    assert ev.min() > -1e-10 and ev.max() < 1 + 1e-10
    assert np.sum(np.abs(ev - 1) < 1e-8) == 2
    # S reproduit les fonctions lineaires
    assert np.allclose(S @ (2 + 3 * X), 2 + 3 * X, atol=1e-8)


def test_df_and_lambda_inverse():
    lam = lambda_for_df(X, 7.0)
    assert np.isclose(effective_df(X, lam), 7.0, atol=1e-6)
    assert effective_df(X, 10 * lam) < 7.0


def test_loocv_formula_bruteforce():
    x = X[::3]
    y = np.cos(4 * x) + 0.2 * rng.randn(len(x))
    lam = 1e-4
    errs = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        s = SmoothingSpline(lam=lam).fit(x[m], y[m])
        errs.append(y[i] - s.predict(x[i:i + 1])[0])
    assert np.isclose(np.mean(np.square(errs)), loocv_score(x, y, lam), rtol=1e-6)


def test_haar():
    y = rng.randn(64)
    c = haar_transform(y)
    W = haar_matrix(64)
    assert np.allclose(W.T @ W, np.eye(64))
    assert np.allclose(W.T @ y, c)
    assert np.allclose(inverse_haar_transform(c), y)
    assert np.isclose(np.sum(c ** 2), np.sum(y ** 2))


def test_soft_threshold():
    assert np.allclose(soft_threshold([-3, -0.5, 0.2, 2], 1), [-2, 0, 0, 1])


def test_logistic_irls_vs_sklearn():
    from sklearn.linear_model import LogisticRegression
    Z = rng.randn(300, 3)
    p = 1 / (1 + np.exp(-(Z @ [1, -2, 0.5] + 0.3)))
    y = (rng.uniform(size=300) < p).astype(int)
    b = logistic_irls(np.column_stack([np.ones(300), Z]), y)
    sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(Z, y)
    assert np.allclose(b, np.r_[sk.intercept_, sk.coef_.ravel()], atol=1e-4)


def test_set_lambda_equals_refit():
    y = np.sin(5 * X) + 0.1 * rng.randn(60)
    s = SmoothingSpline(lam=1e-3).fit(X, y).set_lambda(1e-2)
    r = SmoothingSpline(lam=1e-2).fit(X, y)
    assert np.allclose(s.coef_, r.coef_) and np.isclose(s.df_, r.df_)
