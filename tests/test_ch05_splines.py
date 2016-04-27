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
