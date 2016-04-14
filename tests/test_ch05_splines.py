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
