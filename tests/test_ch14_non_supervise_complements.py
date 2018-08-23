"""Tests du module esl.ch14_non_supervise_complements (exercices 14.13 à 14.25)."""

import numpy as np
import pytest

from esl import ch14_non_supervise_complements as nsc


# --- courbes principales et SOM (14.13) -------------------------------------

def test_project_on_polyline_segment():
    F = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    X = np.array([[0.5, 0.3], [1.4, 0.5], [-1.0, 0.0]])
    lam, d2 = nsc.project_on_polyline(X, F)
    np.testing.assert_allclose(lam, [0.5, 1.5, 0.0])
    np.testing.assert_allclose(d2, [0.09, 0.16, 1.0])


def test_kernel_smoother_reproduces_lines():
    t = np.linspace(0, 1, 50)
    y = 2 * t + 1
    np.testing.assert_allclose(nsc.kernel_smoother(t, y, t, 0.1), y, atol=1e-8)


def test_principal_curve_helix():
    rng = np.random.RandomState(0)
    s = np.linspace(0, 2 * np.pi, 200)
    X = np.c_[np.cos(s), np.sin(s), s] + 0.1 * rng.randn(200, 3)
    pc = nsc.principal_curve(X, n_iter=10)
    assert abs(np.corrcoef(pc["lam"], s)[0, 1]) > 0.99
    # distance moyenne au carré de l'ordre du bruit (3 x 0.01)
    assert pc["d2"].mean() < 0.05


def test_som_without_neighbourhood_moves_towards_data():
    rng = np.random.RandomState(0)
    X = np.r_[rng.randn(50, 2) * 0.1 + [3, 0], rng.randn(50, 2) * 0.1 - [3, 0]]
    M, coords = nsc.som_fit(X, (2,), n_epochs=20, radius=(0.0, 0.0))
    assert coords.shape == (2, 1)
    assert np.allclose(sorted(M[:, 0]), [-3, 3], atol=0.2)


# --- analyse factorielle (14.14, 14.15) --------------------------------------

def _fa_data(seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(4, 1)
    return rng.randn(400, 1) @ A.T + rng.randn(400, 4) * [0.5, 0.8, 1.0, 0.3]


def test_factor_ml_beats_or_matches_em():
    X = _fa_data()
    S = np.cov(X.T, bias=True)
    A, psi, ll = nsc.factor_analysis_ml(X, 1)
    A2, psi2, hist = nsc.factor_analysis_em(X, 1, n_iter=2000)
    assert np.all(np.diff(hist) > -1e-10)          # EM : vraisemblance croissante
    assert ll >= hist[-1] - 1e-6
    np.testing.assert_allclose(ll, nsc.factor_loglik(S, A, psi))
    np.testing.assert_allclose(A @ A.T + np.diag(psi), A2 @ A2.T + np.diag(psi2), atol=1e-3)


def test_factor_scale_equivariance():
    X = _fa_data(1) * [1.0, 10.0, 0.1, 5.0]
    A, psi, _ = nsc.factor_analysis_ml(X, 1)
    As, psis, _ = nsc.factor_analysis_ml(X / X.std(0), 1)
    A_star, psi_star = nsc.standardize_factor_model(A, psi)
    np.testing.assert_allclose(A_star @ A_star.T, As @ As.T, atol=1e-4)
    np.testing.assert_allclose(psi_star, psis, atol=1e-4)
    # la décomposition standardisée a une diagonale unité
    np.testing.assert_allclose((A_star ** 2).sum(1) + psi_star, 1.0)
