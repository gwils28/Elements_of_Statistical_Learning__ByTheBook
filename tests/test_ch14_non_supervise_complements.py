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


def test_hinton_example():
    rng = np.random.RandomState(0)
    Z = rng.randn(200, 3)
    X = np.c_[Z[:, 0], Z[:, 0] + 0.001 * Z[:, 1], 10 * Z[:, 2]]
    v = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    assert abs(v[2]) > 0.99
    a, psi, _ = nsc.factor_analysis_ml(X, 1)
    b = a[:, 0] / psi
    b /= np.linalg.norm(b)
    assert abs(b[2]) < 1e-4


# --- ACP à noyau (14.16, 14.17) ----------------------------------------------

def test_kernel_pca_matches_sklearn_and_linear_pca():
    from sklearn.decomposition import KernelPCA
    rng = np.random.RandomState(0)
    X, X0 = rng.randn(40, 3), rng.randn(4, 3)
    m = nsc.kernel_pca(nsc.rbf_kernel(X, X, 2.0))
    z0 = nsc.kernel_pca_transform(m, nsc.rbf_kernel(X0, X, 2.0))[:, :3]
    z0_sk = KernelPCA(3, kernel="rbf", gamma=0.5).fit(X).transform(X0)
    np.testing.assert_allclose(np.abs(z0), np.abs(z0_sk), atol=1e-8)
    # noyau linéaire : ACP classique, rang 3
    ml = nsc.kernel_pca(X @ X.T)
    assert len(ml["d"]) == 3
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    np.testing.assert_allclose(ml["d"], sv, rtol=1e-8)


def test_kernel_pca_polynomial_rank():
    rng = np.random.RandomState(1)
    X = rng.randn(50, 2)
    assert len(nsc.kernel_pca((1 + X @ X.T) ** 2)["d"]) == 5


# --- densité gaussienne inclinée (14.18) -------------------------------------

@pytest.mark.parametrize("lam", [0.1, 10.0, 1e3])
def test_tilted_gaussian_moments(lam):
    rng = np.random.RandomState(0)
    s = rng.uniform(-1, 1, 500)
    s = (s - s.mean()) / s.std()
    fit = nsc.tilted_gaussian_fit(s, lam=lam, order=3)
    t, f, d, y = fit["grid"], fit["f"], fit["delta"], fit["y"]
    for k in range(3):
        np.testing.assert_allclose((t ** k * f * d).sum(), (t ** k * y).sum(), atol=1e-8)


def test_tilted_gaussian_second_order_variants():
    rng = np.random.RandomState(0)
    s = rng.uniform(-1, 1, 500)
    s = (s - s.mean()) / s.std()
    m2 = lambda fit: (fit["grid"] ** 2 * fit["f"] * fit["delta"]).sum()
    target = (fit0 := nsc.tilted_gaussian_fit(s, 50.0, order=2))["grid"] ** 2 @ fit0["y"]
    assert abs(m2(fit0) - target) > 1e-3
    fit1 = nsc.tilted_gaussian_fit(s, 50.0, order=2, center_penalty=True)
    np.testing.assert_allclose(m2(fit1), target, atol=1e-8)


# --- ICA (14.19, 14.20) -----------------------------------------------------

def test_gaussian_term_rotation_invariant():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    v = [nsc.gaussian_loglik_term(nsc.random_orthogonal(3, rng), X) for _ in range(5)]
    np.testing.assert_allclose(v, v[0])


def test_whiten_and_fastica():
    rng = np.random.RandomState(0)
    S = rng.uniform(-np.sqrt(3), np.sqrt(3), (2000, 2))
    X = S @ np.array([[1.0, 2.0], [1.0, -1.0]]).T
    Z, _ = nsc.whiten(X)
    np.testing.assert_allclose(Z.T @ Z / len(Z), np.eye(2), atol=1e-10)
    A = nsc.fastica_symmetric(Z)
    np.testing.assert_allclose(A.T @ A, np.eye(2), atol=1e-10)
    C = np.abs(np.corrcoef((Z @ A).T, S.T)[:2, 2:])
    assert np.all(C.max(1) > 0.99)
    w, hist = nsc.fastica_one_unit(Z, rng.randn(2))
    assert len(hist) < 15 and abs(np.linalg.norm(w) - 1) < 1e-12


# --- graphes (14.21, 14.22) --------------------------------------------------

def test_laplacian_nullspace_counts_components():
    rng = np.random.RandomState(0)
    blocks = [3, 4, 5]
    N = sum(blocks)
    W = np.zeros((N, N))
    start = 0
    for b in blocks:
        B = rng.uniform(0.1, 1, (b, b))
        W[start:start + b, start:start + b] = B + B.T
        start += b
    np.fill_diagonal(W, 0)
    L = nsc.graph_laplacian(W)
    np.testing.assert_allclose(L.sum(1), 0, atol=1e-12)
    ev = np.linalg.eigvalsh(L)
    assert np.sum(ev < 1e-10) == 3
    assert nsc.connected_components(W).max() + 1 == 3


def test_knn_graph_symmetric():
    rng = np.random.RandomState(0)
    W = nsc.knn_graph(rng.randn(30, 2), 4, c=1.0)
    np.testing.assert_allclose(W, W.T)
    assert np.all(np.diag(W) == 0) and np.all((W > 0).sum(1) >= 4)


def test_pagerank_book_example():
    L = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]])
    p, _ = nsc.pagerank_power(L)
    np.testing.assert_allclose(p, [1.49, 0.78, 1.58, 0.15], atol=0.005)
    np.testing.assert_allclose(p.sum(), 4)
    np.testing.assert_allclose(p, nsc.pagerank_linear(L), atol=1e-10)
    A = nsc.pagerank_matrix(L)
    np.testing.assert_allclose(A.sum(0), 1)
