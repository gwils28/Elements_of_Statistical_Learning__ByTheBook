"""Tests du module esl.ch08_inference (chapitre 8)."""

import numpy as np
import pytest

from esl import ch08_inference as inf


@pytest.fixture
def smooth_data():
    rng = np.random.RandomState(0)
    x = np.sort(rng.uniform(0, 3, 50))
    y = 1 + 1.6 * np.sin(2.6 * x + 0.9) + 0.9 * rng.randn(50)
    knots = inf.bspline_knots(np.quantile(x, [0.25, 0.5, 0.75]), (0, 3))
    return x, y, knots


# --------------------------------------------------------------------------- B-splines

def test_bspline_partition_of_unity(smooth_data):
    x, _, knots = smooth_data
    H = inf.bspline_basis(np.linspace(0, 3, 101), knots)
    assert H.shape == (101, 7)
    assert np.all(H >= -1e-12)
    np.testing.assert_allclose(H.sum(1), 1.0, atol=1e-12)


def test_bspline_penalty_exact_on_polynomials(smooth_data):
    _, _, knots = smooth_data
    Om = inf.bspline_penalty(knots)
    xs = np.linspace(0, 3, 60)
    H = inf.bspline_basis(xs, knots)
    # coefficients représentant exactement 1, x et x^2 (polynômes de degré <= 3)
    for f, integral in [(lambda t: np.ones_like(t), 0.0), (lambda t: t, 0.0),
                        (lambda t: t ** 2, 4.0 * 3)]:          # int_0^3 (2)^2 = 12
        c = np.linalg.lstsq(H, f(xs), rcond=None)[0]
        assert np.allclose(H @ c, f(xs), atol=1e-10)
        assert c @ Om @ c == pytest.approx(integral, abs=1e-8)
    assert np.allclose(Om, Om.T)
    assert np.linalg.eigvalsh(Om).min() > -1e-8


def test_loocv_shortcut_matches_refit():
    rng = np.random.RandomState(1)
    x = np.sort(rng.uniform(0, 1, 40))
    y = np.sin(5 * x) + 0.3 * rng.randn(40)
    knots = inf.bspline_knots(np.unique(x)[1:-1], (x.min(), x.max()))
    H, Om, lam = inf.bspline_basis(x, knots), inf.bspline_penalty(knots), 1e-4
    S = inf.smoother_matrix(H, Om, lam)
    errs = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b = np.linalg.solve(H[m].T @ H[m] + lam * Om, H[m].T @ y[m])
        errs.append((y[i] - H[i] @ b) ** 2)
    assert inf.loocv_linear_smoother(S, y) == pytest.approx(np.mean(errs), rel=1e-8)
    assert inf.gcv_linear_smoother(S, y) > 0


# --------------------------------------------------- moindres carrés, bootstrap, Bayes

def test_ls_fit_matches_lstsq(smooth_data):
    x, y, knots = smooth_data
    H = inf.bspline_basis(x, knots)
    beta, s2, cov = inf.ls_fit(H, y)
    np.testing.assert_allclose(beta, np.linalg.lstsq(H, y, rcond=None)[0], atol=1e-10)
    assert s2 == pytest.approx(np.mean((y - H @ beta) ** 2))
    np.testing.assert_allclose(cov, np.linalg.inv(H.T @ H) * s2)


def test_parametric_bootstrap_matches_formula(smooth_data):
    x, y, knots = smooth_data
    H = inf.bspline_basis(x, knots)
    Hg = inf.bspline_basis(np.linspace(0, 3, 30), knots)
    beta, _, cov = inf.ls_fit(H, y)
    curves = inf.parametric_bootstrap_curves(H, y, Hg, 20000, np.random.RandomState(0))
    se = inf.pointwise_se(Hg, cov)
    np.testing.assert_allclose(curves.std(0), se, rtol=0.03)
    assert np.max(np.abs(curves.mean(0) - Hg @ beta)) < 4 * se.max() / np.sqrt(20000) * 3


def test_posterior_limits(smooth_data):
    x, y, knots = smooth_data
    H = inf.bspline_basis(x, knots)
    beta, s2, cov = inf.ls_fit(H, y)
    m, C = inf.gaussian_posterior(H, y, s2, 1e9)
    np.testing.assert_allclose(m, beta, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(C, cov, rtol=1e-5, atol=1e-8)
    m0, _ = inf.gaussian_posterior(H, y, s2, 1e-9)            # a priori très concentré
    assert np.abs(m0).max() < 1e-6
    # a priori N(0, tau I) : moyenne a posteriori = ridge de paramètre sigma2 / tau
    m1, _ = inf.gaussian_posterior(H, y, s2, 2.0)
    ridge = np.linalg.solve(H.T @ H + s2 / 2.0 * np.eye(7), H.T @ y)
    np.testing.assert_allclose(m1, ridge)


def test_bootstrap_indices_shape_and_range():
    idx = inf.bootstrap_indices(10, 5, np.random.RandomState(0))
    assert idx.shape == (5, 10) and idx.min() >= 0 and idx.max() <= 9


# ----------------------------------------------------------------------------- EM, Gibbs

def test_em_monotone_and_book_estimates():
    y = inf.MIXTURE_DATA
    res = inf.em_mixture(y, (4.5, y.var(), 1.0, y.var(), 0.5), n_iter=200, tol=1e-12)
    assert np.all(np.diff(res["loglik"]) >= -1e-10)
    mu1, v1, mu2, v2, pi = res["theta"]
    # valeurs du livre : 4.62, 0.87, 1.06, 0.77, 0.546
    assert mu1 == pytest.approx(4.62, abs=0.06)
    assert mu2 == pytest.approx(1.06, abs=0.06)
    assert v1 == pytest.approx(0.87, abs=0.08)
    assert v2 == pytest.approx(0.77, abs=0.08)
    assert pi == pytest.approx(0.546, abs=0.015)
    g = res["gamma"]
    assert np.all((g >= 0) & (g <= 1))
    assert np.mean(g) == pytest.approx(pi, abs=1e-6)        # point fixe


def test_em_is_stationary_point():
    y = inf.MIXTURE_DATA
    res = inf.em_mixture(y, (4.5, y.var(), 1.0, y.var(), 0.5), n_iter=500, tol=1e-14)
    th = np.array(res["theta"])
    eps = 1e-6
    for j in range(5):
        d = np.zeros(5); d[j] = eps
        grad = (inf.mixture_loglik(y, th + d) - inf.mixture_loglik(y, th - d)) / (2 * eps)
        assert abs(grad) < 1e-4


def test_Q_minus_R_decomposition():
    y = inf.MIXTURE_DATA
    rng = np.random.RandomState(3)
    for _ in range(20):
        th = (rng.uniform(0, 6), rng.uniform(.3, 3), rng.uniform(0, 6), rng.uniform(.3, 3),
              rng.uniform(.1, .9))
        thp = (rng.uniform(0, 6), rng.uniform(.3, 3), rng.uniform(0, 6), rng.uniform(.3, 3),
               rng.uniform(.1, .9))
        g, gp = inf.responsibilities(y, th), inf.responsibilities(y, thp)
        R = np.sum(g * np.log(gp) + (1 - g) * np.log(1 - gp))
        assert inf.mixture_loglik(y, thp) == pytest.approx(inf.mixture_Q(y, thp, th) - R)


def test_gibbs_mixture_centered_on_em():
    y = inf.MIXTURE_DATA
    mu1, v1, mu2, v2, pi = inf.em_mixture(y, (4.5, y.var(), 1.0, y.var(), 0.5),
                                          n_iter=200)["theta"]
    mus, props = inf.gibbs_mixture(y, v1, v2, pi, (5.0, 1.0), 2000, np.random.RandomState(0))
    assert mus.shape == (2000, 2) and props.shape == (2000,)
    assert mus[200:, 0].mean() == pytest.approx(mu1, abs=0.2)
    assert mus[200:, 1].mean() == pytest.approx(mu2, abs=0.2)
    assert props[200:].mean() == pytest.approx(pi, abs=0.05)


# -------------------------------------------------------------- bagging, souches, bumping

def test_bagging_error_curves_simple():
    # 3 modèles, 2 points test, 2 classes
    probas = np.array([[[0.9, 0.1], [0.4, 0.6]],
                       [[0.0, 1.0], [0.3, 0.7]],
                       [[0.3, 0.7], [0.45, 0.55]]])
    y = np.array([1, 1])
    ev, ep = inf.bagging_error_curves(probas, y)
    np.testing.assert_allclose(ev, [0.5, 0.5, 0.0])    # b=2 : égalité -> classe 0 (argmax)
    np.testing.assert_allclose(ep, [0.5, 0.0, 0.0])


def test_bagging_linear_smoother_parametric_converges(smooth_data):
    x, y, knots = smooth_data
    H = inf.bspline_basis(x, knots)
    beta, _, _ = inf.ls_fit(H, y)
    curves = inf.parametric_bootstrap_curves(H, y, H, 40000, np.random.RandomState(1))
    assert np.max(np.abs(curves.mean(0) - H @ beta)) < 0.02


def test_stump_matches_brute_force():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 3)
    y = np.where(X[:, 1] + 0.5 * rng.randn(40) > 0.2, 1, -1)
    w = rng.rand(40)
    s = inf.fit_stump(X, y, w)
    wn = w / w.sum()
    best = np.inf
    for j in range(3):
        for t in np.unique(X[:, j])[:-1]:
            left = X[:, j] <= t
            e = (min(wn[left & (y > 0)].sum(), wn[left & (y < 0)].sum())
                 + min(wn[~left & (y > 0)].sum(), wn[~left & (y < 0)].sum()))
            best = min(best, e)
    assert s["error"] == pytest.approx(best)
    pred = np.where(inf.stump_proba(s, X) > 0.5, 1, -1)
    assert np.sum(wn * (pred != y)) == pytest.approx(best)


def test_adaboost_reduces_training_error():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 2)
    y = np.where(X.sum(1) > 1, 1, -1)
    stumps, alphas = inf.adaboost_stumps(X, y, 100)
    err = np.mean(np.sign(inf.adaboost_decision(stumps, alphas, X)) != y)
    assert err < inf.fit_stump(X, y)["error"] / 2


def test_bumping_never_worse_than_original():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (100, 2))
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
    fit = lambda Xb, yb: inf.fit_stump(Xb, yb)
    loss = lambda s, X_, y_: np.mean(np.where(inf.stump_proba(s, X_) > 0.5, 1, -1) != y_)
    model, b, losses = inf.bumping(fit, loss, X, y, 15, rng)
    assert len(losses) == 16
    assert losses[b] == losses.min() <= losses[0]


# ----------------------------------------------------------------------------- stacking

def test_loo_residuals_linear_match_refit():
    rng = np.random.RandomState(0)
    X = rng.randn(25, 3)
    y = X @ [1.0, -2.0, 0.5] + rng.randn(25)
    _, r = inf.loo_residuals_linear(X, y)
    for i in range(25):
        m = np.arange(25) != i
        Xc = np.column_stack([np.ones(24), X[m]])
        b = np.linalg.lstsq(Xc, y[m], rcond=None)[0]
        assert r[i] == pytest.approx(y[i] - np.r_[1.0, X[i]] @ b)


def test_stacking_weights_constraints():
    rng = np.random.RandomState(0)
    F = rng.randn(60, 4)
    y = F @ [0.5, 0.3, 0.2, 0.0] + 0.1 * rng.randn(60)
    w_free = inf.stacking_weights(F, y, "none")
    w_pos = inf.stacking_weights(F, y, "nonneg")
    w_sx = inf.stacking_weights(F, y, "simplex")
    np.testing.assert_allclose(w_free, np.linalg.lstsq(F, y, rcond=None)[0])
    assert np.all(w_pos >= 0) and np.all(w_sx >= 0)
    assert w_sx.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w_sx, [0.5, 0.3, 0.2, 0.0], atol=0.05)
    rss = lambda w: np.sum((y - F @ w) ** 2)
    assert rss(w_free) <= rss(w_pos) + 1e-10 <= rss(w_sx) + 2e-10
    with pytest.raises(ValueError):
        inf.stacking_weights(F, y, "autre")
