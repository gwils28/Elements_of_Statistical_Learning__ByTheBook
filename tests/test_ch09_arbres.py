"""Tests du module esl.ch09_arbres (modèles additifs, CART, PRIM, MARS)."""

import numpy as np
import pytest

from esl.ch09_arbres import (
    GAM, MARS, ArbreCART, MatriceLisseur, SplineLisseur, auc, backfitting,
    impurity, in_box, local_linear_matrix, logistic_irls, prim, prim_peel,
    roc_curve,
)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


# --- lisseurs ---------------------------------------------------------------

def test_spline_reproduit_les_droites(rng):
    x = np.sort(rng.uniform(0, 1, 60))
    sm = SplineLisseur(x, df=5)
    S = sm.matrix()
    lin = 2.0 - 3.0 * x
    assert np.allclose(S @ lin, lin, atol=1e-6)
    assert np.allclose(S, S.T, atol=1e-8)
    assert abs(np.trace(S) - 5) < 1e-3


def test_spline_egale_spline_de_lissage_scipy(rng):
    from scipy.interpolate import make_smoothing_spline
    x = np.sort(rng.uniform(0, 1, 40))
    y = np.sin(6 * x) + 0.3 * rng.randn(40)
    lam = 1e-4
    ours = SplineLisseur(x, lam=lam).fit(y)
    ref = make_smoothing_spline(x, y, lam=lam)(x)
    assert np.allclose(ours, ref, atol=1e-5)


def test_spline_valeurs_propres_ridge(rng):
    x = rng.uniform(0, 1, 30)
    S = SplineLisseur(x, lam=1e-3, ridge=1e-2).matrix()
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    assert ev.max() < 1 and ev.min() > -1e-8


def test_lineaire_local_reproduit_les_droites(rng):
    x = rng.uniform(0, 1, 50)
    S = local_linear_matrix(x, 0.1)
    assert np.allclose(S @ (1 + 4 * x), 1 + 4 * x)
    assert np.allclose(S.sum(1), 1)


# --- backfitting --------------------------------------------------------------

def test_backfitting_projections_egal_mco(rng):
    N = 80
    x1, x2 = rng.randn(N), rng.randn(N)
    x2 = 0.6 * x1 + 0.8 * x2
    y = 1 + x1 - 2 * x2 ** 2 + 0.5 * rng.randn(N)
    D1 = np.column_stack([x1, x1 ** 2])
    D2 = np.column_stack([x2, x2 ** 2, x2 ** 3])
    P = lambda D: D @ np.linalg.solve(D.T @ D, D.T)
    alpha, F, _ = backfitting(y - y.mean(), [MatriceLisseur(P(D1)), MatriceLisseur(P(D2))],
                              intercept=False, tol=1e-12, max_iter=5000)
    D = np.column_stack([D1, D2])
    beta = np.linalg.lstsq(D, y - y.mean(), rcond=None)[0]
    assert np.allclose(F.sum(1), D @ beta, atol=1e-6)


def test_backfitting_deux_lisseurs_formule_fermee(rng):
    N = 50
    x1, x2 = rng.uniform(size=N), rng.uniform(size=N)
    y = rng.randn(N)
    S1 = SplineLisseur(x1, lam=1e-3, ridge=0.05).matrix()
    S2 = SplineLisseur(x2, lam=1e-3, ridge=0.05).matrix()
    _, F, _ = backfitting(y, [MatriceLisseur(S1), MatriceLisseur(S2)],
                          intercept=False, tol=1e-13, max_iter=10000)
    I = np.eye(N)
    f2 = np.linalg.solve(I - S2 @ S1, S2 @ (I - S1) @ y)
    f1 = np.linalg.solve(I - S1 @ S2, S1 @ (I - S2) @ y)
    assert np.allclose(F[:, 0], f1, atol=1e-7)
    assert np.allclose(F[:, 1], f2, atol=1e-7)


def test_gam_gaussien_retrouve_fonctions(rng):
    N = 400
    X = rng.uniform(-1, 1, (N, 2))
    f1, f2 = np.sin(3 * X[:, 0]), X[:, 1] ** 2
    y = f1 + f2 + 0.1 * rng.randn(N)
    g = GAM(df=5, n_knots=15).fit(X, y)
    assert np.corrcoef(g.F_[:, 0], f1)[0, 1] > 0.99
    assert np.corrcoef(g.F_[:, 1], f2)[0, 1] > 0.99
    assert np.allclose(g.decision_function(X), g.alpha_ + g.F_.sum(1), atol=1e-8)


def test_irls_vs_sklearn(rng):
    from sklearn.linear_model import LogisticRegression
    X = rng.randn(300, 3)
    y = (X @ [1, -1, 0.5] + rng.randn(300) > 0).astype(int)
    b = logistic_irls(X, y)
    sk = LogisticRegression(C=1e10, tol=1e-10, max_iter=1000).fit(X, y)
    assert np.allclose(b, np.r_[sk.intercept_, sk.coef_[0]], atol=1e-4)


# --- arbres ---------------------------------------------------------------------

def test_impuretes_deux_classes():
    p = np.array([[0.5, 0.5], [1.0, 0.0], [0.2, 0.8]])
    assert np.allclose(impurity(p, "gini"), [0.5, 0, 2 * 0.2 * 0.8])
    assert np.allclose(impurity(p, "misclass"), [0.5, 0, 0.2])
    assert np.allclose(impurity(p, "entropy")[0], np.log(2))


def test_souche_optimale_par_force_brute(rng):
    x = rng.uniform(size=(60, 2))
    y = (x[:, 1] > 0.3) * 2.0 + 0.3 * rng.randn(60)
    t = ArbreCART(task="regression", max_leaves=2).fit(x, y)
    best = np.inf
    for j in range(2):
        for s in np.unique(x[:, j])[:-1]:
            L, R = y[x[:, j] <= s], y[x[:, j] > s]
            best = min(best, ((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum())
    R_tree = t.risk[t.is_leaf_].sum()
    assert np.isclose(R_tree, best)
    assert t.n_leaves() == 2


def test_arbre_regression_comme_sklearn(rng):
    from sklearn.tree import DecisionTreeRegressor
    X = rng.randn(200, 3)
    y = X[:, 0] * np.sin(X[:, 1]) + 0.2 * rng.randn(200)
    ours = ArbreCART(task="regression", min_samples_split=5).fit(X, y)
    sk = DecisionTreeRegressor(min_samples_split=5, random_state=0).fit(X, y)
    Xt = rng.randn(100, 3)
    # mêmes coupures, aux ex-aequo près (petits nœuds profonds)
    assert np.mean(np.isclose(ours.predict(Xt), sk.predict(Xt))) > 0.95
    assert np.allclose(ours.predict(X), sk.predict(X))
    assert ours.n_leaves() == sk.get_n_leaves()
    # chemin d'élagage : sklearn mesure le risque divisé par N
    a, trees = ours.pruning_path()
    ref = sk.cost_complexity_pruning_path(X, y).ccp_alphas
    assert np.allclose(a[1:] / len(y), ref[1:], rtol=1e-6, atol=1e-10)
