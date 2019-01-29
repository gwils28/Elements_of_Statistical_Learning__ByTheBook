import numpy as np
import pytest

from esl.ch16_ensembles import (
    adaboost_shrunk, block_correlated_data, extract_rules, forward_stagewise,
    isle_ensemble, l1_arc_length, lasso_cd, lasso_path_cd, lasso_path_fast, logistic_irls,
    ridge_path_df, rule_matrix, standardize, tree_predictions)


def _data(n=80, p=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    beta = np.r_[3, -2, 1.5, np.zeros(p - 3)]
    y = X @ beta + rng.randn(n)
    return X, y
