"""Tests du module esl.ch14_non_supervise (chapitre 14)."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from esl import ch14_non_supervise as ns


# --- règles d'association ---------------------------------------------------

def _apriori_force_brute(Z, t, taille_max):
    out = {}
    for k in range(1, taille_max + 1):
        for S in combinations(range(Z.shape[1]), k):
            s = Z[:, list(S)].all(1).mean()
            if s >= t:
                out[S] = s
    return out


def test_apriori_egal_force_brute():
    rng = np.random.RandomState(0)
    Z = rng.uniform(size=(300, 8)) < np.linspace(0.3, 0.8, 8)
    Z[:, 1] |= Z[:, 0]
    a = ns.apriori(Z, 0.15, taille_max=4)
    b = _apriori_force_brute(Z, 0.15, 4)
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k])


def test_regles_confiance_lift():
    rng = np.random.RandomState(1)
    Z = rng.uniform(size=(500, 5)) < 0.6
    f = ns.apriori(Z, 0.2)
    R = ns.regles_association(f)
    for _, r in R.head(20).iterrows():
        A, B = list(r.antecedent), list(r.consequent)
        sAB = Z[:, A + B].all(1).mean()
        sA = Z[:, A].all(1).mean()
        sB = Z[:, B].all(1).mean()
        assert r.confiance == pytest.approx(sAB / sA)
        assert r.lift == pytest.approx(sAB / (sA * sB))
