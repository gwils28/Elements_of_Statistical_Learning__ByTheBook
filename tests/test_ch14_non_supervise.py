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
