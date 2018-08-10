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
