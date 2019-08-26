import numpy as np
import pytest
from scipy import stats

from esl import ch18_grande_dimension_complements as m


def test_sigmoid_stable():
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    s = m.sigmoid(z)
    assert np.all(np.isfinite(s))
    assert np.allclose(s, 1 / (1 + np.exp(-np.clip(z, -50, 50))))
