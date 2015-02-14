import numpy as np

from esl import datasets


def test_prostate_shape():
    df = datasets.load_prostate()
    assert df.shape == (97, 10)
    assert df["train"].sum() == 67


def test_saheart_binary():
    df = datasets.load_saheart()
    assert set(df["chd"].unique()) == {0, 1}
    assert set(df["famhist"].unique()) == {0, 1}


def test_spam():
    df = datasets.load_spam()
    assert df.shape == (4601, 58)
    assert df["spam"].sum() == 1813


def test_zip():
    X, y = datasets.load_zip("train")
    assert X.shape == (7291, 256)
    assert set(np.unique(y)) == set(range(10))
    assert X.min() >= -1 and X.max() <= 1


def test_mixture():
    m = datasets.load_mixture_example()
    assert m["x"].shape == (200, 2)
    assert m["xnew"].shape[0] == len(m["prob"]) == 6831


def test_nci():
    X, labels = datasets.load_nci()
    assert X.shape == (64, 6830)
    assert len(labels) == 64
