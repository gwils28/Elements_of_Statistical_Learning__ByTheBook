"""Chargement des jeux de données utilisés dans ESL.

Les fichiers sont stockés dans le dossier ``data/`` à la racine du dépôt
(format CSV, compressés en ``.csv.gz`` pour les plus volumineux).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(name):
    """Lit ``data/<name>.csv`` ou ``data/<name>.csv.gz``."""
    for suffix in (".csv", ".csv.gz"):
        path = DATA_DIR / (name + suffix)
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Jeu de données introuvable : %s" % name)


def load_prostate():
    """Cancer de la prostate (Stamey et al., 1989) - chapitre 3.

    Retourne un DataFrame avec les 8 prédicteurs, la réponse ``lpsa`` et la
    colonne booléenne ``train`` (séparation apprentissage/test du livre).
    """
    return _read("prostate")


def load_saheart():
    """South African Heart Disease - chapitre 4. Réponse : ``chd`` (0/1)."""
    df = _read("SAheart")
    df["famhist"] = (df["famhist"] == "Present").astype(int)
    return df


def load_spam():
    """Spam e-mails (57 prédicteurs) - chapitres 9 et 10. Réponse : ``spam``."""
    df = _read("spam")
    df["spam"] = (df["spam"].astype(str) == "spam").astype(int)
    return df


def load_zip(split="train"):
    """Chiffres manuscrits (images 16x16) - chapitres 11, 13.

    Retourne ``(X, y)`` avec ``X`` de taille (N, 256), niveaux de gris dans
    [-1, 1], et ``y`` les chiffres 0..9.
    """
    df = _read("zip_%s" % split)
    arr = df.to_numpy()
    return arr[:, 1:].astype(float), arr[:, 0].astype(int)


def load_vowel(split="train"):
    """Voyelles (11 classes, 10 prédicteurs) - chapitre 4. Retourne ``(X, y)``."""
    df = _read("vowel_%s" % split)
    return df.drop(columns="y").to_numpy(float), df["y"].to_numpy(int)


def load_waveform(split="train"):
    """Waveform (3 classes, 21 prédicteurs) - chapitre 12. Retourne ``(X, y)``."""
    df = _read("waveform_%s" % split)
    return df.drop(columns="y").to_numpy(float), df["y"].to_numpy(int)


def load_phoneme():
    """Phonèmes (log-périodogrammes, 256 fréquences) - chapitre 5."""
    return _read("phoneme")


def load_bone():
    """Densité minérale osseuse relative - chapitre 5."""
    return _read("bone")


def load_galaxy():
    """Vitesses radiales de la galaxie NGC7531 - chapitre 6."""
    return _read("galaxy")


def load_ozone():
    """Ozone (New York, 1973) - chapitre 6."""
    return _read("ozone")


def load_marketing():
    """Questionnaire marketing (14 variables) - chapitre 14."""
    return _read("marketing")


def load_nci():
    """Micro-puces NCI60 : 6830 gènes x 64 lignées cellulaires - chapitres 14, 18.

    Retourne ``(X, labels)`` avec ``X`` de taille (64, 6830) (une ligne par
    échantillon) et ``labels`` le type de cancer de chaque échantillon.
    """
    df = _read("nci")
    return df.to_numpy(float).T, np.array([c.split(".")[0] for c in df.columns])


def load_countries():
    """Dissimilarités politiques entre 12 pays - chapitre 14."""
    path = DATA_DIR / "countries.csv"
    return pd.read_csv(path, index_col=0)


def load_mixture_example():
    """Simulation « mixture » des chapitres 2 et 13.

    Retourne un dictionnaire avec ``x`` (200, 2), ``y`` (200,), la grille
    ``xnew``, les probabilités bayésiennes ``prob`` sur la grille, ``px1``,
    ``px2`` (axes de la grille) et ``means`` (les 20 centres du mélange).
    """
    folder = DATA_DIR / "mixture_example"
    out = {}
    for path in sorted(folder.glob("*.csv")):
        arr = pd.read_csv(path).to_numpy()
        out[path.stem] = arr[:, 0] if arr.shape[1] == 1 else arr
    out["y"] = out["y"].astype(int)
    return out
