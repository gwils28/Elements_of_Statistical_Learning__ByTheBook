"""Méthodes linéaires de classification - chapitre 4 d'ESL.

Implémentations explicites (numpy / scipy) de :

* la régression linéaire sur matrice indicatrice (section 4.2) ;
* l'analyse discriminante linéaire (LDA), quadratique (QDA) et régularisée
  (RDA) (section 4.3) ;
* la LDA de rang réduit / variables canoniques de Fisher (section 4.3.3) ;
* la régression logistique binaire par Newton-Raphson / IRLS (section 4.4.1),
  multinomiale par Newton-Raphson (exercice 4.4) et pénalisée L1 par
  descente par coordonnées (section 4.4.4) ;
* le perceptron de Rosenblatt et l'hyperplan séparateur optimal
  (section 4.5).

Convention : les étiquettes de classe ``y`` peuvent être quelconques ; elles
sont ordonnées par ``np.unique`` et stockées dans ``model["classes"]``.
"""

import numpy as np
from scipy import linalg, optimize


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

def indicator_matrix(y, classes=None):
    """Matrice indicatrice (codage « un parmi K ») des réponses.

    Parameters
    ----------
    y : array (N,)
        Étiquettes.
    classes : array (K,), optional
        Ordre des classes (par défaut ``np.unique(y)``).

    Returns
    -------
    Y : array (N, K)
        ``Y[i, k] = 1`` si ``y[i] == classes[k]``, 0 sinon.
    """
    classes = np.unique(y) if classes is None else np.asarray(classes)
    return (np.asarray(y)[:, None] == classes[None, :]).astype(float)


def error_rate(y_true, y_pred):
    """Taux d'erreur de classification (proportion de prédictions fausses)."""
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def _add_intercept(X):
    X = np.asarray(X, float)
    return np.column_stack([np.ones(X.shape[0]), X])


# ---------------------------------------------------------------------------
# 4.2  Régression sur matrice indicatrice
# ---------------------------------------------------------------------------

def indicator_regression_fit(X, y):
    """Régression linéaire (avec constante) de la matrice indicatrice sur X.

    Returns
    -------
    model : dict
        ``B`` (p+1, K) coefficients (première ligne = constantes) et
        ``classes``.
    """
    classes = np.unique(y)
    Y = indicator_matrix(y, classes)
    Xa = _add_intercept(X)
    B, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    return {"B": B, "classes": classes}


def indicator_regression_predict(model, X, return_fitted=False):
    """Classe de plus grande composante ajustée, règle (4.4) du livre."""
    F = _add_intercept(X) @ model["B"]
    pred = model["classes"][np.argmax(F, axis=1)]
    return (pred, F) if return_fitted else pred
