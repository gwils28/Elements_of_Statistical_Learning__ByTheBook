# Chapitre 7 - Évaluation et sélection de modèles

Ce dossier couvre le chapitre 7 (*Model Assessment and Selection*) de *The Elements of
Statistical Learning* : comment estimer l'erreur de prédiction d'un modèle et s'en servir
pour choisir sa complexité. Les fonctions réutilisables se trouvent dans
`esl/ch07_selection.py` (tests : `tests/test_ch07_selection.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch07_cours_1_erreur_biais_variance.ipynb` | Erreur d'apprentissage, erreur de test conditionnelle et espérée (fig. 7.1) ; découpage apprentissage/validation/test ; décomposition biais-variance démontrée et vérifiée pour les k-PPV (7.10) et les moindres carrés (7.11)-(7.12) ; biais de modèle et d'estimation ; reproduction de la fig. 7.3 (régression et classification). |
| `ch07_cours_2_optimisme_criteres.ipynb` | Erreur intra-échantillon et optimisme (7.21)-(7.24) vérifiés par simulation ; Cp et AIC ; AIC sur les phonèmes « aa »/« ao » (fig. 7.4) ; nombre effectif de paramètres trace(S) et définition par covariance ; degrés de liberté d'une sélection adaptative ; BIC, approche bayésienne et MDL ; AIC contre BIC (convergence) ; dimension VC (fig. 7.5, 7.6) et bornes VC. |
| `ch07_cours_3_validation_croisee_bootstrap.ipynb` | Validation croisée à K blocs et courbe d'apprentissage (fig. 7.8) ; courbe CV et règle un écart-type (fig. 7.9) ; leave-one-out sans réajustement et GCV ; bonne et mauvaise manière de faire une CV (fig. 7.10) ; souches et CV (fig. 7.11) ; bootstrap naïf, leave-one-out, .632 et .632+ ; comparaison AIC/BIC/CV/bootstrap (fig. 7.7, 7.13) ; erreur conditionnelle contre erreur espérée (fig. 7.15). |

## Notebooks d'exercices

| Exercice | Notebook | Description |
|---|---|---|
| 7.1 | `ch07_exercices_01-05.ipynb` | Démonstration de l'estimation (7.24) de l'erreur intra-échantillon pour un ajustement linéaire. |
| 7.2 | `ch07_exercices_01-05.ipynb` | Décomposition de l'erreur de classement (7.62) et approximation normale (7.63) : biais et variance interagissent de façon multiplicative en perte 0-1. |
| 7.3 | `ch07_exercices_01-05.ipynb` | Résidu leave-one-out (y_i − f̂(x_i))/(1 − S_ii) pour moindres carrés et splines de lissage, inégalité sur les résidus, condition générale (échec pour les k-PPV). |
| 7.4 | `ch07_exercices_01-05.ipynb` | Optimisme moyen = (2/N) Σ Cov(ŷ_i, y_i) pour toute méthode d'ajustement, vérifié sur une sélection pas à pas. |
| 7.5 | `ch07_exercices_01-05.ipynb` | Σ Cov(ŷ_i, y_i) = trace(S) σ² pour un lisseur linéaire. |
| 7.6 | `ch07_exercices_06-10.ipynb` | Degrés de liberté effectifs des k plus proches voisins : N/k. |
| 7.7 | `ch07_exercices_06-10.ipynb` | GCV ≈ Cp avec la variance du bruit estimée par l'erreur d'apprentissage du modèle courant. |
| 7.8 | `ch07_exercices_06-10.ipynb` | La famille I(sin(αx) > 0) pulvérise les points 10^(−i) : dimension VC infinie (vérification en arithmétique exacte). |
| 7.9 | `ch07_exercices_06-10.ipynb` | Prostate : meilleur sous-ensemble, AIC, BIC, CV 5 et 10 blocs, bootstrap .632, erreur test, discussion. |
| 7.10 | `ch07_exercices_06-10.ipynb` | Souches sur 2^20 prédicteurs binaires : la CV correctement menée reste valable (≈ 50 %). |

## Notions clés

- Erreurs : apprentissage `err`, test conditionnelle `Err_T`, espérée `Err`, intra-échantillon `Err_in`.
- Décomposition biais-variance ; spécificités de la perte 0-1.
- Optimisme ω = (2/N) Σ Cov(ŷ_i, y_i) ; Cp, AIC, BIC ; df(S) = trace(S).
- Dimension de Vapnik-Chervonenkis, bornes VC et SRM.
- Validation croisée (K blocs, leave-one-out, GCV, règle un écart-type, bonne manière).
- Bootstrap : estimateurs naïf, leave-one-out, .632, .632+.
