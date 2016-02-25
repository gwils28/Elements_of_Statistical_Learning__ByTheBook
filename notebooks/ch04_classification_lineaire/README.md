# Chapitre 4 - Méthodes linéaires de classification

Notebooks consacrés au chapitre 4 de *The Elements of Statistical Learning* : classifieurs dont
les frontières de décision sont des hyperplans (régression sur indicatrices, analyse
discriminante, régression logistique, hyperplans séparateurs).

Les fonctions réutilisables sont dans le module [`esl/ch04_classification.py`](../../esl/ch04_classification.py)
(LDA, QDA/RDA, LDA de rang réduit, logistique binaire par IRLS, logistique multinomiale par
Newton, logistique pénalisée L1, perceptron, hyperplan séparateur optimal), testé par
[`tests/test_ch04_classification.py`](../../tests/test_ch04_classification.py).

## Cours

| Notebook | Contenu |
|---|---|
| [`ch04_cours_1_indicatrices_lda_qda.ipynb`](ch04_cours_1_indicatrices_lda_qda.ipynb) | Frontières linéaires ; régression sur matrice indicatrice et **masquage** (fig. 4.2, 4.3) ; LDA déduite du modèle gaussien et de la formule de Bayes (fig. 4.5) ; QDA et LDA dans un espace augmenté (fig. 4.6) ; RDA sur les voyelles (fig. 4.7) ; LDA comme centroïde le plus proche après blanchiment ; tableau 4.1 (voyelles). |
| [`ch04_cours_2_lda_rang_reduit.ipynb`](ch04_cours_2_lda_rang_reduit.ipynb) | LDA de rang réduit : variables canoniques, problème de Fisher (quotient de Rayleigh) ; projections des voyelles (fig. 4.4, 4.8) ; direction discriminante contre droite des centroïdes (fig. 4.9) ; erreur selon la dimension (fig. 4.10) et frontières dans le plan canonique (fig. 4.11) ; lien avec la régression sur indicatrices. |
| [`ch04_cours_3_logistique_hyperplans.ipynb`](ch04_cours_3_logistique_hyperplans.ipynb) | Modèle logistique, Newton-Raphson et IRLS ; données SAheart (fig. 4.12, tableaux 4.2 et 4.3), sélection par Z-scores et par déviance, rapports de cotes, validation des erreurs standard par simulation ; logistique L1, sous-gradient et chemin (fig. 4.13) ; LDA contre logistique ; perceptron de Rosenblatt (fig. 4.14), géométrie des hyperplans (fig. 4.15), convergence et cycles ; hyperplan séparateur optimal, conditions KKT, points de support (fig. 4.16). |

## Exercices

| Notebook | Exercices |
|---|---|
| [`ch04_exercices_01-05.ipynb`](ch04_exercices_01-05.ipynb) | 4.1 à 4.5 |
| [`ch04_exercices_06-09.ipynb`](ch04_exercices_06-09.ipynb) | 4.6 à 4.9 |

- **Ex. 4.1** - Ramener $\max a^TBa$ sous $a^TWa = 1$ à un problème aux valeurs propres standard ($b = W^{1/2}a$).
- **Ex. 4.2** - À deux classes, les moindres carrés donnent la direction de la LDA, $\hat\Sigma^{-1}(\hat\mu_2-\hat\mu_1)$, pour tout codage ; le seuil diffère si $N_1 \neq N_2$.
- **Ex. 4.3** - La LDA sur les réponses ajustées $\hat Y$ de la régression sur indicatrices est identique à la LDA sur $X$.
- **Ex. 4.4** - Newton-Raphson (IRLS à poids non diagonaux) pour la logistique multinomiale, avec entrée agrandie ; application aux voyelles.
- **Ex. 4.5** - Sur des données séparables, le maximum de vraisemblance logistique n'existe pas (coefficients divergents), en dimension 1, $p$ et pour $K$ classes.
- **Ex. 4.6** - Preuve de convergence du perceptron en au plus $\|\beta_{\text{start}}-\beta_{\text{sep}}\|^2$ mises à jour.
- **Ex. 4.7** - Le critère « somme des distances signées » $D^*$ : non borné ou direction des centroïdes ; ce n'est pas l'hyperplan optimal.
- **Ex. 4.8** - Maximum de vraisemblance gaussien avec centroïdes de rang $L$ ; sa règle de Bayes est la LDA de rang réduit.
- **Ex. 4.9** - Programmation de la QDA, erreur de test sur les voyelles (0,53) et analyse du sur-apprentissage.

## Données utilisées

- `vowel` (voyelles, 11 classes, 10 prédicteurs) : `datasets.load_vowel("train" / "test")` ;
- `SAheart` (maladies cardiaques, Afrique du Sud) : `datasets.load_saheart()` ;
- données simulées (masquage, gaussiennes, exemples séparables).
