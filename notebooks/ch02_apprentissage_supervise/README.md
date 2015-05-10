# Chapitre 2 - Vue d'ensemble de l'apprentissage supervisé

Le chapitre 2 pose le cadre commun à tout le livre : deux méthodes de référence
(moindres carrés et plus proches voisins), la théorie de la décision statistique, le fléau
de la dimension, les grandes familles d'estimateurs et le compromis biais-variance.

Les fonctions réutilisables (moindres carrés, $k$-ppv, simulation « mixture » et classifieur
de Bayes, formules du fléau de la dimension, poids de lissage, lisseurs) sont dans
[`esl/ch02_supervise.py`](../../esl/ch02_supervise.py), testées par
[`tests/test_ch02_supervise.py`](../../tests/test_ch02_supervise.py).

## Notebooks de cours

| Notebook | Sections du livre | Contenu |
|---|---|---|
| [`ch02_cours_1_moindres_carres_knn.ipynb`](ch02_cours_1_moindres_carres_knn.ipynb) | 2.1–2.3 | Notations ; moindres carrés (équations normales démontrées) ; $k$ plus proches voisins ; simulation « mixture » : figures 2.1 (frontière linéaire), 2.2 (15-ppv), 2.3 (1-ppv), 2.5 (classifieur de Bayes), 2.4 (erreurs d'apprentissage/test en fonction de $N/k$) ; nombre effectif de paramètres ; stabilité des frontières (biais/variance). |
| [`ch02_cours_2_decision_dimension.ipynb`](ch02_cours_2_decision_dimension.ipynb) | 2.4–2.5 | EPE et fonction de régression $E(Y\mid X=x)$ (démonstration) ; $k$-ppv et régression linéaire comme approximations ; perte absolue et médiane ; perte 0-1 et classifieur de Bayes, erreur de Bayes par intégration ; fléau de la dimension : figure 2.6 (côté du sous-cube), éq. (2.24) simulée, figures 2.7 et 2.8 (MSE, biais², variance du 1-ppv), éq. (2.28) simulée, figure 2.9 (1-ppv contre moindres carrés). |
| [`ch02_cours_3_modeles_biais_variance.ipynb`](ch02_cours_3_modeles_biais_variance.ipynb) | 2.6–2.9 | Modèle à erreur additive ; bases de fonctions ; maximum de vraisemblance = moindres carrés sous bruit gaussien (vérifié numériquement) ; nécessité de restreindre la classe de fonctions ; pénalité de rugosité, noyau de Nadaraya-Watson, fonctions à base radiale ; décomposition (2.46)-(2.47) pour les $k$-ppv vérifiée par simulation ; figure 2.11 (erreurs d'apprentissage et de test en fonction de la complexité). |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| [`ch02_exercices_01-05.ipynb`](ch02_exercices_01-05.ipynb) | 2.1 à 2.5 |
| [`ch02_exercices_06-09.ipynb`](ch02_exercices_06-09.ipynb) | 2.6 à 2.9 |

- **Ex. 2.1** - Avec des cibles indicatrices $t_k$, classer à la plus grande composante de $\hat y$ équivaut à choisir la cible la plus proche ($\|t_k-\hat y\|^2 = 1 - 2\hat y_k + \|\hat y\|^2$).
- **Ex. 2.2** - Calcul de la frontière de Bayes de la simulation mixture (égalité des deux densités de mélange ; ligne de niveau et recherche de racines), erreur de Bayes.
- **Ex. 2.3** - Démonstration de l'éq. (2.24) $d(p,N) = (1-2^{-1/N})^{1/p}$, formule bonus de la distance moyenne, vérifications par simulation.
- **Ex. 2.4** - Entrées gaussiennes : les projections $a^Tx_i$ sont $\mathcal N(0,1)$ alors que le point test est à distance $\approx\sqrt p$ ; les points de prédiction sont au bord.
- **Ex. 2.5** - Démonstration de (2.27) (conditionnement, éq. (3.8)) et de (2.28) (astuce de la trace), vérifications numériques.
- **Ex. 2.6** - Avec des $x$ ex-aequo, les moindres carrés se ramènent à des moindres carrés pondérés sur les valeurs distinctes (modèles linéaire et non linéaire).
- **Ex. 2.7** - Estimateurs linéaires en $y$ : poids de la régression linéaire et des $k$-ppv ; biais et variance conditionnels et non conditionnels, relation entre eux ; vérification par simulation.
- **Ex. 2.8** - Chiffres 2 et 3 : erreurs d'apprentissage et de test de la régression linéaire et des $k$-ppv ($k = 1, 3, 5, 7, 15$).
- **Ex. 2.9** - Démonstration (argument de symétrie) de $E[R_{tr}(\hat\beta)] \le E[R_{te}(\hat\beta)]$ et illustration numérique.
