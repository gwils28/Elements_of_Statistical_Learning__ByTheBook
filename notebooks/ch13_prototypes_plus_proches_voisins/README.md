# Chapitre 13 - Méthodes à prototypes et plus proches voisins

Notebooks associés au chapitre 13 de *The Elements of Statistical Learning*
(sources : `.nbsrc/ch13/`, fonctions réutilisables : `esl/ch13_prototypes.py`,
tests : `tests/test_ch13_prototypes.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch13_cours_1_prototypes.ipynb` | Règle du prototype le plus proche (cellules de Voronoï) ; K-means (algorithme de Lloyd, décroissance du critère) et classifieur K-means ; LVQ1 ; mélanges gaussiens à covariance commune ajustés par EM ; figures 13.1 (simulation à trois classes), 13.2 (K-means contre mélange gaussien sur les données *mixture*) et panneaux K-means/LVQ de la figure 13.5 (problèmes « facile » et « difficile »). |
| `ch13_cours_2_plus_proches_voisins.ipynb` | Règle des k-ppv, biais et variance ; figures 13.3 et 13.4 (erreurs apprentissage / test / VC 10 blocs en fonction de k, 7-ppv) ; théorème de Cover et Hart et vérification numérique ; chiffres manuscrits (zip) : rotations, droite tangente, 7 vecteurs tangents, 1-ppv euclidien contre distance tangente (tableau 13.1) ; fléau de la dimension (figure 13.12) ; voisinages étirés (figure 13.13) ; métrique DANN (figures 13.14 et 13.15) ; réduction de dimension globale (éq. (13.10)-(13.11)) ; coût calculatoire, multi-édition et condensation de Hart. |

## Notebooks d'exercices

| Exercice | Notebook | Description |
|---|---|---|
| 13.1 | `ch13_exercices_01-04.ipynb` | K-means et EM pour un mélange à covariance $\sigma I$ : analogie détaillée et coïncidence quand $\sigma \to 0$ (vérifiée numériquement). |
| 13.2 | `ch13_exercices_01-04.ipynb` | Démonstration de la formule (13.7) du rayon médian du 1-voisinage, comparée à une simulation. |
| 13.3 | `ch13_exercices_01-04.ipynb` | Théorème de Cover et Hart : inégalité (13.5) (Cauchy–Schwarz), convergence et borne (13.12) (Jensen), simulation à 3 classes. |
| 13.4 | `ch13_exercices_01-04.ipynb` | Décomposition QR d'une transformation affine (rotation, homothéties, cisaillement), dérivées des images transformées, mise en œuvre par lisseur à noyau sur images 16×16. |
| 13.5 | `ch13_exercices_05-08.ipynb` | Meilleure approximation de rang $L$ d'une famille de matrices (décomposition autour de la moyenne + Eckart–Young via Ky Fan). |
| 13.6 | `ch13_exercices_05-08.ipynb` | Moyenne de formes invariante affine : deux critères, solutions par vecteurs propres, comparaison via la SVD ; illustration avec de grandes lettres aberrantes. |
| 13.7 | `ch13_exercices_05-08.ipynb` | k-ppv sur les problèmes « facile » et « difficile » : figure 13.5 (gauche), validation croisée à 5 blocs contre pénalité de type AIC. |
| 13.8 | `ch13_exercices_05-08.ipynb` | Classes tournées aléatoirement : k-ppv, « hints », métrique invariante, distance tangente ; l'énoncé donne deux classes de même loi (erreur de Bayes 50 %), d'où une variante informative. |
