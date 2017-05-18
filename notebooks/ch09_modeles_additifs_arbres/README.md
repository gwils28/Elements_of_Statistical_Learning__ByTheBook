# Chapitre 9 - Modèles additifs, arbres et méthodes apparentées

Notebooks accompagnant le chapitre 9 de *The Elements of Statistical Learning*
(sections 9.1 à 9.7). Les implémentations « à la main » se trouvent dans le module
`esl/ch09_arbres.py` (testé par `tests/test_ch09_arbres.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch09_cours_1_modeles_additifs.ipynb` | Modèle additif et fonctions de lien ; rappel sur la spline de lissage comme lisseur linéaire (matrice $S_\lambda$, degrés de liberté) ; somme des carrés pénalisée (9.7) et identifiabilité ; algorithme de **backfitting** (alg. 9.1) illustré sur données simulées ; **local scoring** pour le modèle additif logistique (alg. 9.2) ; exemple **spam** : erreur test, table 9.1, figure 9.1, parties linéaire / non linéaire, indicatrices de zéro, pertes asymétriques. |
| `ch09_cours_2_arbres_cart.ipynb` | Partitions binaires récursives (figure 9.2) ; arbres de régression (critère (9.13), recherche rapide des coupures) ; élagage coût-complexité (9.16) et maillon le plus faible ; validation croisée ; arbres de classification : Gini, entropie, erreur (figure 9.3), exemple (400, 400), interprétations de Gini ; prédicteurs qualitatifs, matrice de pertes, instabilité, défaut d'additivité ; exemple **spam** : figure 9.4 (VC), arbre élagué (figure 9.5), table 9.3, sensibilité / spécificité, courbes ROC et AUC (figure 9.6). |
| `ch09_cours_3_prim_mars_hme_manquantes.ipynb` | **PRIM** : épluchage et recollage (alg. 9.3, figures 9.7-9.8), patience, règles sur spam ; **MARS** : fonctions charnières (figures 9.9, 9.11), passes avant / arrière, GCV (9.20), scénarios simulés (table 9.4), spam, lien avec CART ; **mélanges hiérarchiques d'experts** (9.25)-(9.30) et ajustement par EM d'un HME à deux experts ; **données manquantes** : MCAR / MAR (9.31)-(9.32), simulation des stratégies d'imputation, coupures de substitution ; coûts de calcul (section 9.7). |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| `ch09_exercices_01-03.ipynb` | **9.1** - Une spline de lissage (et la régression linéaire locale) préserve la partie linéaire : $Sy=\hat y+Sr$ ; inutilité du recentrage pour un lisseur symétrique. **9.2** - Le backfitting est un Gauss–Seidel par blocs pour le système (9.33) ; convergence et limite pour deux lisseurs symétriques de valeurs propres $<1$. **9.3** - Avec des projections orthogonales, les équations du backfitting équivalent aux équations normales $D^TD\beta=D^Ty$. |
| `ch09_exercices_04-06.ipynb` | **9.4** - Deux fois le même lisseur : résidu limite $(I+S)^{-1}(I-S)y$, RSS croissante, comparaison avec un seul terme. **9.5** - Degrés de liberté d'un arbre de régression : formule naïve $m$, simulation (bien plus que $m$), matrices de lissage approchées (partition gelée, différences finies, perturbations de Ye). **9.6** - Données ozone : modèle additif de la racine cubique de l'ozone, comparaison à l'affichage conditionnel de la figure 6.9, arbre, MARS et PRIM, comparaison par validation croisée. |

## Module `esl/ch09_arbres.py`

- `SplineLisseur`, `MatriceLisseur`, `local_linear_matrix` : lisseurs linéaires ;
- `backfitting`, `GAM` (gaussien ou logistique par local scoring), `logistic_irls` ;
- `ArbreCART` (régression / classification, poids, élagage coût-complexité), `cart_cv`,
  `impurity`, `trace_arbre` ;
- `prim_peel`, `prim_paste`, `prim`, `in_box`, `box_rules` ;
- `MARS` (version simplifiée : nœuds candidats aux quantiles, GCV avec $c=3$) ;
- `roc_curve`, `auc`, `SPAM_NAMES`.
