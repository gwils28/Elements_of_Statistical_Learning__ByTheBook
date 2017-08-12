# Chapitre 10 - Boosting et arbres additifs

Notebooks consacrés au chapitre 10 de *The Elements of Statistical Learning* (boosting,
modèles additifs pas à pas, gradient boosting, interprétation des arbres boostés).
Les fonctions réutilisables se trouvent dans le module `esl/ch10_boosting.py` (arbre de
régression pondéré à $J$ feuilles, AdaBoost.M1, gradient boosting pour les pertes
quadratique, absolue, de Huber et la déviance, gradient boosting à $K$ classes, importance
relative et dépendance partielle) ; les tests sont dans `tests/test_ch10_boosting.py`.

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch10_cours_1_adaboost_perte_exponentielle.ipynb` | AdaBoost.M1 (algorithme 10.1) ; simulation des sphères emboîtées (10.2) et figure 10.2 ; repondération visualisée en dimension 2 ; modélisation additive pas à pas (algorithme 10.2) ; équivalence AdaBoost / perte exponentielle (éq. 10.9–10.15) ; figure 10.3 ; minimiseur en population (10.16)–(10.17) et lien avec la déviance binomiale (10.18). |
| `ch10_cours_2_pertes_gradient_boosting.ipynb` | Pertes de classification et marge (figure 10.4), influence des marges négatives, déviance multinomiale ; pertes de régression robustes (figure 10.5) et expérience avec valeurs aberrantes ; boosting d'arbres (10.25)–(10.30) ; gradient boosting (algorithme 10.3, tableau 10.2) ; taille des arbres et ANOVA (figures 10.9 et 10.10) ; rétrécissement (figure 10.11) et sous-échantillonnage (figure 10.12). |
| `ch10_cours_3_interpretation_spam.ipynb` | Méthodes « clé en main » pour la fouille de données ; gradient boosting sur les données spam ; importance relative (10.42)–(10.43, figure 10.6) ; dépendance partielle (10.47)–(10.49), calcul par parcours d'arbre, figures 10.7 et 10.8 ; étude complète sur des données immobilières **simulées** sur le modèle de *California Housing* (non disponibles) : perte de Huber, erreur absolue moyenne, importances, dépendances partielles comparées aux vraies, interactions et effet de la localisation. |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| `ch10_exercices_01-06.ipynb` | 10.1 à 10.6 |
| `ch10_exercices_07-12.ipynb` | 10.7 à 10.12 |

- **10.1** - Dérivation du coefficient $\beta_m = \frac12\log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$ d'AdaBoost (10.12), avec vérification numérique.
- **10.2** - Le minimiseur en population de la perte exponentielle est la moitié du log-rapport de cotes (10.16).
- **10.3** - La dépendance partielle retrouve les effets additifs et multiplicatifs, pas l'espérance conditionnelle (exemple gaussien).
- **10.4** - AdaBoost avec des arbres : programme, figure 10.2 (apprentissage et test), comportement sur 3000 itérations, classes qui se chevauchent (erreur de Bayes 14 %).
- **10.5** - Perte exponentielle multiclasse : minimiseur sous contrainte par Lagrange, algorithme SAMME (vérifié contre scikit-learn sur *waveform*).
- **10.6** - Test de McNemar : erreur standard de 0,6 %, statistique $(33-18)^2/51$ et $p \approx 0{,}036$, application à nos modèles sur spam.
- **10.7** - Valeurs optimales des feuilles pour la perte exponentielle (10.32) et boosting de souches à valeurs réelles.
- **10.8** - Log-vraisemblance multinomiale, pas de Newton diagonal (10.57) et mise à jour symétrique de somme nulle (10.58).
- **10.9** - Dérivation de l'algorithme 10.4 (gradient boosting à $K$ classes) et mise en œuvre sur *waveform*.
- **10.10** - Pour $K=2$, un seul arbre par itération suffit ($f_2 = -f_1$), vérifié numériquement.
- **10.11** - Calcul efficace de la dépendance partielle d'un modèle d'arbres : méthode exacte et parcours pondéré de Friedman, comparés à la force brute.
- **10.12** - Avec $f = X_1$ et $(X_1,X_2)$ gaussien de corrélation $\rho$, $E(f\mid X_2) = \rho X_2$ alors que la dépendance partielle est plate.

## Remarques

- La séparation apprentissage/test des données spam du livre n'étant pas disponible, une
  séparation aléatoire (1536 messages de test) est utilisée ; les taux d'erreur obtenus
  (environ 5 %) sont proches de ceux du livre (4,5 %).
- Sur la simulation (10.2), notre réalisation donne une erreur de test d'AdaBoost un peu plus
  élevée que celle du livre (environ 12 % contre 5,8 % après 400 itérations) ; l'implémentation
  coïncide pourtant exactement avec `AdaBoostClassifier` de scikit-learn.
