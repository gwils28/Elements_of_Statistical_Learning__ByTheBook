# Chapitre 15 - Forêts aléatoires

Notebooks consacrés au chapitre 15 de *The Elements of Statistical Learning* (forêts
aléatoires, Breiman 2001). Les fonctions réutilisables (arbre CART avec tirage aléatoire
des variables candidates, forêt, erreur OOB, importances, proximités, MDS classique,
décomposition biais/variance, simulateurs) sont dans `esl/ch15_forets.py`, testées par
`tests/test_ch15_forets.py`.

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch15_cours_1_definition_oob.ipynb` | Bagging et variance d'une moyenne de variables corrélées (formule (15.1), simulation) ; algorithme 15.1 et implémentation « maison » à partir d'arbres de régression (critère des moindres carrés, équivalence avec Gini en binaire) ; forêt sur les données spam et comparaison à scikit-learn ; erreur *out-of-bag* (figure 15.4) ; bagging, forêt et boosting sur spam (figure 15.1) avec test de McNemar ; simulation des sphères emboîtées (figure 15.2). |
| `ch15_cours_2_importance_analyse.ipynb` | Réglages par défaut de $m$ et $n_{\min}$ ; importance des variables Gini et par permutation OOB sur spam (figure 15.5) ; matrice de proximité et MDS sur les données mixture et spam (figure 15.6) ; variables de bruit et probabilité hypergéométrique, profondeur des arbres (figure 15.8) ; décomposition de la variance, corrélation entre arbres et compromis biais/variance en fonction de $m$ sur le modèle (15.8) (figures 15.9 et 15.10), comparaison à ridge ; forêt et 3 plus proches voisins sur les données mixture (figure 15.11). |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| `ch15_exercices_01-04.ipynb` | 15.1 – 15.4 |
| `ch15_exercices_05-07.ipynb` | 15.5 – 15.7 |

- **Exercice 15.1** - Démonstration de la formule (15.1) ; pour $\rho<0$, la matrice de covariance équicorrélée n'existe que si $\rho\ge-1/(B-1)$ (valeurs propres), domaine où la formule reste positive.
- **Exercice 15.2** - Un échantillon bootstrap qui évite $z_i$ est un échantillon bootstrap de $\mathbf Z_{-i}$ : l'erreur OOB converge vers la validation croisée leave-one-out (vérification numérique, vitesse en $B^{-1/2}$).
- **Exercice 15.3** - Surface de probabilité de Mease et Wyner (15.11) : deux plateaux $q$ et $1-q$ séparés par un hyperplan diagonal ; erreur de Bayes $q$ ; version réduite de la figure 15.7 (forêt contre boosting avec variables de bruit).
- **Exercice 15.4** - Moyennes bootstrap : $\operatorname{Var}\bar x^*=(2n-1)\sigma^2/n^2$, corrélation $n/(2n-1)$, variance de la moyenne baguée ; le bagging n'aide pas une statistique linéaire.
- **Exercice 15.5** - Corrélation entre deux arbres $\rho(x)$ par les formules de variance et covariance totales (15.12) ; vérification par simulation, covariance conditionnelle nulle.
- **Exercice 15.6** - Spam : erreurs OOB et test en fonction de $m$ (de 1 au bagging, 3 graines).
- **Exercice 15.7** - Importance par permutation en régression linéaire standardisée : $\mathbb E_P[\mathrm{RSS}^*_j-\mathrm{RSS}]=2\hat\beta_j^2$ (preuve, énumération exacte, version sur ensemble test).

## Remarques

- Les données spam ne contenant pas l'indicateur apprentissage/test du livre, une
  séparation aléatoire de même taille (1536 courriels test, graine fixée) est utilisée.
- Les données *California housing* (figure 15.3) ne sont pas fournies : cette figure n'est
  pas reproduite.
- scikit-learn sert de référence et pour les expériences lourdes (courbes jusqu'à 1000
  arbres, boosting, simulations répétées) ; les tailles de simulation sont réduites par
  rapport au livre pour que chaque notebook s'exécute en moins de 3 minutes.
