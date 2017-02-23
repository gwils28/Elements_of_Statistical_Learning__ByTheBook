# Chapitre 8 - Inférence sur les modèles et moyennage

Ce dossier couvre le chapitre 8 de *The Elements of Statistical Learning* (sections 8.1 à 8.9) :
maximum de vraisemblance, bootstrap, approche bayésienne, algorithme EM, MCMC (Gibbs),
bagging, stacking et bumping. Les fonctions réutilisables sont dans `esl/ch08_inference.py`
(testées dans `tests/test_ch08_inference.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch08_cours_1_bootstrap_vraisemblance_bayes.ipynb` | Exemple de lissage par B-splines (fig. 8.1-8.2) ; erreurs standard des moindres carrés ; bootstrap non paramétrique et paramétrique ; maximum de vraisemblance (score, information de Fisher, normalité asymptotique, intervalles du rapport de vraisemblance) ; bootstrap avec choix adaptatif des nœuds ; approche bayésienne, a priori gaussien, tirages a priori et a posteriori (fig. 8.3-8.4) ; bootstrap comme loi a posteriori non informative (loi de Dirichlet, bootstrap bayésien). |
| `ch08_cours_2_em_et_gibbs.ipynb` | Mélange de deux gaussiennes sur les données du tableau 8.1 ; algorithme EM (algorithme 8.1, fig. 8.5-8.6, tableau 8.2) ; dégénérescence de la vraisemblance ; EM en général, décomposition $\ell = Q - R$ et inégalité de Jensen ; EM comme double maximisation (analogue de la fig. 8.7) ; chaînes de Markov, échantillonneur de Gibbs (algorithmes 8.3-8.4, fig. 8.8). |
| `ch08_cours_3_bagging_stacking_bumping.ipynb` | Bagging (vote et moyenne des probabilités) d'arbres sur données simulées corrélées (fig. 8.9-8.10) ; réduction de variance en perte quadratique (éq. 8.52) ; contre-exemple en perte 0-1 ; sagesse des foules (fig. 8.11) ; bagging contre boosting de souches (fig. 8.12) ; moyennage bayésien de modèles ; stacking des meilleurs sous-ensembles sur les données *prostate* ; bumping sur le problème XOR (fig. 8.13), avec variables de bruit. |

Note : les 50 points de l'exemple de lissage du livre n'étant pas publiés, ils sont simulés
avec une allure semblable. Les arbres de classification utilisent scikit-learn (CART relève du
chapitre 9) ; bagging, bumping, souches, AdaBoost et stacking sont codés à la main.

## Exercices

| Notebook | Exercice | Description |
|---|---|---|
| `ch08_exercices_01-04.ipynb` | 8.1 | Inégalité de Jensen : $E_q\log(r/q) \le 0$, maximal en $r = q$ (Kullback-Leibler) ; conséquence $R(\theta', \theta) \le R(\theta, \theta)$. |
| | 8.2 | Multiplicateurs de Lagrange : le maximiseur de $F(\theta', \tilde P)$ est $\Pr(Z^m \mid Z, \theta')$ (étape E). |
| | 8.3 | Justification de l'estimateur (8.50) de densité marginale par Gibbs (théorème ergodique, Rao-Blackwell) ; comparaison avec un estimateur à noyau. |
| | 8.4 | Bagging d'un lisseur B-spline par bootstrap paramétrique : convergence vers l'estimateur original en $1/\sqrt{B}$. |
| `ch08_exercices_05-07.ipynb` | 8.5 | Généralisation à $K$ classes des pertes de la figure 10.4 (codage symétrique) ; coupe 1D, cartes dans le plan $\sum_k f_k = 0$, cohérence de Fisher. |
| | 8.6 | Données de densité osseuse : spline de lissage choisie par validation croisée, bandes à 90 % fréquentistes, bayésiennes (8.28) et bootstrap (100 réplications). |
| | 8.7 | EM comme algorithme MM : $Q(\theta', \theta) + \ell(\theta) - Q(\theta, \theta)$ minore la log-vraisemblance. |
