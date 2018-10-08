# Chapitre 14 - Apprentissage non supervisé

Notebooks du chapitre 14 de *The Elements of Statistical Learning* (pages 485 à 585 du livre) :
règles d'association, classification (analyse de groupements), cartes auto-organisatrices,
composantes principales et leurs extensions, analyse en composantes indépendantes,
positionnement multidimensionnel.

Les fonctions réutilisables sont dans le module `esl/ch14_non_supervise.py` (testé par
`tests/test_ch14_non_supervise.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| [`ch14_cours_1_regles_association.ipynb`](ch14_cours_1_regles_association.ipynb) | Supervisé / non supervisé (14.1) ; ensembles d'items, support, anti-monotonie et algorithme **Apriori** ; règles d'association (confiance, lift) sur les données démographiques `marketing` (6876 × 50 indicatrices, figure 14.2) ; estimation de densité par classification « données contre référence » (14.10)-(14.14), figure 14.3 avec régression logistique sur splines naturelles ; choix de la densité de référence (uniforme, produit des marginales) ; règles d'association généralisées et **PRIM**. |
| [`ch14_cours_2_classification.ipynb`](ch14_cours_2_classification.ipynb) | Matrices de proximité, dissimilarités par type de variable, corrélation, pondération et piège de la standardisation (figure 14.5) ; critère intra/inter-groupes et nombre de partitions ; **K-means** (figures 14.4, 14.6), mélanges gaussiens comme K-means « doux » (figure 14.7) ; K-means sur les données NCI (figure 14.8, tableau 14.2) ; quantification vectorielle d'une image ; **K-médoïdes** sur les dissimilarités entre pays (figure 14.10) ; **statistique Gap** (figure 14.11) ; **classification hiérarchique** : liens simple, complet, moyen, dendrogrammes NCI (figures 14.12, 14.13), dissimilarités cophénétiques et ultramétrique, invariance par transformation monotone, double réordonnancement (figure 14.14), classification descendante. |
| [`ch14_cours_3_reduction_dimension.ipynb`](ch14_cours_3_reduction_dimension.ipynb) | **Cartes auto-organisatrices** en ligne et batch sur la demi-sphère (figures 14.15 à 14.18) ; **ACP** (14.49)-(14.54), demi-sphère (figure 14.21), chiffres « 3 » manuscrits (figures 14.22 à 14.24) ; **Procrustes** et moyenne de formes (figures 14.25, 14.26, sur des « S » synthétiques) ; **courbes principales** (figure 14.27) ; **clustering spectral** et laplacien de graphe (figure 14.29) ; **ACP à noyau** (figure 14.30) ; présentation de l'ACP parcimonieuse et de la **NMF** ; analyse factorielle et **ACI** (FastICA, figures 14.37, 14.38) ; **MDS** classique et par moindres carrés (figure 14.43). |

## Exercices

| Notebook | Exercices |
|---|---|
| [`ch14_exercices_01-06.ipynb`](ch14_exercices_01-06.ipynb) | 14.1 à 14.6 |
| [`ch14_exercices_07-12.ipynb`](ch14_exercices_07-12.ipynb) | 14.7 à 14.12 |

### Exercices 14.1 à 14.12

- **14.1** - Une distance euclidienne pondérée est la distance usuelle sur des variables remises à l'échelle par $\sqrt{w_l/\sum w}$ ; conséquence pour les K-means.
- **14.2** - Log-vraisemblance et algorithme EM pour un mélange gaussien isotrope ; quand $\sigma \to 0$, EM devient les K-means (vérification numérique).
- **14.3** - Avec une référence « colonnes permutées », données et référence ont les mêmes marges : le premier pas de CART ou PRIM est aveugle ; remèdes (anticipation sur des paires de variables, etc.), illustrés sur `marketing`.
- **14.4** - Arbre de classification données contre référence permutée sur les données démographiques ; description des feuilles, comparaison avec PRIM (livre) et les K-means.
- **14.5** - SOM (réglages du texte) et K-means sur les trois groupes proches d'une sphère ; erreur de reconstruction, pureté des nœuds, effet du niveau de bruit.
- **14.6** - K-means et SOM sur les 64 échantillons NCI pour $K = 2, 5, 10, 20$ ; quand le voisinage rétrécit, la SOM rejoint les K-means.
- **14.7** - Dérivation de $\hat\mu = \bar x$, $\hat\lambda_i = \mathbf{V}_q^T(x_i - \bar x)$ ; famille des solutions $\bar x + \mathbf{V}_q c$ (avec le lemme de Ky Fan sur les traces).
- **14.8** - Solution du problème de Procrustes $\hat{\mathbf{R}} = \mathbf{U}\mathbf{V}^T$ et du problème avec échelle $\hat\beta = \operatorname{tr}(\mathbf{D})/\|\mathbf{X}_1\|_F^2$ ; comparaison à SciPy.
- **14.9** - Algorithme de moyenne de Procrustes avec facteurs d'échelle (contrainte de normalisation) appliqué à trois « S ».
- **14.10** - Moyenne invariante affine : $\mathbf{M}$ = premiers vecteurs propres de la moyenne des projecteurs ; comparaison avec l'exercice 14.9.
- **14.11** - Échelle classique : la solution est $\mathbf{E}_k\mathbf{D}_k$ (Eckart-Young) ; lien avec l'ACP.
- **14.12** - ACP parcimonieuse : à $\Theta$ fixé, $K$ elastic net ; à $\mathbf{V}$ fixé, Procrustes réduit $\Theta = \mathbf{U}\mathbf{Q}^T$ ; vérifications numériques.

### Exercices 14.13 à 14.25

| Notebook | Exercices | Notions |
|---|---|---|
| [`ch14_exercices_13-17.ipynb`](ch14_exercices_13-17.ipynb) | 14.13 – 14.17 | courbes principales, cartes auto-organisatrices, analyse factorielle (maximum de vraisemblance, EM, cas de Heywood), ACP à noyau, espace de Hilbert à noyau reproduisant |
| [`ch14_exercices_18-21.ipynb`](ch14_exercices_18-21.ipynb) | 14.18 – 14.21 | ICA directe (densités gaussiennes inclinées pénalisées), invariance par rotation, FastICA comme méthode de Newton, laplacien de graphe et composantes connexes |
| [`ch14_exercices_22-25.ipynb`](ch14_exercices_22-25.ipynb) | 14.22 – 14.25 | PageRank et chaînes de Markov, algorithmes MM, NMF de Poisson, ajustement proportionnel itératif, NMF contre ACP sur les chiffres manuscrits |

Fonctions réutilisables : `esl/ch14_non_supervise_complements.py` (tests : `tests/test_ch14_non_supervise_complements.py`).

- **Ex. 14.13** - Hélice bruitée dans $\mathbb{R}^3$ : courbe principale (algorithme alterné lissage/projection) dont les fonctions coordonnées retrouvent $\cos s$, $\sin s$, $s$ ; SOM en chaîne et en grille, comparées aux $K$-moyennes.
- **Ex. 14.14** - $\Delta\Sigma\Delta = (\Delta A)(\Delta A)^T + \Delta D_\varepsilon\Delta$ : le modèle factoriel de la matrice de corrélation s'obtient en redimensionnant les lignes de $A$ ; équivariance de l'estimateur, contrairement à l'ACP.
- **Ex. 14.15** - Exemple de Hinton : la 1re composante principale s'aligne sur $X_3$ (variance maximale), le 1er facteur sur $X_1 + X_2$ (seule corrélation) ; convergence lente de l'EM dans un cas de Heywood.
- **Ex. 14.16** - ACP à noyau : nombre de composantes $=$ rang de $\tilde K$, $z_m = \tilde K\alpha_m$ avec $\alpha_m = u_m/d_m$, projection de nouvelles observations ; vérification contre scikit-learn.
- **Ex. 14.17** - Maximisation de la variance empirique dans $\mathcal H_K$ sous $\|g\|_{\mathcal H_K}=1$ : solution $\hat c = u_1/d_1$ et composantes suivantes par orthogonalité dans $\mathcal H_K$.
- **Ex. 14.18** - Conditions de stationnarité sur la partie quadratique non pénalisée : $\hat f = \varphi e^{\hat g}$ est une densité de moyenne 0 et de variance 1 ; avec une pénalité d'ordre 2, rendre le terme $s^2$ libre ; vérification par une version discrétisée (Newton).
- **Ex. 14.19** - $\sum_j(a_j^Tx)^2 = \|x\|^2$ pour $A$ orthogonale : le terme gaussien de la log-vraisemblance ne dépend pas de $A$.
- **Ex. 14.20** - Lagrangien, hessienne approchée par un multiple de l'identité, et étape de Newton renormalisée $=$ point fixe (14.96) de FastICA ; convergence très rapide sur un mélange de sources uniformes.
- **Ex. 14.21** - $f^TLf = \frac12\sum w_{ii'}(f_i - f_{i'})^2$ : le noyau du laplacien est engendré par les indicatrices des $m$ composantes connexes ; illustration sur un graphe des plus proches voisins.
- **Ex. 14.22** - $\sum_ip_i = N$ découle de (14.108) ; méthode de la puissance appliquée au réseau de la figure 14.47, vérification sur la figure 14.46 et simulation du surfeur aléatoire.
- **Ex. 14.23** - Minorante de la log-vraisemblance de Poisson par l'inégalité de Jensen ; sa maximisation par blocs redonne les mises à jour multiplicatives (14.74) ; vérification de la minoration et de la monotonie.
- **Ex. 14.24** - En rang 1, la NMF est l'ajustement proportionnel itératif du modèle d'indépendance, au point fixe dès la 1re itération : $\hat x_{ij} = x_{i+}x_{+j}/x_{++}$ (tableau revenu × études des données *marketing*).
- **Ex. 14.25** - NMF à 25 facteurs contre ACP à 24 composantes plus la moyenne sur les « 2 » de la base zip : images de base localisées et poids creux pour la NMF, directions globales de signes mêlés pour l'ACP.

## Remarques

- Les données `marketing`, `nci`, `countries` et les chiffres manuscrits (`zip`) sont ceux du livre.
  La photographie de R. A. Fisher (quantification vectorielle) et les signatures « S » (Procrustes)
  n'étant pas disponibles, elles sont remplacées par une image et des courbes synthétiques.
- scikit-learn n'est utilisé que pour les arbres de classification (exercices 14.3, 14.4) ; SciPy
  sert au tracé des dendrogrammes et à la classification hiérarchique des 6830 gènes.
