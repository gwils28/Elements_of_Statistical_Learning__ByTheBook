# Chapitre 3 - Méthodes linéaires de régression

Notebooks associés au chapitre 3 de *The Elements of Statistical Learning*
(*Linear Methods for Regression*). Les exemples utilisent les données du cancer de la
prostate (67 observations d'apprentissage, 30 de test, 8 prédicteurs standardisés).

Les fonctions réutilisables sont dans `esl/ch03_regression.py` (moindres carrés par QR,
Gram–Schmidt, meilleur sous-ensemble, pas à pas, stagewise, ridge, lasso par descente de
coordonnées, LAR et lasso-LAR, PCR, PLS, validation croisée) et testées dans
`tests/test_ch03_regression.py` (comparaisons à scikit-learn et à des formules fermées).

## Cours

| Notebook | Contenu |
|---|---|
| [`ch03_cours_1_moindres_carres.ipynb`](ch03_cours_1_moindres_carres.ipynb) | Modèle linéaire, équations normales, matrice chapeau et géométrie de la projection ; variance de $\hat\beta$, loi gaussienne, scores $Z$ et test $F$, intervalles et régions de confiance (figure 3.3) ; prostate : tableaux 3.1 et 3.2, test $F$ (3.16), erreur de test ; théorème de Gauss–Markov et compromis biais–variance ; orthogonalisations successives (algorithme 3.1, figure 3.4), décomposition QR et stabilité numérique ; plusieurs réponses. |
| [`ch03_cours_2_selection_ridge_lasso.ipynb`](ch03_cours_2_selection_ridge_lasso.ipynb) | Meilleur sous-ensemble (figure 3.5), validation croisée à 10 plis et règle « un écart-type », pas à pas ascendant/descendant, régression stagewise et FS$_\varepsilon$, simulation de la figure 3.6 (réduite) ; ridge : solution explicite, SVD, composantes principales (figure 3.9), degrés de liberté effectifs, chemin (figure 3.8) ; lasso : seuillage doux, descente de coordonnées, chemin (figure 3.10), géométrie des contraintes (figures 3.11, 3.12) ; panneaux correspondants de la figure 3.7. |
| [`ch03_cours_3_lar_pcr_pls.ipynb`](ch03_cours_3_lar_pcr_pls.ipynb) | LAR (algorithme 3.2) : directions équiangulaires, corrélations à égalité (figure 3.14), modification lasso (3.2a, figure 3.15), degrés de liberté (3.60) ; PCR et rétrécissement des composantes (figure 3.17) ; PLS (algorithme 3.3) et critère (3.64) ; comparaison complète sur la prostate : figure 3.7, tableau 3.3, variabilité du choix par validation croisée ; problème jouet à deux variables (figure 3.18). |

## Exercices 3.1 à 3.15

| Notebook | Exercice | Description |
|---|---|---|
| [`ch03_exercices_01-05.ipynb`](ch03_exercices_01-05.ipynb) | 3.1 | La statistique $F$ pour le retrait d'un coefficient égale le carré du score $Z$. |
| | 3.2 | Bandes de confiance d'une régression cubique : ponctuelle contre déduite de l'ellipsoïde (3.15) ; simulation des couvertures ponctuelle et simultanée. |
| | 3.3 | Théorème de Gauss–Markov, version scalaire et version matricielle ($\hat{\mathbf{V}} \preceq \tilde{\mathbf{V}}$). |
| | 3.4 | Coefficients des moindres carrés en une seule passe de Gram–Schmidt ; lien avec $\mathbf{R}\hat\beta = \mathbf{Q}^T\mathbf{y}$. |
| | 3.5 | Ridge et lasso avec entrées centrées : correspondance des paramètres, $\hat\beta_0^c = \bar y$. |
| [`ch03_exercices_06-10.ipynb`](ch03_exercices_06-10.ipynb) | 3.6 | Ridge comme moyenne et mode a posteriori sous un a priori gaussien, $\lambda = \sigma^2/\tau$. |
| | 3.7 | Opposé du log a posteriori avec ordonnée à l'origine (a priori plat) : critère ridge, $\lambda = \sigma^2/\tau^2$. |
| | 3.8 | QR de la matrice non centrée et SVD de la matrice centrée : même sous-espace ; égalité aux signes près ssi variables non corrélées rangées par variance décroissante. |
| | 3.9 | Sélection pas à pas ascendante efficace par mise à jour de la QR. |
| | 3.10 | Sélection pas à pas descendante : retirer la variable de plus petit $\lvert z_j\rvert$. |
| [`ch03_exercices_11-15.ipynb`](ch03_exercices_11-15.ipynb) | 3.11 | Régression multivariée pondérée par $\Sigma^{-1}$ : même solution que les régressions séparées ; moindres carrés généralisés si $\Sigma_i$ varie. |
| | 3.12 | Ridge comme moindres carrés sur des données augmentées par $\sqrt\lambda\,\mathbf{I}$. |
| | 3.13 | Coefficients de la PCR (3.62) et $\hat\beta^{\mathrm{pcr}}(p) = \hat\beta^{\mathrm{ls}}$. |
| | 3.14 | Entrées orthogonales : PLS atteint les moindres carrés en une étape. |
| | 3.15 | Critère (3.64) des directions PLS (preuve par espaces de Krylov) ; compromis entre régression et composantes principales. |

## Exercices 3.16 à 3.30

Fonctions associées : `esl/ch03_regression_complements.py` (tests : `tests/test_ch03_regression_complements.py`).

| Notebook | Exercices |
|---|---|
| [`ch03_exercices_16-20.ipynb`](ch03_exercices_16-20.ipynb) | 3.16 à 3.20 |
| [`ch03_exercices_21-25.ipynb`](ch03_exercices_21-25.ipynb) | 3.21 à 3.25 |
| [`ch03_exercices_26-30.ipynb`](ch03_exercices_26-30.ipynb) | 3.26 à 3.30 |

- **3.16** - Formules du tableau 3.4 en plan orthonormé : seuillage dur (meilleur sous-ensemble), contraction $1/(1+\lambda)$ (ridge), seuillage doux (lasso) ; vérification numérique.
- **3.17** - Le tableau 3.3 refait sur *spam* (moindres carrés, sélection ascendante à la place du meilleur sous-ensemble, ridge, lasso, PCR, PLS) : validation croisée à 10 plis, règle « un écart-type », erreur test.
- **3.18** - Les PLS sont le gradient conjugué appliqué aux équations normales : optimalité sur les espaces de Krylov, preuve et vérification.
- **3.19** - La norme des coefficients croît quand on régularise moins : oui pour la ridge et les PLS (preuves), seulement en norme $\ell_1$ pour le lasso (contre-exemple en norme $\ell_2$).
- **3.20** - Corrélations canoniques par SVD généralisée : blanchiment, éq. (3.86)–(3.87), toute la suite des variables canoniques.
- **3.21** - Régression de rang réduit : blanchiment, théorème d'Eckart–Young (prouvé), identification avec la CCA $\hat{\mathbf B}\mathbf U_m\mathbf U_m^-$ ; contrôle par optimisation numérique.
- **3.22** - La régression de rang réduit est inchangée avec la covariance résiduelle : mêmes vecteurs propres généralisés.
- **3.23** - Corrélations liées et décroissantes en avançant vers les moindres carrés, formule de $\lambda(\alpha)$ et application au LAR (analogue de la figure 3.14).
- **3.24** - La direction LAR est équiangulaire avec les prédicteurs actifs.
- **3.25** - « Look-ahead » du LAR : formule du pas $\alpha$ et de la variable entrante, comparée à une recherche sur grille.
- **3.26** - Sélection ascendante (maximise $|\mathrm{Corr}(x_{j\cdot\mathcal A}, r)|$) contre LAR : contre-exemple explicite et fréquence des ordres d'entrée différents.
- **3.27** - Conditions KKT du lasso (introduites pas à pas), les trois scénarios, lien avec la corrélation, linéarité par morceaux du chemin (prostate).
- **3.28** - Lasso avec une variable dupliquée : segment de solutions $\{(\theta a, (1-\theta)a)\}$.
- **3.29** - Ridge avec $m$ copies : coefficients égaux $x^Ty/(m\,x^Tx+\lambda)$, argument de symétrie et de stricte convexité.
- **3.30** - L'elastic net comme lasso sur données augmentées $[\mathbf X; \sqrt{\lambda\alpha}\,\mathbf I]$, $[\mathbf y; \mathbf 0]$ ; effet de groupe.
