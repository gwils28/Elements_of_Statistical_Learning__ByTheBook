# Chapitre 12 - Machines à vecteurs de support et discriminants flexibles

Notebooks associés au chapitre 12 de *The Elements of Statistical Learning* (sections 12.1
à 12.7). Les fonctions réutilisables (noyaux, solveurs SMO et points intérieurs, SVM de
classification et de régression, LDA/QDA, FDA, PDA, MDA, simulateurs *mixture* et *waveform*)
sont dans `esl/ch12_svm.py`, testées dans `tests/test_ch12_svm.py` (comparaison à
`sklearn.svm.SVC`, `sklearn.svm.SVR` et à `LinearDiscriminantAnalysis`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch12_cours_1_classifieur_vaste_marge.ipynb` | Hyperplans et distance signée ; problème de la marge maximale (12.3)–(12.4) ; Lagrangien, conditions KKT et dualité introduits pas à pas ; cas séparable ; cas non séparable (variables d'écart, coût $C$, dual (12.13), trois catégories de points) ; algorithme SMO ; exemple « mixture » (figure 12.2) et effet de $C$. |
| `ch12_cours_2_svm_noyaux.ipynb` | Astuce du noyau (12.19)–(12.24), noyaux semi-définis positifs ; SVM polynomial et radial sur « mixture » (figure 12.3) ; SVM comme « perte charnière + pénalité » (12.25), comparaison des pertes (figure 12.4) et minimiseurs de population (tableau 12.1) ; régression logistique à noyau (figure 12.5) ; erreur test selon $C$ et $\gamma$ (figure 12.6) ; chemin de régularisation (figure 12.7) ; SVM pour la régression, pertes $\varepsilon$-insensible et de Huber (figure 12.8) ; régression ridge à noyau (12.45)–(12.49). |
| `ch12_cours_3_fda_pda_mda.ipynb` | *Optimal scoring* et équivalence avec la LDA (12.52)–(12.55) ; FDA (gaussiennes emboîtées, figure 12.9 ; « mixture », figure 12.13) ; MDA et algorithme EM (12.59)–(12.63) ; PDA et pénalité de rugosité (12.57)–(12.58) ; données *waveform* (figure 12.14), erreur de Bayes, tableau 12.4 simplifié (10 simulations) et vues canoniques de la MDA (figure 12.15). |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| `ch12_exercices_01-05.ipynb` | 12.1 à 12.5 |
| `ch12_exercices_06-11.ipynb` | 12.6 à 12.11 |

* **12.1** - Équivalence entre le critère « charnière + pénalité » (12.25) et le problème quadratique (12.8), avec $\lambda = 1/C$ ; vérification par optimisation directe.
* **12.2** - La forme à noyau (12.29) est (12.25) pour $K(x,x') = \langle h(x),h(x')\rangle$ ; vérification avec le noyau radial.
* **12.3** - Régression à noyau sans pénaliser la constante : représentation $\hat\beta_0 + \sum_i\hat a_iK(x,x_i)$ avec $\sum_i\hat a_i = 0$, système linéaire bordé pour la perte quadratique.
* **12.4** - Dans l'espace canonique complet de la LDA, les différences de distances euclidiennes égalent les différences de distances de Mahalanobis ; interprétation en rang réduit.
* **12.5** - Phonèmes : prototypes lisses (base de splines naturelles orthonormée) et $k$-moyennes en dimension réduite ; découpage par locuteurs ; comparaison $K\in\{1,3,5,7\}$, $J\in\{5,10,15\}$.
* **12.6** - *Optimal scoring* sous forme matricielle, normalisations, problème aux valeurs propres généralisé, solution triviale constante.
* **12.7** - *Optimal scoring* pénalisé : lisseur $S_\lambda = H(H^TH+\lambda\Omega)^{-1}H^T$ et lien avec les directions de la PDA.
* **12.8** - Les coefficients de l'*optimal scoring* sont proportionnels aux directions discriminantes de la LDA.
* **12.9** - La LDA sur les valeurs ajustées d'une régression sur indicatrices est identique à la LDA d'origine.
* **12.10** - LDA régularisée à noyau : la direction discriminante ne dépend que de la matrice de Gram ; application avec le noyau radial.
* **12.11** - Modèle MDA2 à centres partagés : densités de classes, algorithme EM, identité avec la MDA pour une initialisation par classe.

## Remarques

* Les méthodes FDA/MARS, FDA/BRUTO et CART du livre ne sont pas recodées ici : la FDA est
  illustrée avec des régressions polynomiales et sur base radiale (ridge), et la MDA
  « pénalisée 4 ddl » du tableau 12.4 est remplacée par une MDA à covariance régularisée par
  une matrice de rugosité.
* L'erreur test sur la simulation « mixture » est estimée sur un grand échantillon simulé à
  partir des 20 centres du modèle générateur ; l'erreur de Bayes sur *waveform* est calculée
  par intégration numérique.
