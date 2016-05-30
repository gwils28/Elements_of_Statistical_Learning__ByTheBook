# Chapitre 5 - Expansions en bases et régularisation

Notebooks consacrés au chapitre 5 de *The Elements of Statistical Learning*
(*Basis Expansions and Regularization*). Les fonctions réutilisables sont dans
`esl/ch05_splines.py` (testées par `tests/test_ch05_splines.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch05_cours_1_polynomes_par_morceaux_et_splines.ipynb` | Expansions en bases (§5.1) ; polynômes par morceaux et ordres de continuité (figures 5.1, 5.2) ; splines et base des puissances tronquées ; B-splines et récurrence de Cox-de Boor (figure 5.20), conditionnement ; splines cubiques naturelles et variance ponctuelle (figure 5.3) ; phonèmes « aa »/« ao » : régression logistique brute et à coefficients lissés (figure 5.5 et tableau d'erreurs) ; filtrage d'entrées (§5.3). |
| `ch05_cours_2_splines_de_lissage.ipynb` | Critère des splines de lissage (§5.4) ; forme matricielle et calcul exact de la pénalité en B-splines ; données *bone* (figure 5.6, λ ≈ 0,00022 retrouvé) ; matrice de lissage vs projection, degrés de liberté effectifs ; forme de Reinsch et décomposition spectrale (figure 5.7) ; noyaux équivalents (figure 5.8) ; compromis biais-variance, EPE, CV et GCV (figure 5.9) ; régression logistique non paramétrique par IRLS sur *SAheart* (§5.6). |
| `ch05_cours_3_noyaux_rkhs_et_ondelettes.ipynb` | Produits tensoriels et splines de plaque mince (§5.7) ; noyaux, théorème de Mercer, RKHS, propriété reproduisante et théorème du représentant présentés pas à pas (§5.8) ; noyau gaussien sur les données *mixture* (figures 5.13-5.15) ; ridge à noyau vs spline de lissage ; ondelettes de Haar : multirésolution (figure 5.16), transformée rapide, seuillage doux et seuil universel, comparaison au lissage par spline (figures 5.17 et 5.19, sur signaux simulés). |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| `ch05_exercices_01-05.ipynb` | 5.1 à 5.5 |
| `ch05_exercices_06-10.ipynb` | 5.6 à 5.10 |
| `ch05_exercices_11-15.ipynb` | 5.11 à 5.15 |
| `ch05_exercices_16-19.ipynb` | 5.16 à 5.19 |

- **5.1** - La base des puissances tronquées (5.3) est une base des splines cubiques à deux nœuds.
- **5.2** - Propriétés des B-splines par récurrence : support, positivité, partition de l'unité, structure polynomiale, densité d'une somme d'uniformes.
- **5.3** - Programme reproduisant la figure 5.3 (variances ponctuelles), validé par Monte-Carlo.
- **5.4** - Contraintes linéaires des splines naturelles et dérivation de la base (5.4)-(5.5).
- **5.5** - Phonèmes (5 classes) : QDA sur entrées filtrées par splines naturelles, choix des nœuds par CV à 10 plis (plis par locuteur).
- **5.6** - Splines périodiques : base des puissances tronquées en $x \bmod T$ et trois contraintes de recollement.
- **5.7** - La spline naturelle interpolante minimise $\int f''^2$ ; la solution de (5.9) est une spline naturelle.
- **5.8** - Schéma en $N+2$ B-splines sur les $N-2$ nœuds intérieurs ; matrices bandes et résolution en $O(N)$.
- **5.9** - Forme de Reinsch $\mathbf S_\lambda = (\mathbf I + \lambda\mathbf K)^{-1}$ (et forme de Green-Silverman de $\mathbf K$).
- **5.10** - Biais et variance ponctuels d'une spline de lissage ; version de la figure 5.9 avec moyenne et quantiles.
- **5.11** - Le noyau de $\mathbf K$ est engendré par les fonctions affines.
- **5.12** - Spline de lissage pondérée ; traitement des ex æquo par moyennes pondérées.
- **5.13** - Ajouter un point ajusté ne change rien ; dérivation de la formule de CV (5.27).
- **5.14** - Contraintes $\sum\alpha_j = 0$, $\sum\alpha_jx_j = 0$ des splines de plaque mince ; ajustement en 2D.
- **5.15** - Propriété reproduisante, $J(g) = \alpha^T\mathbf K\alpha$, théorème du représentant.
- **5.16** - Ridge primale vs ridge à noyau : $\hat{\mathbf f} = \mathbf K(\mathbf K + \lambda\mathbf I)^{-1}\mathbf y$ ; cas $M < N$.
- **5.17** - Fonctions propres d'un noyau estimées par l'extension de Nyström (comparées au cas gaussien exact).
- **5.18** - $p$ moments nuls ⇒ les polynômes de degré $< p$ sont dans $V_0$ (illustré avec Haar et Daubechies D4).
- **5.19** - Transformée de Haar par algorithme pyramidal en $O(N)$.

## Remarques

- Le signal RMN et les données d'ozone de Los Angeles (figures 5.7, 5.17, 5.19) ne sont
  pas disponibles : les figures correspondantes utilisent les données *bone* ou des
  signaux simulés de même nature.
- Seules les ondelettes de Haar sont implémentées (les symmlets du livre sont évoquées
  et l'ondelette de Daubechies D4 est utilisée à l'exercice 5.18).
