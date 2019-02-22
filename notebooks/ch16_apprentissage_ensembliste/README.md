# Chapitre 16 - Apprentissage ensembliste

Notebooks couvrant le chapitre 16 de *The Elements of Statistical Learning* (2e éd.).
Les fonctions réutilisables sont dans `esl/ch16_ensembles.py` (tests :
`tests/test_ch16_ensembles.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch16_cours_1_boosting_chemins_regularisation.ipynb` | Ensembles et codes correcteurs (ECOC) ; régression pénalisée sur un dictionnaire d'arbres (16.1)–(16.4) ; algorithme 16.1 *forward stagewise* ; figure 16.1 (lasso et stagewise sur les données prostate) ; équivalence exacte dans le cas orthonormé ; lien avec le boosting à rétrécissement ; principe du « pari sur la parcimonie » (version réduite de la figure 16.2) ; chemins sur données corrélées par blocs (figures 16.3 et 16.4) ; lasso monotone et longueur d'arc $L_1$ ; marge $L_1$ d'AdaBoost sur les données mixture (figure 16.5). |
| `ch16_cours_2_ensembles_post_traitement.ipynb` | Post-traitement lasso d'un ensemble (16.8)–(16.9) ; forêt aléatoire post-traitée sur les données spam (figure 16.6) ; importance sampling, critère $Q(\gamma)$ et largeur $\sigma$ (16.10)–(16.12) ; algorithme ISLE 16.2 ; simulation de régression (16.13) comparant GBM, GBM stochastique, forêt et leurs versions post-traitées (figure 16.8) ; ensembles de règles (16.14)–(16.15) et RuleFit simplifié, règles seules contre règles + termes linéaires (figure 16.10). |

## Exercices

Notebook `ch16_exercices_01-04.ipynb` :

- **Ex. 16.1** - Génération exacte des données gaussiennes corrélées par blocs de la section 16.2.3 (facteur commun par bloc, un coefficient non nul par bloc, réglage du bruit pour NSR = 0,72), vérifiée empiriquement.
- **Ex. 16.2** - Longueur d'arc $L_1$ d'un chemin de coefficients : $\Lambda(t)\ge\|\alpha(t)\|_1$, avec égalité si et seulement si le chemin est monotone ; illustrations numériques.
- **Ex. 16.3** - Régression linéaire (resp. logistique) sur les règles-feuilles $R_1,R_4,R_5,R_6$ : on retrouve l'arbre de régression (moyennes par feuille) et l'arbre de classification (proportions par feuille).
- **Ex. 16.4** - Programmation complète de la simulation de la figure 16.2 : lasso contre ridge, régression et classification, scénarios dense / 10 / 30 coefficients non nuls, 5 rapports bruit/signal, 20 simulations par case.

## Module `esl/ch16_ensembles.py`

`forward_stagewise` (algorithme 16.1), `lasso_cd`, `lasso_path_cd` (descente de
coordonnées), `lasso_path_fast` (même critère via scikit-learn), `ridge_path_df`,
`block_correlated_data` (ex. 16.1), `l1_arc_length` (ex. 16.2), `adaboost_shrunk`
(marge $L_1$), `isle_ensemble` (algorithme 16.2), `tree_predictions`,
`staged_ensemble_predictions`, `extract_rules`, `rule_matrix`, `rule_to_str` (règles
16.14), `logistic_irls`.
