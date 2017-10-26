# Chapitre 11 - Réseaux de neurones

Le chapitre 11 présente deux méthodes qui reposent sur la même idée : construire des
**combinaisons linéaires des entrées** (variables dérivées) puis modéliser la réponse comme une
**fonction non linéaire** de ces combinaisons. La *projection pursuit regression* (PPR) utilise
des fonctions estimées par lissage ; le réseau de neurones à une couche cachée utilise des
sigmoïdes paramétrées, ajustées par **rétropropagation** du gradient.

Les fonctions réutilisables (réseau numpy avec rétropropagation, vérification du gradient,
spline de lissage à ddl fixés, PPR) sont dans [`esl/ch11_reseaux.py`](../../esl/ch11_reseaux.py),
testées dans [`tests/test_ch11_reseaux.py`](../../tests/test_ch11_reseaux.py).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| [`ch11_cours_1_projection_pursuit.ipynb`](ch11_cours_1_projection_pursuit.ipynb) | Modèle PPR (11.1), fonctions ridge (figure 11.1), le produit $X_1X_2$ comme somme de ridges, ajustement par spline de lissage + Gauss–Newton (11.2)–(11.4) avec convergence d'un *single index model*, construction progressive et ses limites sur la somme de sigmoïdes ; réseau à une couche cachée (11.5)–(11.6), diagramme (figure 11.2), sigmoïde et échelle (figure 11.3), le réseau comme PPR (11.7), nombre de poids (11.8). |
| [`ch11_cours_2_retropropagation.ipynb`](ch11_cours_2_retropropagation.ipynb) | Critères (11.9)–(11.10) ; rétropropagation dérivée pas à pas (11.11)–(11.15), forme matricielle et implémentation numpy ; vérification par différences finies (choix du pas) ; descente de gradient par lots / en ligne / L-BFGS ; valeurs initiales (symétrie, petits et grands poids) ; arrêt précoce ; weight decay et élimination de poids (11.16)–(11.17) ; mélange du chapitre 2 (figures 11.4 et 11.5) ; mise à l'échelle des entrées ; minima multiples et moyenne des prédictions. |
| [`ch11_cours_3_simulations_zip.ipynb`](ch11_cours_3_simulations_zip.ipynb) | Simulations « somme de sigmoïdes » et « radiale » (RSB (11.18), variance radiale exacte) : erreur de test selon le nombre d'unités et le weight decay (figures 11.6, 11.7, 11.8) ; chiffres manuscrits : exemples (figure 11.9), recalcul du tableau 11.1 (liaisons / poids des cinq architectures), Net-1 et Net-2 sur 320 chiffres au fil des époques (analogue de la figure 11.11), poids des unités cachées, réseaux régularisés sur 320 et 1000 chiffres ; discussion (section 11.8). |

## Notions couvertes

- **PPR** : fonctions ridge, approximation universelle, lissage par spline (ddl effectifs),
  étape de Gauss–Newton (moindres carrés pondérés), construction progressive, backfitting.
- **Réseau à une couche cachée** : unités cachées, sigmoïde, sortie identité ou softmax,
  correspondance exacte avec la PPR.
- **Rétropropagation** : règle de la chaîne, erreurs $\delta$ et $s$, deux passes,
  vérification par différences finies.
- **Entraînement** : descente de gradient par lots et en ligne, époques, taux d'apprentissage,
  approximation stochastique, quasi-Newton.
- **Régularisation et pratique** : valeurs initiales, arrêt précoce, weight decay,
  standardisation, nombre d'unités, minima multiples.
- **Exemples** : mélange du chapitre 2, simulations de la section 11.6, codes postaux
  (architectures locales et poids partagés).

## Exercices

| Notebook | Exercices |
|---|---|
| [`ch11_exercices_01-04.ipynb`](ch11_exercices_01-04.ipynb) | **11.1** correspondance exacte réseau ↔ PPR ($\omega_m = \alpha_m/\|\alpha_m\|$, $g_m(v) = \beta_m\sigma(\alpha_{0m} + \|\alpha_m\|v)$), cas de la classification, vérification numérique et fonctions ridge d'un réseau ajusté. **11.2** petits poids ⇒ modèle presque linéaire (Taylor–Lagrange, borne $\lvert\sigma''\rvert \le \sqrt3/18$, ordres $\varepsilon$ et $\varepsilon^2$ vérifiés). **11.3** rétropropagation pour l'entropie croisée ($\delta = f - y$), implémentation vérifiée, comparaison des gradients quadratique / entropie croisée. **11.4** réseau sans couche cachée = logistique multinomiale (contrastes, non-identifiabilité, effet du weight decay), vérification sur *vowel*. |
| [`ch11_exercices_05-07.ipynb`](ch11_exercices_05-07.ipynb) | **11.5** réseau à 10 unités par rétropropagation (descente avec inertie) sur $Y = \sigma(a_1^TX) + (a_2^TX)^2 + 0{,}3Z$ : courbes d'erreur par époque selon $\lambda$ (et variante bruitée montrant le sur-ajustement), nombre minimal d'unités (2). **11.6** PPR avec splines de lissage à ddl fixés sur les mêmes données : effet des ddl et du nombre de termes, 2 termes suffisent, comparaison avec le réseau. **11.7** *spam* : réseau à 10 unités contre modèle additif logistique (P-splines, Newton–Raphson) - erreurs de test, matrices de confusion, dépendances partielles, importances par permutation. |
