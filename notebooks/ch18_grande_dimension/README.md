# Chapitre 18 - Problèmes en grande dimension ($p \gg N$)

Ce chapitre traite de la prédiction et de l'évaluation de variables quand le nombre de
variables $p$ dépasse largement le nombre d'observations $N$ (génomique, protéomique,
textes…). Fonctions réutilisables : `esl/ch18_grande_dimension.py` (tests :
`tests/test_ch18_grande_dimension.py`).

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| [`ch18_cours_1_pgrandN_centroides.ipynb`](ch18_cours_1_pgrandN_centroides.ipynb) | Pourquoi $p \gg N$ est difficile : simulation ridge de la figure 18.1 (erreur de test exacte, degrés de liberté, cas $p = N$). LDA diagonale (règle d'indépendance, éq. (18.2)). Centroïdes réduits : écarts standardisés $d_{kj}$ et justification de $m_k$, seuillage doux/dur (fig. 18.2), choix de $\Delta$ par validation croisée et profils des centroïdes sur NCI (analogue de la fig. 18.4), carte de chaleur des gènes retenus (fig. 18.3), probabilités de classe (18.8). |
| [`ch18_cours_2_regularisation_noyaux.ipynb`](ch18_cours_2_regularisation_noyaux.ipynb) | RDA, logistique multinomiale ridge, SVM quand $p > N$. Astuce de calcul par la SVD : ridge en $O(N^2p)$ (18.15), théorème (18.16)-(18.17) vérifié sur une logistique ridge, multinomiale ridge sur NCI et comparaison de méthodes (analogue du tableau 18.1). Lasso (au plus $N$ variables), elastic net (fig. 18.5 sur données simulées), fused lasso et approximateur de signal sur des données CGH simulées (fig. 18.8). Classification avec seulement des produits scalaires : noyau de spectre sur des protéines simulées, centroïdes et distances par noyau (18.27)-(18.28), courbes ROC (fig. 18.9), double centrage (18.31), ACP à noyau, limites des noyaux (standardisation). |
| [`ch18_cours_3_acp_supervisee_tests_multiples.ipynb`](ch18_cours_3_acp_supervisee_tests_multiples.ipynb) | ACP supervisée : modèle à variable latente (18.32)-(18.33), algorithme 18.1, simulation (18.37) (fig. 18.15), comparaison à la PLS seuillée (fig. 18.16, éq. (18.35) démontrée), pré-conditionnement du lasso (fig. 18.17). Tests multiples : statistiques $t$ et permutations (fig. 18.18), $p$-valeurs poolées (18.41), FWER et Bonferroni, FDR et procédure de Benjamini-Hochberg (fig. 18.19), vérification par simulation de la borne (18.45), FDR « plug-in » (algorithme 18.3) et son équivalence avec BH, procédure SAM (fig. 18.20), interprétation bayésienne du pFDR (18.53), application à NCI (mélanomes). |

> Les données SRBCT, leucémie, protéines (Leslie et al.), lymphomes et radiation du livre
> ne sont pas disponibles hors ligne : les figures correspondantes sont reproduites sur les
> données NCI ou sur des données simulées de même structure.

## Exercices

### Exercices 18.1 à 18.10 - [`ch18_exercices_01-05.ipynb`](ch18_exercices_01-05.ipynb), [`ch18_exercices_06-10.ipynb`](ch18_exercices_06-10.ipynb)

- **18.1** - Ridge normalisée pour $\lambda \to \infty$ : convergence vers la direction PLS à une composante $\mathbf X^Ty$ (vérifiée sur *prostate* et en $p \gg N$).
- **18.2** - Les centroïdes réduits sont la solution d'un problème lasso sur les écarts de classe (avec $s_0 = 0$, $m_k^2 = 1/N_k$) ; vérification numérique sur NCI.
- **18.3** - Dans la multinomiale ridge, $\sum_k \hat\beta_{kj} = 0$ ; non-identifiabilité des intercepts et remèdes.
- **18.4** - Démonstration de la formule de calcul (18.15) de la ridge (et identité « push-through »).
- **18.5** - Preuve du théorème de réduction (18.16)-(18.17) ; vérification avec une perte charnière au carré et contre-exemple L1.
- **18.6** - La RDA (18.9) se calcule comme une LDA à covariance ridge sur les $N$ coordonnées de la SVD ; application à NCI.
- **18.7** - Moindres carrés pour $p \gg N$ : infinité de solutions interpolantes, ridge via la SVD, solution de norme minimale $\mathbf V\mathbf D^{-1}\mathbf U^Ty$.
- **18.8** - Directions d'empilement des données (*data piling*), marge $2/\|\hat\beta\|$ et direction d'empilement maximal $\mathbf X^+y$.
- **18.9** - Empilement maximal contre SVM à marge dure : la marge du SVM est toujours au moins aussi grande ; simulation en fonction de $p$.
- **18.10** - La limite $\lambda \downarrow 0$ de l'analyse discriminante ridge donne la direction d'empilement maximal.

### Exercices 18.11 à 18.20

Fonctions associées : `esl/ch18_grande_dimension_complements.py` (tests :
`tests/test_ch18_grande_dimension_complements.py`).

- [`ch18_exercices_11-15.ipynb`](ch18_exercices_11-15.ipynb) - séparabilité et
  logistique, raccourcis $X = RV^T$, méthodes à noyau, 1-ppv pondéré, ACP à noyau.
  - **18.11** - Classes séparables : la log-vraisemblance logistique a pour borne supérieure 0 sans l'atteindre, donc $\hat\beta = \infty$ ; en $p \ge N-1$ tout est séparable (illustration par montée de gradient et logistique ridge).
  - **18.12** - Validation croisée de la ridge : $X_{T} = R_{T}V^T$ pour toute sous-matrice de lignes, donc $R$ se calcule une seule fois (courbes de CV identiques, $p = 5000$).
  - **18.13** - Logistique ridge à partir de $K = XX^T$ : $\hat\alpha = UD^{-1}\hat\beta$ avec $\hat\beta$ ajusté sur $R = UD$ ; vérification contre l'ajustement direct et version à noyau gaussien.
  - **18.14** - 1-ppv pondéré par les distances : $\log(d_-/d_+)$ comme log-rapport d'estimateurs de densité par plus proches voisins, a priori quelconques, généralisation aux $K$-ppv, courbes ROC.
  - **18.15** - ACP à noyau : démonstration de la projection (18.58) d'un nouveau point, vérification contre l'ACP linéaire et `KernelPCA`, exemple des cercles concentriques.
- [`ch18_exercices_16-20.ipynb`](ch18_exercices_16-20.ipynb) - tests multiples
  (FWER, FDR, pFDR).
  - **18.16** - Bonferroni : inégalité de Boole ($\text{FWER} \le \alpha$), encadrement $1 - e^{-\alpha} \le 1-(1-\alpha/M)^M \le \alpha$ en cas d'indépendance, simulation avec tests corrélés.
  - **18.17** - Équivalence BH / FDR plug-in via l'identité $\widehat{\text{FDR}}(|T|_{(l)}) = Mp_{(l)}/l$ ; vérification sur données simulées.
  - **18.18** - pFDR $= \pi_0\cdot$type I $/ (\pi_0\cdot$type I $+ \pi_1\cdot$puissance$)$ par la formule de Bayes ; contrôle par simulation.
  - **18.19** - Étude « micro-puces » (simulation de même taille que les données de radiosensibilité, non disponibles) : FDR plug-in, BH pour plusieurs $\alpha$, estimation de $\pi_0$ par les quartiles nuls et justification.
  - **18.20** - Démonstration de $\text{pFDR}(\Gamma) = \Pr(Z = 0 \mid t \in \Gamma)$ : $V \mid R = k \sim \mathcal B(k, q)$ ; vérification par simulation.
