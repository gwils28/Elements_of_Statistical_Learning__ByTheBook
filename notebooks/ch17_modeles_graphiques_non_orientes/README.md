# Chapitre 17 - Modèles graphiques non orientés

Notebooks couvrant le chapitre 17 de *The Elements of Statistical Learning* : graphes de
Markov, modèle graphique gaussien (estimation à graphe connu, graphical lasso), modèles
de Ising et machines de Boltzmann.

Les fonctions réutilisables sont dans `esl/ch17_graphes.py` (tests :
`tests/test_ch17_graphes.py`).

> **Données.** Les données de cytométrie en flux du livre (Sachs et al., 11 protéines,
> 7466 cellules) ne sont pas disponibles : elles sont remplacées par des données
> gaussiennes simulées de même taille dont le vrai graphe est celui de la figure 17.1.
> Les données MNIST de la figure 17.7 sont remplacées par les chiffres manuscrits ZIP.

## Notebooks de cours

| Notebook | Contenu |
|---|---|
| `ch17_cours_1_graphes_et_modele_gaussien.ipynb` | Vocabulaire des graphes (adjacence, chemins, connexité, cliques maximales, séparation) ; indépendance conditionnelle ; propriétés de Markov par paires et globale ; factorisation sur les cliques (Hammersley–Clifford) ; modèle gaussien et matrice de précision $\Theta=\Sigma^{-1}$ (éq. 17.6–17.9), corrélations partielles ; log-vraisemblance (17.11) ; **algorithme 17.1** (exemple de la figure 17.4) ; sous-gradient et seuillage doux ; **graphical lasso** (algorithme 17.2), comparaison à scikit-learn ; chemin de régularisation et **figure 17.5** sur données simulées ; Meinshausen–Bühlmann ; compromis biais–variance pour $N$ petit. |
| `ch17_cours_2_modeles_ising_boltzmann.ipynb` | Modèle de Ising (17.28), fonction de partition, nœud constant ; lois conditionnelles logistiques (17.30) ; maximum de vraisemblance par énumération (égalisation des moments 17.34, Newton, IPF) ; coût en $2^p$ ; échantillonnage de Gibbs, champ moyen ; **pseudo-vraisemblance** ; régressions logistiques $\ell_1$ nœud par nœud (règles « min »/« max ») ; nœuds cachés (interaction d'ordre 3) ; **RBM** entraînée par divergence contrastive sur les chiffres ZIP (classification par énergie libre). |

## Notebooks d'exercices

| Notebook | Exercices |
|---|---|
| `ch17_exercices_01-06.ipynb` | 17.1 à 17.6 |
| `ch17_exercices_07-12.ipynb` | 17.7 à 17.12 |

- **17.1** - Indépendances conditionnelles (par paires, locales, séparateurs minimaux) et cliques maximales du graphe de la figure 17.8.
- **17.2** - Graphes réalisant des ensembles donnés d'indépendances conditionnelles (recherche exhaustive des graphes maximaux).
- **17.3** - Covariance partielle $\Sigma_{a.b}=\Theta_{aa}^{-1}$ et corrélation partielle $\rho_{jk\mid\text{reste}}=-\theta_{jk}/\sqrt{\theta_{jj}\theta_{kk}}$.
- **17.4** - $f(x_1\mid x_2,\dots)=f(x_1\mid x_3,\dots)$ équivaut à $X_1\perp X_2\mid X_3,\dots,X_p$.
- **17.5** - Sans arête manquante, $S_{11}\beta-s_{12}=0$ sont les équations normales de la régression (vérification sur les données de prostate).
- **17.6** - Formules d'inverse par blocs (17.41)–(17.42) : reconstruction de $\hat\Theta$ à partir des $\hat\beta$.
- **17.7** - Programmation de l'algorithme 17.1 sur le graphe de la figure 17.1 ; comparaison à l'approche naïve et test du rapport de vraisemblance.
- **17.8** - (a) Lasso par descente de coordonnées comparé à LARS ; (b) graphical lasso programmé à partir de (a), réseaux pour plusieurs $\lambda$, validation croisée.
- **17.9** - Algorithme EM avec valeurs manquantes : étapes E/M, rôle du terme correctif, imputation de Jnk et comparaison à la régression sur les voisins.
- **17.10** - Nécessité du nœud constant $X_0\equiv 1$ dans un modèle binaire à deux variables.
- **17.11** - Les lois conditionnelles du modèle de Ising sont logistiques.
- **17.12** - Le modèle de Ising comme régression de Poisson log-linéaire : $\hat\theta_{00}=\log N-\Phi(\Theta)$ et équivalence des équations de score.
