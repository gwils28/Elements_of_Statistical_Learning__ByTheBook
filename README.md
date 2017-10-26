# The Elements of Statistical Learning - *By The Book*

Notebooks Jupyter pédagogiques qui accompagnent la lecture de
**The Elements of Statistical Learning** (T. Hastie, R. Tibshirani, J. Friedman, 2e édition) :
résumés de cours, reproduction des figures principales et **solutions détaillées de tous
les exercices**, chapitre par chapitre.

## Objectif et public

Les notebooks sont écrits en français pour un lecteur de **niveau Licence de
mathématiques** : algèbre linéaire, calcul différentiel, probabilités et statistique de
base. Toute notion qui dépasse ce niveau (multiplicateurs de Lagrange et conditions KKT,
algorithme EM, espaces de Hilbert à noyau reproduisant, modèles graphiques…) est introduite
avant d'être utilisée.

Chaque exercice suit le même plan :

1. **énoncé** reformulé, avec un rappel des équations du livre utilisées ;
2. **rappels** des notions nécessaires ;
3. **solution** détaillée pas à pas, avec les formules en $\LaTeX$ ;
4. **vérification numérique** et/ou **graphique** (simulation, données du livre) ;
5. un court **bilan** (« À retenir »).

## Organisation du dépôt

```
notebooks/chXX_<theme>/   notebooks de cours (chXX_cours_*) et d'exercices (chXX_exercices_*)
esl/                      paquet Python : chargement des données, style graphique,
                          implémentations « à la main » des méthodes de chaque chapitre
tests/                    tests unitaires (pytest) des implémentations
data/                     jeux de données utilisés dans le livre
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter notebook
```

Lancer les tests : `pytest tests/`.

## Avancement


| Chapitre | Thème | Statut |
|---|---|---|
| 1 | [Introduction](notebooks/ch01_introduction) | ✅ |
| 2 | [Vue d'ensemble de l'apprentissage supervisé](notebooks/ch02_apprentissage_supervise) | ✅ |
| 3 | [Méthodes linéaires pour la régression](notebooks/ch03_regression_lineaire) | ✅ |
| 4 | [Méthodes linéaires pour la classification](notebooks/ch04_classification_lineaire) | ✅ |
| 5 | [Expansions de bases et régularisation](notebooks/ch05_bases_et_regularisation) | ✅ |
| 6 | [Méthodes de lissage par noyaux](notebooks/ch06_lissage_par_noyaux) | ✅ |
| 7 | [Évaluation et sélection de modèles](notebooks/ch07_evaluation_selection_modeles) | ✅ |
| 8 | [Inférence et moyennage de modèles](notebooks/ch08_inference_et_moyennage) | ✅ |
| 9 | [Modèles additifs, arbres et méthodes apparentées](notebooks/ch09_modeles_additifs_arbres) | ✅ |
| 10 | [Boosting et arbres additifs](notebooks/ch10_boosting) | ✅ |
| 11 | [Réseaux de neurones](notebooks/ch11_reseaux_de_neurones) | ✅ |
| 12 | SVM et discriminants flexibles | 🚧 |
| 13 | Méthodes de prototypes et plus proches voisins | - |
| 14 | Apprentissage non supervisé | - |
| 15 | Forêts aléatoires | - |
| 16 | Apprentissage ensembliste | - |
| 17 | Modèles graphiques non orientés | - |
| 18 | Problèmes en grande dimension (p ≫ N) | - |

## Références

- T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning: Data Mining,
  Inference, and Prediction*, 2e édition, Springer, 2009. Le livre est librement disponible
  sur la page des auteurs.
- Les jeux de données proviennent du site du livre (voir `data/README.md`).
