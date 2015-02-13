# Jeux de données

Données utilisées dans *The Elements of Statistical Learning*, telles que diffusées sur le
site du livre, converties en CSV (les fichiers volumineux sont compressés en `.csv.gz`).
Elles se chargent avec le module `esl.datasets`.

| Fichier | Chapitres | Description |
|---|---|---|
| `prostate.csv` | 1, 3 | Cancer de la prostate (Stamey et al.) : 97 patients, 8 prédicteurs, réponse `lpsa`, colonne `train` |
| `SAheart.csv` | 4 | South African Heart Disease : 462 sujets, réponse `chd` |
| `spam.csv` | 1, 9, 10, 15 | 4601 e-mails, 57 fréquences de mots/caractères, réponse `spam` |
| `zip_train.csv.gz`, `zip_test.csv.gz` | 1, 11, 13 | Chiffres manuscrits 16×16 (première colonne : le chiffre) |
| `vowel_train.csv`, `vowel_test.csv` | 4 | Reconnaissance de voyelles : 11 classes, 10 prédicteurs |
| `waveform_train.csv`, `waveform_test.csv` | 12 | Simulation waveform : 3 classes, 21 prédicteurs |
| `phoneme.csv.gz` | 5 | Log-périodogrammes de phonèmes (256 fréquences) |
| `bone.csv` | 5 | Densité minérale osseuse relative chez des adolescents |
| `galaxy.csv` | 6 | Vitesses radiales de la galaxie NGC7531 |
| `ozone.csv` | 6 | Ozone, radiation, température et vent (New York) |
| `marketing.csv` | 14 | Questionnaire marketing (14 variables démographiques) |
| `nci.csv.gz` | 1, 14, 18 | Micro-puces NCI60 : 6830 gènes × 64 lignées cellulaires |
| `countries.csv` | 14 | Dissimilarités politiques entre 12 pays |
| `mixture_example/` | 2, 13 | Simulation « mixture » : échantillon, grille et probabilités bayésiennes |
