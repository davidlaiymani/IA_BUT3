# Introduction à l'apprentissage automatique — BUT 3 informatique

Ce module vous fait pratiquer les grandes familles de problèmes de l'apprentissage automatique sur des données tabulaires et temporelles. Il se compose de TP de 2 heures, suivis d'un **projet**.

**Prérequis** : Python et pandas. Un [pense-bête pandas](Pense_bete_Pandas.md) est fourni : gardez-le ouvert pendant les TP.

## Les TP

| TP | Thème | Jeu de données | Notions principales |
|---|---|---|---|
| [TP1](TP1_Clustering/) | **Clustering** | Clients d'un centre commercial | apprentissage non supervisé, standardisation, K-Means, coude et silhouette, ACP, interprétation des segments ; CAH en facultatif |
| [TP2](TP2_Classification/) | **Classification** | Clients d'un opérateur télécom (résiliation) | train / test, baseline, `Pipeline`, régression logistique, k-NN, arbre de décision, matrice de confusion, précision / rappel, ROC, validation croisée |

Les TP suivants seront publiés au fur et à mesure du semestre.

### Contenu de chaque dossier de TP

| Fichier | Rôle |
|---|---|
| `TPn_Cours.md` | le **cours** : notions, intuitions, formules et extraits de code. À lire avant ou pendant la séance. |
| `TPn_Enonce.ipynb` | l'**énoncé** sous forme de notebook : questions, cellules `# TODO` à compléter, cellules ✍️ pour les réponses rédigées. |
| `data/` | le jeu de données du TP. |

Chaque énoncé est calibré pour **2 heures** ; la durée indicative de chaque partie est donnée en tête de notebook. Les parties marquées « facultatif » et « Pour aller plus loin » sont à faire si vous avez terminé le reste.

## Installation

Deux outils au choix pour créer un environnement Python (3.10 ou plus récent) avec les bibliothèques du cours. Toutes les commandes se lancent depuis la racine du dépôt.

### Avec uv (recommandé)

[uv](https://docs.astral.sh/uv/) est un gestionnaire de paquets Python très rapide.

```bash
uv venv --python 3.12
source .venv/bin/activate        # Windows : .venv\Scripts\activate
uv pip install -r requirements.txt
jupyter lab
```

### Avec Conda

Avec [Miniconda](https://docs.anaconda.com/miniconda/) ou [Miniforge](https://github.com/conda-forge/miniforge) :

```bash
conda create -n ia-but3 -c conda-forge python=3.12 --file requirements.txt
conda activate ia-but3
jupyter lab
```

Ouvrez ensuite le notebook d'énoncé depuis le dossier du TP : les chemins vers `data/` sont relatifs à ce dossier.

## Sources des données

| Fichier | Source |
|---|---|
| `TP1_Clustering/data/Mall_Customers.csv` | *Mall Customer Segmentation Data*, Kaggle (jeu de données pédagogique) |
| `TP2_Classification/data/telco_churn.csv` | *Telco Customer Churn*, IBM (dépôt GitHub `IBM/telco-customer-churn-on-icp4d`, licence Apache 2.0), 7 043 clients |
