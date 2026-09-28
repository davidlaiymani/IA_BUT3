# Introduction à l'apprentissage automatique — BUT 3 informatique

Ce module vous fait pratiquer les grandes familles de problèmes de l'apprentissage automatique sur des données tabulaires et temporelles. Il se compose de TP de 2 heures, suivis d'un **projet**.

**Prérequis** : Python et pandas. Un [pense-bête pandas](Pense_bete_Pandas.md) est fourni : gardez-le ouvert pendant les TP.

## Les TP

| TP | Thème | Jeu de données | Notions principales |
|---|---|---|---|
| [TP1](TP1_Clustering/) | **Clustering** | Clients d'un centre commercial | apprentissage non supervisé, standardisation, K-Means, coude et silhouette, ACP, interprétation des segments ; CAH en facultatif |

Les TP suivants seront publiés au fur et à mesure du semestre.

### Contenu de chaque dossier de TP

| Fichier | Rôle |
|---|---|
| `TPn_Cours.md` | le **cours** : notions, intuitions, formules et extraits de code. À lire avant ou pendant la séance. |
| `TPn_Enonce.ipynb` | l'**énoncé** sous forme de notebook : questions, cellules `# TODO` à compléter, cellules ✍️ pour les réponses rédigées. |
| `data/` | le jeu de données du TP. |

Chaque énoncé est calibré pour **2 heures** ; la durée indicative de chaque partie est donnée en tête de notebook. Les parties marquées « facultatif » et « Pour aller plus loin » sont à faire si vous avez terminé le reste.

## Installation

Il faut Python 3.10 ou plus récent.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Ouvrez ensuite le notebook d'énoncé depuis le dossier du TP : les chemins vers `data/` sont relatifs à ce dossier.

## Sources des données

| Fichier | Source |
|---|---|
| `TP1_Clustering/data/Mall_Customers.csv` | *Mall Customer Segmentation Data*, Kaggle (jeu de données pédagogique) |
