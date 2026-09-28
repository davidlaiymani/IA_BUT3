# TP1 — Clustering : cours

[⬅️ Retour au sommaire](../README.md) · Énoncé : [TP1_Enonce.ipynb](TP1_Enonce.ipynb)

## Ce que vous saurez faire à la fin de ce TP

- Expliquer ce qu'est l'apprentissage **non supervisé** et à quoi sert le clustering.
- Préparer des données pour un algorithme fondé sur des distances (choix des variables, standardisation).
- Appliquer **K-Means**, choisir le nombre de groupes avec la méthode du coude et le score de silhouette.
- Connaître l'existence d'autres algorithmes (CAH, DBSCAN) et savoir quand les préférer.
- *(Facultatif)* Construire et lire un dendrogramme de classification ascendante hiérarchique.
- Interpréter des clusters pour en tirer des « personas » utiles au métier.

---

## 1. Apprentissage supervisé ou non supervisé ?

En **apprentissage supervisé** (TP2 à TP5), chaque exemple possède une étiquette connue (`y`) : on apprend à la prédire.

En **apprentissage non supervisé**, il n'y a **pas d'étiquette**. On cherche une structure cachée dans les données `X` seules. Le **clustering** (ou partitionnement) regroupe les individus de façon à ce que :

- les individus d'un même groupe se ressemblent (forte **homogénéité intra-cluster**) ;
- les groupes soient bien distincts les uns des autres (forte **séparation inter-cluster**).

Exemples d'usages :

| Domaine | Individus | Objectif |
|---|---|---|
| Marketing | clients | segmenter la clientèle pour cibler des offres |
| Réseau | machines, flux | repérer des comportements anormaux |
| Biologie | gènes | trouver des gènes au profil d'expression similaire |
| Documents | textes (vectorisés) | regrouper des articles par thème |

> ⚠️ Il n'y a pas de « bonne réponse » unique : un clustering est jugé sur sa **cohérence** (métriques internes) et son **utilité** pour le métier.

## 2. Tout repose sur une notion de distance

Pour dire que deux clients « se ressemblent », on mesure une distance entre leurs vecteurs de caractéristiques. La plus courante est la **distance euclidienne** :

$$d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}$$

### Le piège des échelles

Imaginons deux variables : l'âge (de 18 à 70) et le revenu annuel (de 15 000 à 140 000 €). Un écart de 10 000 € pèse alors 1 000 fois plus qu'un écart de 10 ans : l'algorithme **ignorera l'âge**.

La solution est de **standardiser** chaque variable (centrer-réduire) :

$$z = \frac{x - \mu}{\sigma}$$

Après standardisation, chaque variable a une moyenne de 0 et un écart-type de 1 : elles pèsent autant les unes que les autres.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # renvoie un tableau numpy
```

> 💡 Les variables **catégorielles** (sexe, ville…) se prêtent mal à la distance euclidienne. Pour débuter, on travaille sur des variables numériques et on utilise les catégorielles pour **interpréter** les clusters a posteriori.

## 3. K-Means

### 3.1 Principe

On fixe à l'avance le nombre de groupes `k`. Chaque groupe est représenté par son **centroïde** (le point moyen du groupe).

1. **Initialisation** : choisir `k` centroïdes (aléatoirement, ou avec la stratégie `k-means++` qui les écarte les uns des autres).
2. **Affectation** : chaque point rejoint le centroïde le plus proche.
3. **Mise à jour** : chaque centroïde est recalculé comme la moyenne des points de son groupe.
4. On répète 2 et 3 jusqu'à ce que les affectations ne changent plus.

```
 Itération 0          Itération 1          Convergence
 x  x    o o          x  x    o o          x  x    o o
  x ★     o ★  ──►     x★     o★    ──►     x★     ★o
 x   x   o  o         x   x   o  o         x   x   o  o
```

L'algorithme minimise l'**inertie** (ou WCSS, *within-cluster sum of squares*) : la somme des distances au carré de chaque point à son centroïde.

$$\text{Inertie} = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

### 3.2 Avec scikit-learn

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)   # numéro de cluster pour chaque individu

kmeans.cluster_centers_   # coordonnées des centroïdes (dans l'espace standardisé !)
kmeans.inertia_           # inertie finale
```

- `n_init=10` : l'algorithme est relancé 10 fois avec des initialisations différentes et garde la meilleure. K-Means ne trouve qu'un **optimum local**.
- `random_state` : rend le résultat reproductible.
- Les centroïdes sont dans l'espace standardisé. Pour les lire en unités d'origine : `scaler.inverse_transform(kmeans.cluster_centers_)`, ou plus simplement `df.groupby("cluster").mean()`.

### 3.3 Choisir `k`

**Méthode du coude.** L'inertie diminue toujours quand `k` augmente (avec `k = n`, elle vaut 0). On trace l'inertie en fonction de `k` et on cherche le « coude » : le point à partir duquel ajouter un cluster n'apporte plus grand-chose.

**Score de silhouette.** Pour chaque point `i` :

- `a(i)` = distance moyenne aux points de **son** cluster ;
- `b(i)` = distance moyenne aux points du cluster **voisin le plus proche**.

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$

| Valeur | Interprétation |
|---|---|
| proche de 1 | le point est bien au centre de son groupe |
| proche de 0 | le point est à la frontière entre deux groupes |
| négative | le point serait mieux dans un autre groupe |

Le score global est la moyenne des `s(i)` : on retient le `k` qui le **maximise**.

```python
from sklearn.metrics import silhouette_score
silhouette_score(X_scaled, labels)
```

> 💡 Les deux critères ne donnent pas toujours la même réponse. Le choix final tient aussi compte de l'**interprétabilité** : 5 segments clients bien distincts valent mieux que 9 segments illisibles.

### 3.4 Limites de K-Means

- Il faut fixer `k` à l'avance.
- Il suppose des groupes **ronds** (convexes) et de tailles comparables.
- Il est sensible aux valeurs aberrantes (la moyenne se laisse tirer).
- Chaque point est obligatoirement affecté à un groupe, même s'il est isolé.

## 4. Deux autres algorithmes, en bref

K-Means n'est pas le seul algorithme de clustering. Deux alternatives classiques :

- **Classification ascendante hiérarchique (CAH)** : on fusionne pas à pas les groupes les plus proches, ce qui produit un arbre (le **dendrogramme**). Elle est détaillée en section 7 et pratiquée dans la partie facultative du TP.
- **DBSCAN** : un cluster est une **zone dense** de points (au moins `min_samples` voisins dans un rayon `eps`). Il trouve des groupes de **forme quelconque** (anneaux, croissants) que K-Means ne sait pas séparer, et marque les points isolés comme **bruit**, ce qui en fait aussi un outil de détection d'anomalies. Inconvénient : très sensible au réglage de `eps`, et peu efficace quand les groupes ne sont pas séparés par des zones vides.

| Situation | Choix conseillé |
|---|---|
| Groupes compacts, beaucoup de données, `k` à peu près connu | K-Means |
| Petit jeu de données, besoin de visualiser la structure | CAH |
| Formes irrégulières, présence de bruit ou d'aberrants | DBSCAN |

## 5. Visualiser en plus de 2 dimensions : l'ACP en bref

Avec plus de deux variables, on ne peut pas dessiner les clusters directement. L'**ACP** (analyse en composantes principales, `PCA`) projette les données sur les 2 axes qui conservent le plus de variance :

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
pca.explained_variance_ratio_   # part de variance conservée par chaque axe
```

On colore ensuite les points par cluster. Attention : une projection peut faire se chevaucher des groupes qui sont bien séparés dans l'espace complet.

## 6. Interpréter les clusters

Un clustering n'a de valeur que s'il est **interprétable**. Démarche type :

1. Calculer le profil moyen de chaque cluster : `df.groupby("cluster").mean()`.
2. Compter les effectifs : `df["cluster"].value_counts()`.
3. Comparer chaque cluster à la moyenne globale (au-dessus, en dessous).
4. Donner un nom parlant à chaque groupe (« jeunes gros dépensiers », « clients prudents à haut revenu »…).
5. Proposer une action métier par groupe.

## 7. Pour aller plus loin (facultatif) : la CAH

*Cette section accompagne la partie 5, facultative, de l'énoncé.*

### 7.1 Principe

On part de `n` groupes contenant chacun un seul individu, puis on **fusionne à chaque étape les deux groupes les plus proches**, jusqu'à n'en avoir plus qu'un. L'historique des fusions se représente par un **dendrogramme**.

```
 hauteur
   │        ┌────────┴────────┐
   │     ┌──┴──┐              │        ← couper ici donne 3 groupes
   │   ┌─┴─┐   │          ┌───┴───┐
   │   A   B   C          D       E
```

La hauteur d'une fusion indique la distance entre les groupes fusionnés. **Couper** le dendrogramme à une hauteur donnée fournit une partition : on choisit de couper là où les branches verticales sont les plus longues (les groupes fusionnés sont alors très différents).

### 7.2 Le critère de liaison

Comment mesurer la distance entre deux **groupes** ?

| Liaison | Distance entre groupes | Comportement |
|---|---|---|
| `single` | les deux points les plus proches | forme des chaînes, sensible au bruit |
| `complete` | les deux points les plus éloignés | groupes compacts |
| `average` | moyenne de toutes les paires | compromis |
| `ward` | augmentation d'inertie causée par la fusion | groupes compacts et équilibrés, proche de K-Means ; **le plus utilisé** |

### 7.3 Avec scipy et scikit-learn

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

Z = linkage(X_scaled, method="ward")
dendrogram(Z, truncate_mode="lastp", p=30)   # affiche les 30 dernières fusions

cah = AgglomerativeClustering(n_clusters=5, linkage="ward")
labels_cah = cah.fit_predict(X_scaled)
```

Avantages : pas besoin de fixer `k` avant de voir le dendrogramme, résultat déterministe. Inconvénient : coût en mémoire en O(n²), inutilisable au-delà de quelques dizaines de milliers d'individus.

## À retenir

- Le clustering cherche des groupes **sans étiquette** ; il n'y a pas de vérité unique.
- **Standardisez** toujours les variables avant un algorithme fondé sur des distances.
- K-Means : rapide, groupes ronds, `k` à choisir (coude, silhouette).
- CAH et DBSCAN sont des alternatives quand les groupes ne sont pas « ronds » ou qu'on veut explorer la structure.
- Un bon clustering est un clustering **qu'on sait expliquer**.

## Pour aller plus loin

- [Guide scikit-learn sur le clustering](https://scikit-learn.org/stable/modules/clustering.html) : comparaison visuelle de tous les algorithmes.
- Les modèles de mélange gaussien (`GaussianMixture`) : un clustering « souple » où chaque point a une probabilité d'appartenir à chaque groupe.
