# TP1 — Clustering : cours

[⬅️ Retour au sommaire](../README.md) · Énoncé : [TP1_Enonce.ipynb](TP1_Enonce.ipynb)

## Ce que vous saurez faire à la fin de ce TP

- Expliquer ce qu'est l'apprentissage **non supervisé** et à quoi sert le clustering.
- Préparer des données pour un algorithme fondé sur des distances (choix des variables, standardisation).
- Appliquer **K-Means**, choisir le nombre de groupes avec la méthode du coude et le score de silhouette.
- Connaître l'existence d'autres algorithmes (CAH, DBSCAN) et savoir quand les préférer.
- *(Facultatif)* Construire et lire un dendrogramme de classification ascendante hiérarchique.
- Interpréter des clusters pour en tirer des « personas » utiles au métier.

> 📖 Les mots en **gras** sont définis dans le texte ; vous les retrouverez tous dans le [lexique](#lexique) en fin de document.

---

## 0. Un peu de vocabulaire pour commencer

En apprentissage automatique, les données se présentent presque toujours sous la forme d'un **tableau** (un DataFrame pandas) :

| id | âge | revenu (k$) | score de dépense |
|---|---|---|---|
| 1 | 19 | 15 | 39 |
| 2 | 21 | 15 | 81 |
| 3 | 20 | 16 | 6 |

- Chaque **ligne** est un **individu** (on dit aussi *observation*, *exemple* ou *point*) : ici, un client.
- Chaque **colonne** est une **variable** (en anglais *feature*, en français on dit aussi *caractéristique* ou *attribut*) : ici, l'âge, le revenu, le score.
- Un individu décrit par `p` variables numériques peut être vu comme un **point** dans un espace à `p` **dimensions**. Le client 1 est le point de coordonnées (19, 15, 39) dans un espace à 3 dimensions. C'est ce regard « géométrique » qui permet de parler de distance entre clients.

Par convention, on note `X` le tableau des variables, `n` le nombre d'individus et `p` le nombre de variables.

## 1. Apprentissage supervisé ou non supervisé ?

### 1.1 Deux grandes familles

**Apprentissage supervisé** (TP2 à TP5). Pour chaque individu, on connaît la « bonne réponse », appelée **étiquette** ou **cible** et notée `y` : un client a-t-il résilié son abonnement ? quel est le prix de cette maison ? On donne à l'algorithme des exemples avec leur réponse, et il apprend à **prédire** la réponse pour de nouveaux individus. C'est comme apprendre avec un professeur qui corrige.

**Apprentissage non supervisé** (ce TP). Il n'y a **pas d'étiquette** : on dispose seulement de `X`. On ne cherche pas à prédire quelque chose, mais à **découvrir une structure** dans les données. C'est comme trier une boîte de photos sans qu'on vous dise quelles catégories utiliser : c'est à vous de trouver des regroupements qui ont du sens.

### 1.2 Le clustering

Le **clustering** (en français *partitionnement* ou *classification non supervisée*) consiste à répartir les individus en groupes, appelés **clusters**, de sorte que :

- les individus d'un même cluster se **ressemblent** entre eux : on parle d'**homogénéité intra-cluster** (*intra* = à l'intérieur) ;
- les individus de clusters différents soient **différents** : on parle de **séparation inter-cluster** (*inter* = entre).

```
      ○ ○                          ●●
     ○ ○ ○        ← cluster A      ● ●●     ← cluster B
      ○ ○                           ●●
     (serrés = homogène)      (loin de A = bien séparé)
```

Exemples d'usages :

| Domaine | Individus | Objectif |
|---|---|---|
| Marketing | clients | **segmenter** la clientèle, c'est-à-dire la découper en groupes homogènes, pour adapter les offres à chacun |
| Réseau | machines, flux | repérer des comportements anormaux (ce qui ne ressemble à aucun groupe) |
| Biologie | gènes | trouver des gènes qui réagissent de la même façon |
| Documents | textes (transformés en nombres) | regrouper des articles par thème |

> ⚠️ Comme il n'y a pas d'étiquette, il n'y a pas de « bonne réponse » unique à laquelle se comparer. Un clustering se juge sur deux critères : sa **cohérence** (les groupes sont-ils compacts et bien séparés ? on le mesure avec des indicateurs comme la silhouette, section 3.3) et son **utilité** (les groupes ont-ils un sens pour le métier ?).

## 2. Tout repose sur une notion de distance

### 2.1 Mesurer la ressemblance

Pour dire que deux clients « se ressemblent », il faut transformer cette idée floue en un **nombre**. On utilise une **distance** : plus elle est petite, plus les clients se ressemblent.

La plus courante est la **distance euclidienne**, celle que vous mesurez avec une règle, généralisée à `p` dimensions (c'est le théorème de Pythagore) :

$$d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}$$

On fait la différence variable par variable, on l'élève au carré, on additionne, puis on prend la racine.

**Exemple.** Client A : 25 ans, revenu 40, score 80. Client B : 30 ans, revenu 45, score 20.

$$d(A, B) = \sqrt{(25-30)^2 + (40-45)^2 + (80-20)^2} = \sqrt{25 + 25 + 3600} = \sqrt{3650} \approx 60{,}4$$

On voit que c'est l'écart de score (60 points) qui fait presque toute la distance : les deux clients ont un âge et un revenu proches, mais des habitudes de dépense opposées.

### 2.2 Le piège des échelles

Supposons maintenant que le revenu soit exprimé en **euros** et non en milliers d'euros :

| Client | Âge | Revenu (€) |
|---|---|---|
| A | 25 | 40 000 |
| B | 60 | 41 000 |
| C | 26 | 45 000 |

- $d(A, B) = \sqrt{35^2 + 1000^2} \approx 1\,000{,}6$
- $d(A, C) = \sqrt{1^2 + 5000^2} \approx 5\,000{,}0$

Selon cette distance, A ressemble davantage à B (35 ans d'écart !) qu'à C (1 an d'écart). Les revenus se comptent en milliers alors que les âges se comptent en dizaines : les écarts de revenu **écrasent** complètement ceux d'âge. L'algorithme ignorerait l'âge, simplement à cause d'un choix d'unité.

### 2.3 La solution : standardiser

**Standardiser** (on dit aussi *centrer-réduire*) une variable, c'est l'exprimer non plus dans son unité d'origine, mais en **nombre d'écarts-types par rapport à la moyenne** :

$$z = \frac{x - \mu}{\sigma}$$

- $\mu$ (« mu ») est la **moyenne** de la variable ;
- $\sigma$ (« sigma ») est son **écart-type** : une mesure de la dispersion des valeurs autour de la moyenne. Grossièrement, c'est l'écart « typique » entre une valeur et la moyenne. Un petit écart-type signifie des valeurs resserrées, un grand écart-type des valeurs très étalées.

**Exemple.** Si l'âge moyen des clients est de 40 ans avec un écart-type de 14 ans, un client de 25 ans a pour âge standardisé $z = (25 - 40) / 14 \approx -1{,}07$ : il est environ « un écart-type en dessous de la moyenne ». Un client de 40 ans aurait $z = 0$, un client de 54 ans $z = +1$.

Après standardisation, **toutes les variables ont une moyenne de 0 et un écart-type de 1**, quelle que soit leur unité d'origine : elles pèsent autant les unes que les autres dans le calcul des distances.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # renvoie un tableau numpy (et non un DataFrame)
```

Le vocabulaire de scikit-learn, que vous retrouverez dans tous les TP :

- `fit` : **apprendre** quelque chose à partir des données (ici, calculer la moyenne et l'écart-type de chaque colonne) ;
- `transform` : **appliquer** ce qui a été appris (ici, calculer les `z`) ;
- `fit_transform` : les deux à la suite.

> 💡 Les variables **catégorielles** (qui prennent des valeurs non numériques : sexe, ville…) se prêtent mal à la distance euclidienne : quelle est la « distance » entre Lyon et Nantes ? Dans ce TP, on travaille sur des variables numériques et on utilise les catégorielles pour **interpréter** les clusters une fois qu'ils sont formés.

## 3. K-Means

K-Means (« k moyennes ») est l'algorithme de clustering le plus utilisé : simple, rapide, et souvent efficace.

### 3.1 Principe

On choisit à l'avance le nombre de groupes, noté `k`. Chaque groupe est représenté par un point appelé **centroïde** : c'est le **point moyen** du groupe, son « centre de gravité ». Ses coordonnées sont les moyennes des variables des individus du groupe.

L'algorithme alterne deux étapes simples :

1. **Initialisation** : placer `k` centroïdes de départ (au hasard, ou avec la stratégie `k-means++` qui les choisit éloignés les uns des autres, ce qui marche mieux).
2. **Affectation** : chaque individu rejoint le groupe du centroïde **le plus proche**.
3. **Mise à jour** : chaque centroïde est déplacé au **centre** (la moyenne) des individus de son groupe.
4. On répète 2 et 3 jusqu'à ce que plus aucun individu ne change de groupe. On dit alors que l'algorithme a **convergé**.

### 3.2 Un exemple déroulé à la main

Prenons 6 clients décrits par une seule variable (pour pouvoir tout calculer de tête) : **1, 2, 3, 10, 11, 12**, et `k = 2`. On place volontairement mal les centroïdes de départ : **c₁ = 1** et **c₂ = 2**.

| Étape | Groupe 1 | Groupe 2 | Centroïdes |
|---|---|---|---|
| Départ | | | c₁ = 1, c₂ = 2 |
| Affectation 1 | {1} | {2, 3, 10, 11, 12} | |
| Mise à jour 1 | | | c₁ = 1, c₂ = (2+3+10+11+12) / 5 = 7,6 |
| Affectation 2 | {1, 2, 3} | {10, 11, 12} | (2 est à 1 de c₁ et à 5,6 de c₂, etc.) |
| Mise à jour 2 | | | c₁ = 2, c₂ = 11 |
| Affectation 3 | {1, 2, 3} | {10, 11, 12} | plus rien ne change : **convergence** |

Malgré un départ maladroit, deux itérations suffisent pour trouver les deux groupes évidents.

### 3.3 Ce que K-Means optimise : l'inertie

Pour mesurer la qualité d'un découpage, K-Means utilise l'**inertie** (en anglais *WCSS*, *within-cluster sum of squares*, « somme des carrés intra-cluster ») : la somme, sur tous les individus, du carré de la distance à leur centroïde.

$$\text{Inertie} = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

où $\mu_{c(i)}$ est le centroïde du groupe de l'individu $i$. Une inertie **faible** signifie des groupes **compacts** : chaque individu est proche de son centre.

Dans l'exemple : groupe 1 : $(1-2)^2 + (2-2)^2 + (3-2)^2 = 2$ ; groupe 2 : $(10-11)^2 + 0 + (12-11)^2 = 2$. **Inertie totale = 4.**

Chaque étape d'affectation et de mise à jour fait baisser (ou laisse stable) l'inertie, ce qui garantit que l'algorithme finit par s'arrêter.

**Optimum local.** K-Means garantit de trouver un découpage qu'on ne peut plus améliorer **par de petites modifications**, mais pas forcément le meilleur découpage possible. C'est comme un randonneur qui descend toujours la pente : il arrive au fond d'une vallée, mais pas forcément de la vallée la plus profonde. Selon les centroïdes de départ, le résultat peut changer. D'où l'astuce ci-dessous : lancer l'algorithme plusieurs fois et garder le meilleur résultat (celui d'inertie minimale).

### 3.4 Avec scikit-learn

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)   # numéro de cluster (0, 1, 2 ou 3) pour chaque individu

kmeans.cluster_centers_   # coordonnées des centroïdes (dans l'espace standardisé !)
kmeans.inertia_           # inertie finale
```

- `n_clusters` : le nombre de groupes `k`. C'est un **hyperparamètre** : un réglage que l'on fixe **avant** l'apprentissage, et que l'algorithme ne choisit pas lui-même (par opposition aux centroïdes, que l'algorithme calcule).
- `n_init=10` : l'algorithme est lancé 10 fois avec des initialisations différentes, et on garde le résultat de plus faible inertie. C'est la parade contre l'optimum local.
- `random_state=42` : fixe le générateur de nombres aléatoires. Sans lui, chaque exécution donnerait des initialisations différentes, donc potentiellement des résultats (et des numéros de clusters) différents. Avec lui, le résultat est **reproductible**. La valeur 42 n'a rien de magique.
- `fit_predict` : apprend les centroïdes (`fit`) puis renvoie le groupe de chaque individu (`predict`).
- Les centroïdes sont exprimés en valeurs standardisées, difficiles à lire. Pour les retrouver en unités d'origine : `scaler.inverse_transform(kmeans.cluster_centers_)`, ou plus simplement `df.groupby("cluster").mean()` après avoir ajouté la colonne `cluster` au DataFrame.

> 💡 Les numéros de clusters sont arbitraires : le « cluster 0 » n'est pas plus important que le « cluster 3 », et une nouvelle exécution peut les numéroter autrement. Seul compte le regroupement.

### 3.5 Choisir `k`

C'est la grande difficulté de K-Means : l'algorithme découpera toujours en exactement `k` groupes, même si les données n'en contiennent « naturellement » que 2. Deux outils aident à choisir.

#### La méthode du coude

On entraîne K-Means pour `k = 1, 2, 3, …` et on trace l'inertie en fonction de `k`.

L'inertie **diminue toujours** quand `k` augmente : plus il y a de groupes, plus chaque individu est proche de son centre (avec autant de groupes que d'individus, l'inertie vaut 0, mais ce découpage ne sert à rien). On ne cherche donc pas l'inertie minimale, mais le **coude** de la courbe : la valeur de `k` à partir de laquelle ajouter un groupe ne fait presque plus baisser l'inertie.

Sur notre exemple à 6 points :

| `k` | Meilleur découpage | Inertie |
|---|---|---|
| 1 | {1, 2, 3, 10, 11, 12} | 125,5 |
| 2 | {1, 2, 3} {10, 11, 12} | 4 |
| 3 | {1, 2, 3} {10, 11} {12} | 2,5 |

```
 inertie
 125 ┤ ●
     │  \
     │   \
     │    \
   4 ┤     ●──────●         ← coude à k = 2
     └─────┬──────┬──────► k
     1     2      3
```

Passer de 1 à 2 groupes divise l'inertie par 30 ; passer de 2 à 3 ne gagne presque rien : le coude est à `k = 2`.

#### Le score de silhouette

La silhouette mesure, **pour chaque individu**, s'il est bien placé dans son groupe. On calcule deux distances moyennes :

- `a(i)` : distance moyenne entre `i` et les autres membres de **son** cluster (« suis-je proche des miens ? ») ;
- `b(i)` : distance moyenne entre `i` et les membres du **cluster voisin le plus proche** (« suis-je loin des autres ? »).

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Ce score est toujours compris entre −1 et 1 :

| Valeur | Interprétation |
|---|---|
| proche de 1 | l'individu est bien au cœur de son groupe, loin des autres |
| proche de 0 | l'individu est à la frontière entre deux groupes |
| négative | l'individu est plus proche d'un autre groupe que du sien : il est probablement mal classé |

**Exemple** (point 3 du découpage {1, 2, 3} {10, 11, 12}) : $a = (2 + 1) / 2 = 1{,}5$ ; $b = (7 + 8 + 9) / 3 = 8$ ; $s = (8 - 1{,}5) / 8 \approx 0{,}81$. Le point 3 est bien placé.

Le **score de silhouette** d'un découpage est la moyenne des `s(i)` de tous les individus. On le calcule pour plusieurs valeurs de `k` et on retient celle qui le **maximise**.

```python
from sklearn.metrics import silhouette_score
silhouette_score(X_scaled, labels)
```

> 💡 Le coude et la silhouette ne donnent pas toujours la même réponse, et le coude est parfois peu marqué. Le choix final tient aussi compte de l'**interprétabilité** : 5 segments de clients bien distincts et faciles à décrire valent mieux que 9 segments illisibles, même si la silhouette est un peu meilleure.

### 3.6 Limites de K-Means

- **Il faut fixer `k` à l'avance**, ce qui demande des essais (section 3.5).
- **Il suppose des groupes « ronds »** et de tailles comparables : puisque chaque individu rejoint le centre le plus proche, les frontières entre groupes sont des lignes droites. Il échoue sur des groupes allongés, en anneau ou en croissant.
- **Il est sensible aux valeurs aberrantes** (des individus aux valeurs extrêmes, très différents des autres) : comme le centroïde est une moyenne, un seul client au revenu gigantesque peut « tirer » le centre de son groupe.
- **Tout individu est obligatoirement affecté à un groupe**, même s'il ne ressemble à personne.

## 4. Deux autres algorithmes, en bref

K-Means n'est pas le seul algorithme de clustering. Deux alternatives classiques :

- **Classification ascendante hiérarchique (CAH)** : au lieu de fixer `k` au départ, on fusionne pas à pas les groupes les plus proches, ce qui produit un arbre (le **dendrogramme**) dans lequel on choisit ensuite le nombre de groupes. Elle est détaillée en section 7 et pratiquée dans la partie facultative du TP.
- **DBSCAN** : un cluster est défini comme une **zone dense** de points, c'est-à-dire une zone où les individus sont nombreux et serrés (au moins `min_samples` voisins dans un rayon `eps`). DBSCAN trouve des groupes de **forme quelconque** (anneaux, croissants) que K-Means ne sait pas séparer, et marque les individus isolés comme **bruit** au lieu de les forcer dans un groupe, ce qui en fait aussi un outil de détection d'anomalies. Inconvénient : il est très sensible au réglage de `eps`, et peu efficace quand les groupes ne sont pas séparés par des zones vides.

| Situation | Choix conseillé |
|---|---|
| Groupes compacts, beaucoup de données, `k` à peu près connu | K-Means |
| Petit jeu de données, besoin de visualiser la structure | CAH |
| Formes irrégulières, présence de bruit ou de valeurs aberrantes | DBSCAN |

## 5. Visualiser en plus de 2 dimensions : l'ACP en bref

Avec 2 variables, on dessine les clusters sur un simple nuage de points. Avec 3 variables ou plus, c'est impossible directement. L'**ACP** (*analyse en composantes principales*, `PCA` en anglais) résout ce problème en **projetant** les données sur un plan.

**L'analogie de l'ombre.** Éclairez un objet en 3D avec une lampe : son ombre sur le mur est une image en 2D. Selon l'angle de la lampe, l'ombre est plus ou moins fidèle : l'ombre d'une tasse vue de côté montre l'anse, vue de dessus ce n'est qu'un disque. L'ACP cherche automatiquement **le meilleur angle**, celui qui donne l'ombre la plus étalée, donc la plus informative.

Techniquement, l'ACP cherche les axes selon lesquels les données sont le plus **dispersées**. Cette dispersion se mesure par la **variance** (le carré de l'écart-type). Le premier axe (la « composante principale 1 ») est la direction de plus grande variance, le deuxième la suivante, perpendiculaire à la première, etc.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)                 # on veut 2 axes
X_2d = pca.fit_transform(X_scaled)        # coordonnées de chaque individu sur ces 2 axes
pca.explained_variance_ratio_             # part de la variance totale conservée par chaque axe
```

Si `explained_variance_ratio_` vaut par exemple `[0.44, 0.33]`, les deux axes conservent 77 % de la dispersion totale des données : la projection est assez fidèle. On trace ensuite `X_2d` en colorant chaque point selon son cluster.

> ⚠️ Une projection perd de l'information : deux groupes bien séparés dans l'espace complet peuvent se chevaucher sur le dessin, comme deux objets distincts peuvent avoir des ombres qui se superposent. L'ACP sert ici uniquement à **visualiser** ; le clustering, lui, est calculé sur toutes les variables.

## 6. Interpréter les clusters

K-Means renvoie des numéros de groupes, pas des explications. C'est à vous de donner du **sens** aux groupes, et c'est la partie la plus utile pour le métier. Démarche type :

1. **Profil moyen** de chaque cluster : `df.groupby("cluster").mean()`. On obtient, pour chaque groupe, l'âge moyen, le revenu moyen, etc.
2. **Effectifs** : `df["cluster"].value_counts()`. Un cluster de 3 clients n'a pas le même poids qu'un cluster de 60.
3. **Comparer à la moyenne globale** : chaque groupe est-il plus jeune, plus riche, plus dépensier que l'ensemble des clients ?
4. **Nommer** chaque groupe par une étiquette parlante, appelée **persona** : un portrait-type (« jeunes gros dépensiers », « clients aisés mais économes »…).
5. **Proposer une action** adaptée à chaque groupe (offre ciblée, programme de fidélité…).

Un clustering que l'on ne sait pas expliquer ne sert à rien, même avec une excellente silhouette.

## 7. Pour aller plus loin (facultatif) : la CAH

*Cette section accompagne la partie 5, facultative, de l'énoncé.*

### 7.1 Principe

La **classification ascendante hiérarchique** porte bien son nom :

- **ascendante** : on part du bas (chaque individu forme un groupe à lui seul) et on remonte en fusionnant ;
- **hiérarchique** : les groupes s'emboîtent les uns dans les autres, comme les dossiers d'un disque dur.

L'algorithme :

1. Au départ, `n` groupes contenant chacun un individu.
2. On cherche les **deux groupes les plus proches** et on les fusionne.
3. On recommence jusqu'à n'avoir plus qu'un seul groupe contenant tout le monde.

L'historique des fusions se représente par un arbre, le **dendrogramme** :

```
 hauteur
   │        ┌────────┴────────┐
   │     ┌──┴──┐              │        ← couper ici donne 3 groupes : {A, B}, {C}, {D, E}
   │   ┌─┴─┐   │          ┌───┴───┐
   │   A   B   C          D       E
```

- En bas, les individus (les « feuilles » de l'arbre).
- Chaque trait horizontal est une **fusion** ; sa **hauteur** indique la distance entre les deux groupes fusionnés. A et B, qui fusionnent très bas, se ressemblent beaucoup ; la fusion finale, tout en haut, réunit des groupes très différents.
- **Couper** l'arbre par une ligne horizontale donne un découpage : le nombre de branches coupées est le nombre de groupes. On coupe de préférence là où les traits verticaux sont **les plus longs** : cela signifie qu'il a fallu « monter » beaucoup pour fusionner ces groupes, donc qu'ils sont vraiment différents.

### 7.2 Le critère de liaison

On sait mesurer la distance entre deux individus (section 2), mais comment mesurer la distance entre deux **groupes** ? Il existe plusieurs conventions, appelées **critères de liaison** :

| Liaison | Distance entre deux groupes = | Comportement |
|---|---|---|
| `single` (simple) | distance entre leurs deux points **les plus proches** | forme des chaînes : deux groupes fusionnent dès qu'un de leurs points se touche. Sensible au bruit |
| `complete` (complète) | distance entre leurs deux points **les plus éloignés** | groupes compacts |
| `average` (moyenne) | moyenne des distances entre toutes les paires de points | compromis entre les deux |
| `ward` | **augmentation d'inertie** que provoquerait la fusion | groupes compacts et de tailles équilibrées, proche de K-Means ; **le plus utilisé** |

### 7.3 Avec scipy et scikit-learn

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

Z = linkage(X_scaled, method="ward")          # calcule toutes les fusions successives
dendrogram(Z, truncate_mode="lastp", p=30)   # dessine l'arbre (seulement les 30 dernières fusions, pour la lisibilité)

cah = AgglomerativeClustering(n_clusters=5, linkage="ward")
labels_cah = cah.fit_predict(X_scaled)       # coupe l'arbre pour obtenir 5 groupes
```

- **Avantages** : on choisit `k` **après** avoir vu la structure des données dans le dendrogramme ; pas de hasard, donc le résultat est toujours le même.
- **Inconvénient** : il faut calculer et stocker les distances entre toutes les paires d'individus, soit environ `n²/2` nombres. Pour 100 000 individus, cela fait 5 milliards de distances : la CAH est réservée aux jeux de données de taille modeste.

## À retenir

- Le clustering cherche des groupes **sans étiquette** ; il n'y a pas de vérité unique, on juge sur la cohérence et l'utilité.
- Tout repose sur une **distance** : **standardisez** toujours les variables pour qu'aucune n'écrase les autres à cause de son unité.
- K-Means alterne **affectation** au centroïde le plus proche et **mise à jour** des centroïdes ; il minimise l'**inertie** mais peut s'arrêter sur un optimum local (d'où `n_init`).
- Le nombre de groupes `k` se choisit avec le **coude**, la **silhouette** et le bon sens métier.
- CAH et DBSCAN sont des alternatives quand les groupes ne sont pas « ronds » ou qu'on veut explorer la structure.
- Un bon clustering est un clustering **qu'on sait expliquer**.

## Lexique

| Terme | Définition |
|---|---|
| **ACP** (PCA) | Méthode qui projette des données à nombreuses variables sur quelques axes en conservant le plus possible leur dispersion ; utile pour visualiser. |
| **Centroïde** | Point moyen d'un cluster : ses coordonnées sont les moyennes des variables des membres du groupe. |
| **Cluster** | Groupe d'individus qui se ressemblent, produit par un algorithme de clustering. |
| **Clustering** | Répartition automatique des individus en groupes homogènes, sans étiquette connue. |
| **Convergence** | Moment où un algorithme itératif s'arrête parce que ses résultats ne changent plus. |
| **Dendrogramme** | Arbre qui représente les fusions successives de la CAH. |
| **Dimension** | Nombre de variables décrivant chaque individu (un individu est un point dans un espace à `p` dimensions). |
| **Distance euclidienne** | Distance « à la règle » entre deux points, calculée avec le théorème de Pythagore. |
| **Écart-type** | Mesure de la dispersion d'une variable : l'écart typique entre une valeur et la moyenne. |
| **Étiquette** (ou cible) | Réponse connue que l'on cherche à prédire en apprentissage supervisé (absente en non supervisé). |
| **Hyperparamètre** | Réglage d'un algorithme fixé par l'utilisateur avant l'apprentissage (ex. : `k` pour K-Means). |
| **Individu** | Une ligne du tableau de données (un client, une machine…). |
| **Inertie** | Somme des carrés des distances de chaque individu à son centroïde ; plus elle est faible, plus les groupes sont compacts. |
| **Liaison** (critère de) | Façon de mesurer la distance entre deux groupes dans la CAH (single, complete, average, ward). |
| **Optimum local** | Solution qu'on ne peut pas améliorer par de petites modifications, mais qui n'est pas forcément la meilleure possible. |
| **Persona** | Portrait-type donnant un nom et un sens métier à un cluster. |
| **Reproductible** | Qui donne le même résultat à chaque exécution (grâce à `random_state`). |
| **Segmentation** | En marketing, découpage d'une clientèle en groupes homogènes. |
| **Silhouette** | Score entre −1 et 1 qui mesure si un individu est proche de son groupe et loin des autres. |
| **Standardisation** | Transformation d'une variable en « nombre d'écarts-types par rapport à la moyenne » (moyenne 0, écart-type 1). |
| **Valeur aberrante** | Individu aux valeurs extrêmes, très différent des autres. |
| **Variable** (feature) | Une colonne du tableau de données (âge, revenu…). |
| **Variance** | Carré de l'écart-type ; mesure la dispersion des données. |

## Pour aller plus loin

- [Guide scikit-learn sur le clustering](https://scikit-learn.org/stable/modules/clustering.html) : comparaison visuelle de tous les algorithmes sur des jeux de données jouets.
- Les modèles de mélange gaussien (`GaussianMixture`) : un clustering « souple », où chaque individu reçoit une **probabilité** d'appartenir à chaque groupe au lieu d'un seul numéro.
