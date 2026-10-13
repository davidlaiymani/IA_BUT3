# TP3 — XGBoost et les méthodes d'ensemble : cours

[⬅️ Retour au sommaire](../README.md) · Énoncé : [TP3_Enonce.ipynb](TP3_Enonce.ipynb)

## Ce que vous saurez faire à la fin de ce TP

- Expliquer avec vos mots pourquoi **combiner beaucoup d'arbres** donne un meilleur modèle qu'un arbre seul.
- Distinguer les deux grandes façons de les combiner : le **bagging** (forêts aléatoires) et le **boosting** (XGBoost).
- Dire **quand** XGBoost est un bon choix… et quand il ne l'est pas.
- L'utiliser **correctement** : trois jeux de données, taux d'apprentissage, **arrêt anticipé**.
- Tenir compte de classes déséquilibrées et **comprendre ce que le modèle a appris** (importance des variables, valeurs SHAP).

L'objectif n'est pas de faire de vous des spécialistes d'XGBoost, mais que vous sachiez le sortir de la boîte à outils au bon moment et vous en servir sans vous tromper.

> 📖 Les mots en **gras** sont définis dans le texte ; vous les retrouverez tous dans le [lexique](#lexique) en fin de document.

---

## 0. Ce qu'il faut avoir en tête avant de commencer

Ce TP prolonge directement le TP2. Quatre rappels suffisent.

**Les données tabulaires.** Ce sont des données rangées dans un tableau : une ligne par individu, une colonne par variable, avec des types mélangés (nombres, catégories, dates) et souvent des trous. C'est le quotidien des entreprises : fichiers clients, commandes, capteurs, dossiers administratifs. Par opposition, une image, un texte ou un son ne sont pas des données tabulaires.

**L'arbre de décision.** Un arbre pose une suite de questions simples (« contrat au mois ? », « ancienneté ≤ 12 mois ? ») et range chaque individu dans une **feuille**, qui donne la prédiction. Sa **profondeur** est le nombre maximal de questions posées à la suite.

**Le sur-apprentissage.** Un modèle trop souple finit par apprendre par cœur le jeu d'entraînement, y compris le **bruit**, c'est-à-dire les hasards propres à cet échantillon qui ne se reproduiront pas sur d'autres données. Il est excellent sur les données qu'il a vues, décevant sur les autres. Un arbre profond en est l'exemple type.

**Les hyperparamètres et l'AUC.** Un **hyperparamètre** est un réglage que l'on fixe avant l'apprentissage (la profondeur d'un arbre, par exemple). L'**AUC** est la note de 0,5 (hasard) à 1 (parfait) qui mesure si le modèle classe bien les individus, quel que soit le seuil de décision.

## 1. L'idée de départ : plusieurs avis valent mieux qu'un

### 1.1 Le défaut de l'arbre seul

Un arbre de décision est lisible, mais il a deux défauts :

- peu profond, il est trop simple et se trompe souvent ;
- profond, il sur-apprend, et surtout il est **instable** : retirez quelques dizaines de lignes du jeu d'entraînement et vous obtenez un arbre différent, qui pose d'autres questions.

### 1.2 La sagesse des foules

Demandez à une seule personne d'estimer le nombre de bonbons dans un bocal : elle se trompera sans doute beaucoup. Demandez-le à cent personnes et faites la moyenne : le résultat est souvent étonnamment proche de la réalité, parce que les erreurs des uns compensent celles des autres.

**Exemple chiffré.** Supposons que l'on dispose de modèles qui ont chacun raison 7 fois sur 10, et qui se trompent **indépendamment** les uns des autres. On les fait voter à la majorité :

| Nombre de modèles qui votent | Probabilité que la majorité ait raison |
|---|---|
| 1 | 70 % |
| 3 | 78 % |
| 5 | 84 % |
| 11 | 92 % |
| 101 | plus de 99,9 % |

Avec 3 modèles, la majorité a raison si les trois ont raison (0,7³ = 0,343) ou si exactement deux sur trois ont raison (3 × 0,7² × 0,3 = 0,441), soit 78 % au total.

Le mot important est **indépendamment**. Cent copies du même modèle font cent fois la même erreur : le vote n'apporte rien. Il faut des modèles **différents** les uns des autres.

### 1.3 Les méthodes d'ensemble

Une **méthode d'ensemble** applique cette idée : au lieu d'un seul modèle, on en entraîne beaucoup (des arbres, le plus souvent) et on combine leurs réponses. Il existe deux grandes façons de s'y prendre, qui diffèrent par la manière de fabriquer des arbres différents.

## 2. Deux façons de combiner des arbres

### 2.1 Le bagging : des arbres indépendants qui votent

**L'image : un jury.** Chaque juré a vu une partie du dossier, se forge son opinion seul, puis tout le monde vote.

Pour obtenir des arbres différents à partir d'un seul jeu de données, on utilise le **bootstrap** : on fabrique un nouvel échantillon de même taille en tirant des lignes **au hasard et avec remise**. Certaines lignes sont donc tirées plusieurs fois, d'autres pas du tout.

**Exemple.** À partir des clients {1, 2, 3, 4, 5, 6}, trois tirages possibles :

| Échantillon | Clients tirés | Clients absents |
|---|---|---|
| 1 | 1, 1, 3, 4, 6, 6 | 2, 5 |
| 2 | 2, 3, 3, 4, 5, 5 | 1, 6 |
| 3 | 1, 2, 2, 2, 4, 6 | 3, 5 |

Sur un grand jeu de données, chaque échantillon contient environ 63 % des individus d'origine, les autres places étant occupées par des doublons. Chaque arbre voit donc un jeu un peu différent.

Le **bagging** (*bootstrap aggregating*) consiste à entraîner un arbre par échantillon, puis à les faire **voter** (ou à faire la moyenne de leurs probabilités).

La **forêt aléatoire** (*random forest*) est un bagging d'arbres avec une seconde dose de hasard : à chaque question, l'arbre ne peut choisir que parmi un sous-ensemble de variables tiré au sort. Les arbres se ressemblent encore moins, et leurs erreurs se compensent encore mieux.

```
 données ──┬─► échantillon 1 ─► arbre profond 1 ─┐
           ├─► échantillon 2 ─► arbre profond 2 ─┼─► vote / moyenne
           └─► échantillon B ─► arbre profond B ─┘
           (les arbres sont construits indépendamment, en parallèle)
```

Chaque arbre de la forêt est profond et sur-apprend à sa façon ; c'est la **moyenne** qui gomme leurs excès. La forêt corrige donc surtout l'**instabilité** de l'arbre seul.

### 2.2 Le boosting : des arbres qui se corrigent les uns les autres

**L'image : le golf.** Le premier coup envoie la balle à peu près dans la bonne direction. Le deuxième ne repart pas du départ : il vise à combler **l'écart qui reste** jusqu'au trou. Le troisième corrige l'écart laissé par le deuxième, et ainsi de suite, par coups de plus en plus courts.

Le **boosting** construit les arbres **l'un après l'autre**. Chaque nouvel arbre n'essaie pas de résoudre tout le problème : il apprend à corriger les **erreurs** que commet encore l'ensemble des arbres déjà construits. On utilise de **petits arbres** (peu profonds), chacun très imparfait ; c'est leur accumulation qui donne un modèle puissant.

```
 données ─► arbre 1 ─► erreurs restantes ─► arbre 2 ─► erreurs restantes ─► arbre 3 ─► …
 prédiction finale = somme des contributions de tous les arbres
```

### 2.3 Comparaison

| | Bagging (forêt aléatoire) | Boosting (XGBoost) |
|---|---|---|
| Image | un jury qui vote | un golfeur qui s'approche du trou |
| Construction | arbres indépendants, en parallèle | arbres à la suite, chacun corrige les précédents |
| Arbres | profonds | petits (3 à 8 niveaux) |
| Défaut de l'arbre qu'on corrige | l'instabilité | la simplicité |
| Risque quand on ajoute des arbres | aucun (ça plafonne) | sur-apprentissage si on en met trop |
| Réglage | facile, marche bien par défaut | demande un peu plus de soin |
| Performance typique | très bonne | souvent la meilleure |

## 3. Le gradient boosting, déroulé à la main

Le **gradient boosting** est la forme de boosting utilisée aujourd'hui. Pour bien voir le mécanisme, prenons un problème de régression minuscule : prédire le prix de 4 maisons à partir de leur surface.

| Maison | Surface (m²) | Prix réel (k€) |
|---|---|---|
| A | 40 | 100 |
| B | 60 | 140 |
| C | 100 | 260 |
| D | 120 | 300 |

**Étape 0 : une première prédiction grossière.** On prédit la même chose pour tout le monde : le prix moyen, soit 200 k€.

**Étape 1 : mesurer ce qui reste à expliquer.** Pour chaque maison, on calcule le **résidu** : prix réel − prédiction. C'est l'erreur qu'il reste à corriger.

| Maison | Prix réel | Prédiction | Résidu |
|---|---|---|---|
| A | 100 | 200 | −100 |
| B | 140 | 200 | −60 |
| C | 260 | 200 | +60 |
| D | 300 | 200 | +100 |

**Étape 2 : entraîner un petit arbre sur les résidus.** L'arbre ne cherche pas à prédire le prix, mais le résidu. Avec une seule question, la meilleure est « surface ≤ 80 m² ? ». Chaque feuille prédit la moyenne des résidus qu'elle contient :

```
            surface ≤ 80 m² ?
             /            \
           oui            non
       (A et B)         (C et D)
     résidu moyen     résidu moyen
        −80              +80
```

**Étape 3 : corriger, mais prudemment.** On n'ajoute pas toute la correction proposée par l'arbre, seulement une fraction, fixée par le **taux d'apprentissage** (*learning rate*). Prenons 0,5 : on ajoute 0,5 × (−80) = −40 aux petites maisons et +40 aux grandes.

| Maison | Prix réel | Nouvelle prédiction | Nouveau résidu |
|---|---|---|---|
| A | 100 | 200 − 40 = 160 | −60 |
| B | 140 | 160 | −20 |
| C | 260 | 200 + 40 = 240 | +20 |
| D | 300 | 240 | +60 |

**Étape 4 : recommencer.** On entraîne un deuxième arbre sur les **nouveaux** résidus (il prédit −40 pour les petites maisons, +40 pour les grandes), on ajoute la moitié de sa correction, etc.

| | Prédictions (A, B, C, D) | Erreur moyenne |
|---|---|---|
| Départ (moyenne) | 200, 200, 200, 200 | 80 k€ |
| Après l'arbre 1 | 160, 160, 240, 240 | 40 k€ |
| Après l'arbre 2 | 140, 140, 260, 260 | 20 k€ |
| Après l'arbre 3, 4, … | de plus en plus près des vrais prix | → 0 |

À chaque arbre, l'erreur diminue. Le troisième arbre posera une autre question (par exemple « surface ≤ 50 m² ? ») pour distinguer A de B, ce que les deux premiers ne faisaient pas.

**La prédiction finale** d'une maison est donc : la moyenne de départ + la somme des (petites) corrections de tous les arbres.

### Pourquoi ne pas corriger d'un seul coup ?

Dans cet exemple, on irait plus vite avec un taux d'apprentissage de 1. Mais sur de vraies données, les résidus contiennent une part de **bruit**. Un arbre qui corrige tout, tout de suite, apprend aussi ce bruit. En avançant par petits pas, chaque arbre n'a qu'une influence limitée, et ses erreurs sont rattrapées par les suivants. C'est le golfeur qui préfère trois coups mesurés à un grand coup hasardeux.

La contrepartie est mécanique : **plus les pas sont petits, plus il faut d'arbres** pour arriver au but.

### Et pour une classification ?

Le principe est exactement le même. La seule différence : les arbres additionnent leurs corrections sur un **score**, que la fonction sigmoïde vue au TP2 transforme en probabilité (un score de 0 donne 0,5 ; un score très positif donne une probabilité proche de 1).

L'erreur à faire baisser n'est plus un écart de prix mais la **log loss** (*perte logarithmique*), qui note la qualité des probabilités : pour chaque individu, on prend −ln(probabilité que le modèle donnait à la bonne réponse), et on fait la moyenne. Plus elle est **basse**, mieux c'est.

| Le modèle donnait à la bonne réponse une probabilité de… | Perte pour cet individu |
|---|---|
| 0,9 (sûr de lui, et il a raison) | 0,11 |
| 0,6 (hésitant) | 0,51 |
| 0,1 (sûr de lui, et il a tort) | 2,30 |

La log loss punit donc sévèrement les erreurs commises avec assurance. C'est elle que vous verrez baisser d'arbre en arbre pendant le TP.

> 💡 **D'où vient le mot « gradient » ?** En mathématiques, le gradient indique dans quelle direction modifier quelque chose pour faire baisser une erreur. Les résidus jouent exactement ce rôle : ils disent à chaque arbre dans quel sens corriger les prédictions. Vous n'avez pas besoin d'en savoir plus pour bien utiliser la méthode.

## 4. XGBoost

**XGBoost** (*eXtreme Gradient Boosting*, 2014) est une bibliothèque qui implémente le gradient boosting de façon très efficace. Elle s'est fait connaître en gagnant de très nombreuses compétitions de science des données, et elle est aujourd'hui un standard dans l'industrie. Ce qu'elle apporte concrètement :

| Atout | Ce que cela change pour vous |
|---|---|
| **Valeurs manquantes** gérées nativement | Inutile de les remplacer : à chaque question, l'arbre apprend de quel côté envoyer les individus dont la valeur est inconnue. |
| **Variables catégorielles** gérées nativement | Inutile de faire un encodage *one-hot* : il suffit de convertir les colonnes de texte en type `category` de pandas. |
| Pas de **standardisation** | Comme tous les arbres, il compare des valeurs à des seuils : l'unité des variables n'a aucune importance. |
| **Rapidité** | Des centaines d'arbres en quelques secondes sur des dizaines de milliers de lignes. |
| **Régularisation** intégrée | Des freins internes qui pénalisent les arbres trop compliqués et limitent le sur-apprentissage. |
| **Arrêt anticipé** | Il sait s'arrêter tout seul au bon nombre d'arbres (section 6.3). |

Résultat : là où le TP2 demandait un `Pipeline` avec imputation, encodage et standardisation, XGBoost s'utilise presque directement sur le DataFrame.

```python
from xgboost import XGBClassifier

X[colonnes_texte] = X[colonnes_texte].astype("category")   # seule préparation nécessaire

modele = XGBClassifier(enable_categorical=True, random_state=42)
modele.fit(X_train, y_train)
modele.predict_proba(X_valid)[:, 1]     # même interface que scikit-learn
```

XGBoost n'est pas seul : **LightGBM**, **CatBoost** et le `HistGradientBoostingClassifier` de scikit-learn reposent sur le même principe et donnent des résultats très proches. Ce que vous apprenez ici vaut pour tous.

## 5. Quand utiliser XGBoost ?

| Situation | XGBoost est-il un bon choix ? |
|---|---|
| Données **tabulaires**, de quelques milliers à quelques millions de lignes | ✅ Oui, c'est le modèle à essayer en priorité. |
| Types mélangés, valeurs manquantes, relations compliquées entre variables | ✅ Oui, il les gère sans préparation. |
| On cherche la **meilleure performance** possible | ✅ Oui, il est très souvent en tête sur ce type de données. |
| Très peu de données (quelques centaines de lignes) | ⚠️ Un modèle simple (régression logistique, petit arbre) fait souvent aussi bien et sur-apprend moins. |
| Le modèle doit être **lisible** par un non-spécialiste (règlement, audit, décision à justifier) | ⚠️ Préférer une régression logistique ou un petit arbre : des centaines d'arbres ne se lisent pas. |
| Images, texte, son | ❌ Non : c'est le domaine des réseaux de neurones. |
| Prédire **au-delà** de ce qui a été observé (une tendance qui continue de monter) | ❌ Non : un arbre ne sait répondre que par des valeurs déjà vues à l'entraînement. |

Deux règles de bon sens complètent ce tableau :

1. **Toujours commencer par une référence simple** (`DummyClassifier`, régression logistique). Si XGBoost ne gagne que quelques millièmes d'AUC, le modèle simple, plus rapide et plus lisible, est peut-être le meilleur choix.
2. **Un bon modèle ne rattrape pas de mauvaises données.** Le temps passé à comprendre et à nettoyer les variables rapporte presque toujours plus que le temps passé à régler le modèle.

## 6. Bien utiliser XGBoost

### 6.1 Trois jeux de données au lieu de deux

En boosting, il faut décider **quand arrêter** d'ajouter des arbres. Cette décision se prend en observant les performances sur des données que le modèle n'a pas apprises. Mais si l'on utilise le jeu de test pour cela, il ne peut plus servir à l'évaluation finale : on l'aurait « consulté » pendant la mise au point. On découpe donc en **trois** :

```
 ┌───────────────────────── données ──────────────────────────┐
 │  entraînement (60 %)   │  validation (20 %)  │  test (20 %) │
 └────────────────────────┴─────────────────────┴──────────────┘
    construire les arbres    décider quand         la note finale,
                             s'arrêter, comparer   une seule fois
                             les réglages
```

L'image de l'étudiant du TP2 se prolonge : le jeu d'**entraînement**, ce sont les exercices du cours ; le jeu de **validation**, les examens blancs, que l'on peut repasser pour ajuster sa méthode de travail ; le jeu de **test**, l'examen final, que l'on ne passe qu'une fois.

```python
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25,
                                                      stratify=y_temp, random_state=42)
# 0,25 × 80 % = 20 % : on obtient bien 60 % / 20 % / 20 %
```

### 6.2 Le couple taux d'apprentissage / nombre d'arbres

Deux hyperparamètres vont ensemble :

- `learning_rate` : la taille des pas (la fraction de chaque correction que l'on garde) ;
- `n_estimators` : le nombre d'arbres, donc le nombre de pas.

| `learning_rate` | Comportement |
|---|---|
| grand (0,3, la valeur par défaut) | on arrive vite, avec peu d'arbres, mais de façon un peu brutale |
| moyen (0,05 à 0,1) | bon compromis pour la mise au point |
| petit (0,01 à 0,03) | plus précis, mais il faut beaucoup plus d'arbres et de temps de calcul |

Diviser le taux d'apprentissage par 10 multiplie à peu près par 10 le nombre d'arbres nécessaires. Il est donc inutile de chercher le bon `n_estimators` à la main : on laisse l'arrêt anticipé le trouver.

### 6.3 L'arrêt anticipé

Pendant l'entraînement, XGBoost peut mesurer la log loss sur le jeu de validation après chaque nouvel arbre. Voici ce que l'on observe typiquement :

| Nombre d'arbres | Log loss entraînement | Log loss validation |
|---|---|---|
| 50 | 0,330 | 0,340 |
| 100 | 0,280 | 0,300 |
| 150 | 0,255 | 0,285 |
| **200** | 0,240 | **0,280** |
| 250 | 0,228 | 0,281 |
| 300 | 0,217 | 0,283 |

```
 log loss
   │ ●
   │  ●  ○
   │    ●   ○
   │       ●     ○      ○       ○   ○   ← validation : baisse, puis remonte
   │           ●
   │                ●
   │                      ●     ●       ← entraînement : baisse toujours
   └──────────────┬───────────────────► nombre d'arbres
            minimum de validation :
             c'est ici qu'il faut s'arrêter
```

La perte d'entraînement baisse **toujours** : c'est normal, chaque arbre corrige les erreurs sur ces données-là. La perte de validation, elle, baisse puis **remonte** : à partir de 200 arbres, les nouveaux arbres n'apprennent plus que le bruit du jeu d'entraînement. C'est le sur-apprentissage, vu cette fois en fonction du nombre d'arbres.

L'**arrêt anticipé** (*early stopping*) automatise la décision : on demande volontairement beaucoup trop d'arbres, et XGBoost s'arrête dès que la perte de validation n'a plus progressé depuis un certain nombre d'arbres (la **patience**, réglée par `early_stopping_rounds`). Il conserve le modèle au meilleur moment.

```python
modele = XGBClassifier(
    n_estimators=3000,          # volontairement trop grand
    learning_rate=0.05,
    early_stopping_rounds=50,   # patience : arrêt si aucun progrès pendant 50 arbres
    eval_metric="logloss",      # la mesure surveillée
    enable_categorical=True,
    random_state=42,
)
modele.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
modele.best_iteration           # numéro du meilleur arbre (le premier porte le numéro 0)
```

`eval_set` désigne les données sur lesquelles surveiller la perte : le jeu de **validation**, jamais le jeu de test.

### 6.4 Les autres réglages utiles

| Hyperparamètre | Ce qu'il règle | Valeurs usuelles | En cas de sur-apprentissage |
|---|---|---|---|
| `max_depth` | profondeur de chaque arbre, donc sa complexité | 3 à 8 (6 par défaut) | diminuer |
| `subsample` | part des lignes tirées au hasard pour chaque arbre | 0,6 à 1 | diminuer |
| `colsample_bytree` | part des colonnes tirées au hasard pour chaque arbre | 0,5 à 1 | diminuer |
| `min_child_weight` | taille minimale d'une feuille (en gros, le nombre d'individus) | 1 à 10 | augmenter |

`subsample` et `colsample_bytree` reprennent l'idée de la forêt aléatoire : un peu de hasard rend les arbres plus différents les uns des autres.

Pour trouver une bonne combinaison, on utilise une recherche par validation croisée, comme au TP2 (`GridSearchCV`, ou `RandomizedSearchCV` qui teste des combinaisons tirées au hasard, ce qui est plus rapide).

> 💡 **Ne passez pas vos journées à régler.** XGBoost est robuste : dans une large zone de réglages raisonnables, les performances varient très peu. Les réglages par défaut, plus un taux d'apprentissage modéré et l'arrêt anticipé, donnent déjà l'essentiel du résultat. Vous le vérifierez dans la partie facultative du TP.

### 6.5 La recette en résumé

1. Convertir les colonnes de texte en `category`, découper en entraînement / validation / test.
2. Entraîner une **référence** simple pour savoir d'où l'on part.
3. Entraîner XGBoost avec `learning_rate` autour de 0,05 à 0,1, un grand `n_estimators` et l'**arrêt anticipé** sur le jeu de validation.
4. Si le temps le permet, ajuster `max_depth`, `subsample`, `colsample_bytree`.
5. Évaluer **une seule fois** sur le jeu de test.

## 7. Quand une classe est rare

Fraude, panne, résiliation : la classe qui nous intéresse (la classe positive) est souvent minoritaire. Le modèle, qui cherche à se tromper le moins possible en moyenne, a alors tendance à la négliger. Trois leviers, déjà croisés au TP2 :

1. **Choisir une métrique adaptée.** L'exactitude est trompeuse (voir TP2) ; on regarde l'AUC, la précision, le rappel, le F1.
2. **Donner plus de poids aux positifs** avec `scale_pos_weight`. La valeur habituelle est le rapport nombre de négatifs / nombre de positifs : avec 900 négatifs et 100 positifs, `scale_pos_weight = 9`, et rater un positif « coûte » alors au modèle autant que rater 9 négatifs. Effet : le modèle prédit « positif » plus souvent, le **rappel monte**, la **précision baisse**.
3. **Baisser le seuil de décision** (0,5 par défaut) : même effet, obtenu sans réentraîner le modèle.

> ⚠️ Ces deux derniers leviers **déplacent le compromis** entre précision et rappel ; ils ne rendent pas le modèle plus doué pour distinguer les deux classes (l'AUC ne bouge presque pas). Le bon réglage dépend du coût des erreurs pour le métier : vaut-il mieux une fausse alerte ou un cas manqué ?

## 8. Comprendre ce que le modèle a appris

Un arbre seul se lit ; plusieurs centaines d'arbres, non. Deux outils permettent malgré tout d'ouvrir la boîte.

### 8.1 L'importance des variables

XGBoost attribue à chaque variable un score d'importance. Il existe plusieurs façons de le calculer, dont deux à connaître :

- le **gain** : de combien les questions posées sur cette variable ont, en moyenne, fait baisser l'erreur. C'est la mesure la plus pertinente ;
- le **nombre d'utilisations** (`weight`) : combien de fois la variable a servi à poser une question. Cette mesure favorise les variables numériques comme l'âge, qui offrent beaucoup de seuils possibles et sont donc souvent utilisées, même quand chaque question apporte peu.

Les deux classements peuvent être très différents : précisez toujours lequel vous utilisez.

L'importance dit **quelles** variables comptent, pas **dans quel sens** ni **pour qui**.

### 8.2 Les valeurs SHAP

Les **valeurs SHAP** répondent à une question plus précise : *pour cette personne-là, pourquoi le modèle a-t-il donné cette prédiction ?* Elles décomposent chaque prédiction en une somme de contributions, une par variable.

**Exemple.** Le modèle prédit si le revenu d'une personne dépasse 50 000 $. Rappelez-vous (section 3) qu'il calcule un **score**, transformé ensuite en probabilité. On part de la **valeur de base**, le score donné à quelqu'un dont on ne sait rien, puis chaque variable pousse le score vers le haut ou vers le bas :

| | Contribution | Score cumulé |
|---|---|---|
| Valeur de base | | −1,2 (probabilité 0,23) |
| Situation familiale : marié | +0,9 | −0,3 |
| Âge : 45 ans | +0,3 | 0,0 |
| Niveau d'études : court | −0,5 | −0,5 |
| Temps de travail : 40 h | +0,1 | −0,4 |
| **Prédiction** | | **−0,4 (probabilité 0,40)** |

On lit directement que le fait d'être marié tire fortement la prédiction vers « haut revenu », et que le niveau d'études la tire dans l'autre sens. Pour une autre personne, les contributions seraient différentes.

En faisant la moyenne des contributions **en valeur absolue** sur tous les individus, on obtient une importance globale de chaque variable, généralement plus fiable que le gain.

```python
import xgboost as xgb

donnees = xgb.DMatrix(X_valid, enable_categorical=True)     # format interne d'XGBoost
contributions = modele.get_booster().predict(donnees, pred_contribs=True)
# une ligne par individu, une colonne par variable, plus une dernière colonne : la valeur de base
```

> ⚠️ Importance et SHAP décrivent **le modèle**, pas la réalité. Une variable importante est **associée** à la cible dans les données ; cela ne prouve pas qu'elle en est la **cause**. Et si les données contiennent des inégalités (entre hommes et femmes, par exemple), le modèle les apprend et les reproduit.

## 9. Les erreurs classiques

| Erreur | Pourquoi c'est un problème |
|---|---|
| Utiliser le jeu de **test** pour l'arrêt anticipé ou les réglages | La note finale est faussée : elle est trop optimiste. |
| Fixer `n_estimators` au hasard, sans arrêt anticipé | Trop peu d'arbres : modèle inachevé. Trop : sur-apprentissage. |
| Baisser `learning_rate` sans augmenter `n_estimators` | Le modèle s'arrête avant d'être arrivé au but. |
| Oublier de convertir les colonnes de texte en `category` | XGBoost refuse les colonnes de type texte. |
| Juger le modèle sur l'exactitude alors que les classes sont déséquilibrées | Un modèle inutile peut afficher un très bon score. |
| Ne pas entraîner de modèle de référence | Impossible de savoir si XGBoost apporte vraiment quelque chose. |
| Conclure qu'une variable importante est une cause | Le modèle ne voit que des associations. |

## À retenir

- Combiner beaucoup d'arbres différents donne un modèle bien meilleur qu'un arbre seul : c'est le principe des **méthodes d'ensemble**.
- **Bagging** (forêt aléatoire) : des arbres indépendants qui votent. **Boosting** (XGBoost) : de petits arbres construits à la suite, chacun corrigeant les erreurs des précédents.
- Sur des **données tabulaires**, le gradient boosting est le modèle à essayer en priorité, après une référence simple. Il ne convient ni aux images, ni au texte, ni aux très petits jeux de données.
- XGBoost gère seul les catégories et les valeurs manquantes : presque aucune préparation.
- `learning_rate` et `n_estimators` vont ensemble : prenez de petits pas, demandez beaucoup d'arbres et laissez l'**arrêt anticipé** s'arrêter au bon moment, sur un jeu de **validation** distinct du jeu de test.
- Le réglage fin des autres hyperparamètres rapporte peu ; les données et la rigueur de l'évaluation rapportent beaucoup.
- Pour comprendre le modèle : importance par **gain** et valeurs **SHAP**, sans les confondre avec des causes.

## Lexique

| Terme | Définition |
|---|---|
| **Arrêt anticipé** (*early stopping*) | Arrêt automatique de l'entraînement quand la perte sur le jeu de validation ne progresse plus. |
| **Bagging** | Méthode d'ensemble : des modèles entraînés indépendamment sur des échantillons bootstrap, puis combinés par vote ou moyenne. |
| **Boosting** | Méthode d'ensemble : des modèles construits l'un après l'autre, chacun corrigeant les erreurs des précédents. |
| **Bootstrap** | Échantillon de même taille que les données, obtenu par tirage au hasard avec remise. |
| **Bruit** | Hasards propres à un échantillon de données, qui ne se reproduisent pas sur d'autres données. |
| **Données tabulaires** | Données organisées en tableau : une ligne par individu, une colonne par variable. |
| **Forêt aléatoire** | Bagging d'arbres de décision, avec en plus un tirage au hasard des variables à chaque question. |
| **Gain** | Mesure d'importance d'une variable : baisse moyenne de l'erreur apportée par les questions posées sur elle. |
| **Gradient boosting** | Boosting dans lequel chaque arbre apprend les résidus (l'erreur restante) des arbres précédents. |
| **Jeu de validation** | Données mises de côté pour décider quand arrêter l'entraînement et comparer des réglages ; distinct du jeu de test. |
| **Log loss** | Mesure d'erreur d'un classifieur qui note la qualité des probabilités ; plus elle est basse, mieux c'est. |
| **Méthode d'ensemble** | Modèle qui combine les réponses de nombreux modèles plus simples. |
| **Patience** | Nombre d'arbres sans progrès que l'on tolère avant l'arrêt anticipé (`early_stopping_rounds`). |
| **Régularisation** | Tout mécanisme qui freine la complexité d'un modèle pour limiter le sur-apprentissage. |
| **Résidu** | Écart entre la valeur réelle et la prédiction actuelle : ce qu'il reste à expliquer. |
| **Score** | Nombre calculé par le modèle avant d'être transformé en probabilité par la fonction sigmoïde. |
| **Taux d'apprentissage** (*learning rate*) | Fraction de la correction de chaque arbre que l'on ajoute au modèle ; la taille des pas. |
| **Valeur de base** | Score donné par le modèle à un individu dont on ne sait rien ; point de départ de la décomposition SHAP. |
| **Valeurs SHAP** | Décomposition d'une prédiction en contributions de chaque variable, individu par individu. |

## Pour aller plus loin

- [Documentation XGBoost](https://xgboost.readthedocs.io/) : guide des paramètres et tutoriel sur les variables catégorielles.
- [LightGBM](https://lightgbm.readthedocs.io/) et [CatBoost](https://catboost.ai/) : les deux grandes alternatives.
- [`HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/ensemble.html) de scikit-learn : un gradient boosting très proche, sans bibliothèque supplémentaire.
- [Bibliothèque SHAP](https://shap.readthedocs.io/) : de très bonnes visualisations des contributions (`shap.summary_plot`, `shap.plots.waterfall`).
