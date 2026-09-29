# TP2 — Classification supervisée : cours

[⬅️ Retour au sommaire](../README.md) · Énoncé : [TP2_Enonce.ipynb](TP2_Enonce.ipynb)

## Ce que vous saurez faire à la fin de ce TP

- Formuler un problème de **classification** : variables explicatives, cible, jeu d'entraînement et de test.
- Préparer des données hétérogènes (valeurs manquantes, variables catégorielles) avec un **`Pipeline`** scikit-learn.
- Entraîner et comparer une **régression logistique**, des **k plus proches voisins** et un **arbre de décision**.
- Choisir et interpréter les bonnes **métriques** : matrice de confusion, précision, rappel, F1, courbe ROC.
- Détecter le **sur-apprentissage** et régler un hyperparamètre par **validation croisée**.

---

## 1. Le problème de classification

On dispose de `n` exemples décrits par des **variables explicatives** (*features*) `X` et d'une **cible** `y` connue qui prend un nombre fini de valeurs (les **classes**). On cherche une fonction `f` telle que `f(X) ≈ y`, et surtout qui se trompe peu sur des exemples **qu'elle n'a jamais vus**.

| Exemple | `X` | `y` |
|---|---|---|
| Filtre anti-spam | mots du message, expéditeur | spam / non spam |
| Diagnostic | analyses sanguines | malade / sain |
| Attrition des clients (ce TP) | contrat, ancienneté, facture, services… | résilie / reste |
| Reconnaissance de chiffres | pixels de l'image | 0, 1, …, 9 |

Avec deux classes, on parle de **classification binaire** ; on appelle souvent **classe positive** (codée 1) celle qui nous intéresse.

> Si `y` est une quantité continue (un prix, une température), c'est un problème de **régression** : voir le TP4.

## 2. La démarche en scikit-learn

Tous les modèles de scikit-learn partagent la même interface :

```python
modele = UnModele(hyperparametres)   # 1. choisir le modèle et ses réglages
modele.fit(X_train, y_train)         # 2. apprendre sur les données d'entraînement
y_pred = modele.predict(X_test)      # 3. prédire la classe
proba  = modele.predict_proba(X_test)[:, 1]   # ou la probabilité de la classe 1
modele.score(X_test, y_test)         # 4. évaluer (exactitude par défaut)
```

### 2.1 Séparer entraînement et test

Évaluer un modèle sur les données qui ont servi à l'entraîner revient à donner à un étudiant les questions de l'examen à l'avance : le score ne mesure que sa mémoire. On réserve donc une partie des données (typiquement 20 à 30 %) comme **jeu de test**, qu'on ne touche qu'à la toute fin.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

`stratify=y` garantit la même proportion de chaque classe dans les deux jeux.

### 2.2 Toujours commencer par une référence (*baseline*)

Un modèle qui prédit toujours la classe majoritaire atteint déjà une exactitude égale à la proportion de cette classe. Dans ce TP, 73 % des clients restent : « personne ne part » donne déjà **73 %** d'exactitude… et ne repère aucun départ. Un modèle n'a d'intérêt que s'il fait nettement mieux.

```python
from sklearn.dummy import DummyClassifier
DummyClassifier(strategy="most_frequent").fit(X_train, y_train).score(X_test, y_test)
```

## 3. Préparer les données

Les modèles scikit-learn n'acceptent que des **nombres**, **sans valeurs manquantes**.

| Problème | Solution | Classe scikit-learn |
|---|---|---|
| Valeurs manquantes numériques | remplacer par la médiane | `SimpleImputer(strategy="median")` |
| Valeurs manquantes catégorielles | remplacer par la modalité la plus fréquente | `SimpleImputer(strategy="most_frequent")` |
| Variable catégorielle (`"male"`, `"female"`) | une colonne binaire par modalité (*one-hot*) | `OneHotEncoder(handle_unknown="ignore")` |
| Échelles différentes (âge vs prix) | standardiser | `StandardScaler()` |

### 3.1 Pourquoi un `Pipeline` ?

Les paramètres de préparation (médiane d'une colonne, moyenne et écart-type pour la standardisation…) doivent être **appris sur le jeu d'entraînement uniquement**, puis appliqués tels quels au jeu de test. Les calculer sur toutes les données provoque une **fuite de données** (*data leakage*) : de l'information du test s'infiltre dans l'apprentissage et l'évaluation devient trop optimiste.

Le `Pipeline` enchaîne préparation et modèle en un seul objet qui respecte automatiquement cette règle, y compris pendant la validation croisée.

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

num_cols = ["tenure", "MonthlyCharges"]
cat_cols = ["Contract", "InternetService"]

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), cat_cols),
])

modele = Pipeline([("prep", preprocess), ("clf", LogisticRegression())])
modele.fit(X_train, y_train)      # impute, encode, standardise puis entraîne
modele.predict(X_test)            # applique les mêmes transformations au test
```

## 4. Trois modèles de classification

### 4.1 La régression logistique

Malgré son nom, c'est un modèle de **classification**. Il calcule une combinaison linéaire des variables, puis la transforme en probabilité avec la fonction **sigmoïde** :

$$P(y = 1 \mid x) = \sigma(w_0 + w_1 x_1 + \dots + w_p x_p) \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

```
 P(y=1)
   1 ┤                 ●●●●●
     │              ●●
 0.5 ┤ - - - - - - ●  - - - -   seuil de décision
     │          ●●
   0 ┤ ●●●●●●●●
     └────────────┼──────────►  z = w·x
                  0
```

On prédit la classe 1 si la probabilité dépasse 0,5. Les coefficients `w_j` s'interprètent : un coefficient positif augmente la probabilité de la classe 1. Le modèle est rapide, robuste et **interprétable** : c'est une excellente référence.

### 4.2 Les k plus proches voisins (k-NN)

Pour classer un nouveau point, on cherche les `k` exemples d'entraînement les plus proches et on vote à la majorité. Il n'y a pas vraiment d'apprentissage : le modèle mémorise les données.

- `k` petit : frontière très irrégulière, sensible au bruit (sur-apprentissage).
- `k` grand : frontière très lisse, qui finit par ignorer les détails (sous-apprentissage).
- Fondé sur des distances : **standardisation indispensable** (voir TP1).

### 4.3 L'arbre de décision

L'arbre pose une suite de questions binaires sur les variables, choisies pour séparer au mieux les classes :

```
               Contrat au mois ?
                /              \
             non                oui
              |                  |
        reste (7 % de      Fibre optique ?
          départs)           /         \
                          non           oui
                           |             |
                      28 % de       Ancienneté ≤ 12 mois ?
                      départs         /           \
                                    oui            non
                              part (70 %)     43 % de départs
```

À chaque nœud, l'algorithme choisit la question qui rend les deux sous-groupes les plus **purs** possible (critère de Gini ou d'entropie).

- ✅ Très lisible, aucune standardisation nécessaire, gère les interactions entre variables.
- ❌ Un arbre profond apprend le jeu d'entraînement par cœur : il faut limiter sa profondeur (`max_depth`) ou le nombre minimal d'exemples par feuille (`min_samples_leaf`).

Les arbres sont la brique de base des **forêts aléatoires** et du **gradient boosting** (TP3).

## 5. Évaluer un classifieur

### 5.1 La matrice de confusion

|  | prédit 0 | prédit 1 |
|---|---|---|
| **réel 0** | VN (vrais négatifs) | FP (faux positifs) |
| **réel 1** | FN (faux négatifs) | VP (vrais positifs) |

### 5.2 Les métriques dérivées

| Métrique | Formule | Question posée |
|---|---|---|
| Exactitude (*accuracy*) | (VP + VN) / total | Quelle part des prédictions est correcte ? |
| Précision | VP / (VP + FP) | Quand le modèle dit « positif », a-t-il raison ? |
| Rappel (*recall*, sensibilité) | VP / (VP + FN) | Quelle part des vrais positifs le modèle retrouve-t-il ? |
| F1 | 2 · P · R / (P + R) | Compromis entre précision et rappel |

> ⚠️ L'exactitude est trompeuse quand les classes sont **déséquilibrées** : pour une maladie qui touche 1 % des patients, un modèle qui répond toujours « sain » a 99 % d'exactitude et un rappel nul.

Le choix de la métrique dépend du **coût des erreurs** :

- **dépistage médical** : rater un malade (FN) est grave → privilégier le **rappel** ;
- **filtre anti-spam** : envoyer un vrai message à la corbeille (FP) est gênant → privilégier la **précision**.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
ConfusionMatrixDisplay.from_estimator(modele, X_test, y_test)
print(classification_report(y_test, y_pred))
```

### 5.3 Seuil de décision et courbe ROC

Le seuil de 0,5 n'a rien d'obligatoire. **Baisser le seuil** fait prédire « positif » plus souvent : le rappel augmente, la précision baisse. La **courbe ROC** trace le taux de vrais positifs (rappel) en fonction du taux de faux positifs pour tous les seuils possibles.

L'**aire sous la courbe** (AUC) résume la qualité du classement indépendamment du seuil : 0,5 = hasard, 1 = parfait. On peut l'interpréter comme la probabilité qu'un positif tiré au hasard reçoive un score plus élevé qu'un négatif tiré au hasard.

```python
from sklearn.metrics import roc_auc_score, RocCurveDisplay
roc_auc_score(y_test, modele.predict_proba(X_test)[:, 1])
RocCurveDisplay.from_estimator(modele, X_test, y_test)
```

## 6. Sur-apprentissage et validation croisée

### 6.1 Sous-apprentissage et sur-apprentissage

```
 erreur
   │ ●                                   ● erreur sur le test
   │  ●                               ●
   │   ●                          ●
   │     ●                    ●
   │        ●  ●   ●   ●  ●
   │ ○
   │   ○  ○
   │         ○   ○
   │                 ○   ○   ○           ○ erreur sur l'entraînement
   │                            ○   ○   ○
   └────────────────┬─────────────────────► complexité du modèle
   sous-apprentissage│    sur-apprentissage
               complexité idéale
```

- **Sous-apprentissage** : le modèle est trop simple, il se trompe même sur l'entraînement.
- **Sur-apprentissage** : le modèle colle au bruit du jeu d'entraînement (score train excellent) mais généralise mal (score test nettement plus bas).

Le symptôme à surveiller : **un grand écart entre le score d'entraînement et le score de test**.

### 6.2 La validation croisée

Un seul découpage train/test donne une estimation bruitée, surtout sur un petit jeu de données. La **validation croisée à `k` plis** découpe le jeu d'entraînement en `k` parts, entraîne `k` fois le modèle en gardant à chaque fois une part différente pour l'évaluation, puis fait la moyenne.

```
 Pli 1 : [TEST][    ][    ][    ][    ]
 Pli 2 : [    ][TEST][    ][    ][    ]
 Pli 3 : [    ][    ][TEST][    ][    ]      score = moyenne des 5 scores
 Pli 4 : [    ][    ][    ][TEST][    ]
 Pli 5 : [    ][    ][    ][    ][TEST]
```

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(modele, X_train, y_train, cv=5, scoring="accuracy")
scores.mean(), scores.std()
```

### 6.3 Régler les hyperparamètres

Les **hyperparamètres** (`k` des k-NN, `max_depth` de l'arbre…) ne sont pas appris par `fit` : c'est à nous de les choisir. On compare plusieurs valeurs **par validation croisée sur le jeu d'entraînement**, jamais sur le jeu de test (sinon on sur-apprend… le jeu de test).

```python
from sklearn.model_selection import GridSearchCV

grille = {"clf__max_depth": [2, 3, 4, 5, 6, 8, None]}   # « étape__paramètre »
recherche = GridSearchCV(modele, grille, cv=5, scoring="accuracy")
recherche.fit(X_train, y_train)
recherche.best_params_, recherche.best_score_
recherche.score(X_test, y_test)    # évaluation finale, une seule fois
```

## À retenir

- Séparez **train / test** dès le départ (stratifié) et ne regardez le test qu'à la fin.
- Comparez toujours à une **baseline**.
- Mettez la préparation des données dans un **`Pipeline`** pour éviter les fuites.
- L'exactitude ne suffit pas : regardez la **matrice de confusion**, la **précision**, le **rappel** et l'**AUC**, et choisissez selon le coût des erreurs.
- Un grand écart train/test signale du **sur-apprentissage**.
- Réglez les hyperparamètres par **validation croisée** (`GridSearchCV`).

## Pour aller plus loin

- [Guide scikit-learn : évaluation des modèles](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Guide scikit-learn : pipelines et transformations de colonnes](https://scikit-learn.org/stable/modules/compose.html)
- Les **forêts aléatoires** (`RandomForestClassifier`) : une moyenne de nombreux arbres, beaucoup plus robuste qu'un arbre seul.
