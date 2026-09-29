# Pense-bête Pandas

Un aide-mémoire des opérations `pandas` utilisées dans les TP. Gardez-le ouvert à côté de votre notebook.

```python
import pandas as pd
import numpy as np
```

---

## 1. Charger et sauvegarder

| Besoin | Code |
|---|---|
| Lire un CSV | `df = pd.read_csv("data/fichier.csv")` |
| Séparateur `;` et décimales `,` | `pd.read_csv(f, sep=";", decimal=",")` |
| Lire un CSV compressé | `pd.read_csv("data/fichier.csv.gz")` (détection automatique) |
| Parser des dates à la lecture | `pd.read_csv(f, parse_dates=["date"])` |
| Utiliser une colonne comme index | `pd.read_csv(f, index_col="id")` |
| Lire un Excel | `pd.read_excel("fichier.xlsx", sheet_name=0)` |
| Sauvegarder en CSV | `df.to_csv("sortie.csv", index=False)` |

## 2. Premier coup d'œil

| Besoin | Code |
|---|---|
| Premières / dernières lignes | `df.head(10)`, `df.tail()` |
| Échantillon aléatoire | `df.sample(5, random_state=0)` |
| Dimensions (lignes, colonnes) | `df.shape` |
| Types et valeurs non nulles | `df.info()` |
| Statistiques descriptives | `df.describe()` · `df.describe(include="object")` |
| Noms des colonnes | `df.columns.tolist()` |
| Valeurs distinctes | `df["col"].unique()`, `df["col"].nunique()` |
| Fréquences | `df["col"].value_counts()` · `value_counts(normalize=True)` pour des proportions |
| Valeurs manquantes par colonne | `df.isna().sum()` |
| Doublons | `df.duplicated().sum()` |

## 3. Sélectionner

```python
df["age"]                       # une colonne -> Series
df[["age", "revenu"]]           # plusieurs colonnes -> DataFrame
df.loc[3, "age"]                # par étiquette (ligne 3, colonne "age")
df.loc[:, "age":"revenu"]       # tranche de colonnes par nom (bornes incluses)
df.iloc[0:5, 0:2]               # par position (borne de fin exclue)
df.select_dtypes(include="number")   # colonnes numériques
df.select_dtypes(include=["object", "category"])
```

## 4. Filtrer

```python
df[df["age"] > 30]
df[(df["age"] > 30) & (df["sexe"] == "F")]   # ET  -> &, parenthèses obligatoires
df[(df["age"] < 18) | (df["age"] > 65)]      # OU  -> |
df[~df["ville"].isin(["Paris", "Lyon"])]     # NON -> ~
df[df["nom"].str.contains("jean", case=False, na=False)]
df.query("age > 30 and sexe == 'F'")         # syntaxe alternative
df.nlargest(5, "revenu")                     # top 5
```

## 5. Créer et modifier des colonnes

```python
df["revenu_k"] = df["revenu"] / 1000
df["majeur"] = df["age"] >= 18
df["tranche"] = np.where(df["age"] < 30, "jeune", "senior")
df["classe_age"] = pd.cut(df["age"], bins=[0, 18, 40, 65, 120],
                          labels=["enfant", "jeune", "adulte", "senior"])
df["quartile"] = pd.qcut(df["revenu"], q=4, labels=False)
df["sexe"] = df["sexe"].map({"Male": 0, "Female": 1})
df["nom"] = df["nom"].str.strip().str.lower()
df = df.rename(columns={"Annual Income (k$)": "revenu"})
df = df.drop(columns=["id"])
df = df.astype({"code": "category"})
df = df.assign(ratio=lambda d: d["a"] / d["b"])   # pratique dans une chaîne
```

> 💡 Évitez `df.apply(fonction, axis=1)` quand une opération vectorisée existe : elle est souvent 100 fois plus rapide.

## 6. Valeurs manquantes

```python
df.isna().sum()                          # combien par colonne
df.isna().mean().sort_values()           # proportion par colonne
df.dropna()                              # supprime les lignes avec au moins un NaN
df.dropna(subset=["age"])                # seulement si "age" est manquant
df["age"] = df["age"].fillna(df["age"].median())
df["port"] = df["port"].fillna(df["port"].mode()[0])
df["valeur"] = df["valeur"].ffill()      # propage la dernière valeur connue (séries temporelles)
df["valeur"] = df["valeur"].interpolate()
```

## 7. Trier

```python
df.sort_values("age")
df.sort_values(["classe", "age"], ascending=[True, False])
df.sort_index()
df.reset_index(drop=True)                # renumérote 0..n-1
```

## 8. Grouper et agréger

```python
df.groupby("classe")["age"].mean()
df.groupby("classe").agg(age_moyen=("age", "mean"),
                         effectif=("age", "size"),
                         revenu_max=("revenu", "max"))
df.groupby(["contrat", "internet"])["churn"].mean().unstack()   # tableau croisé
pd.crosstab(df["contrat"], df["churn"], normalize="index")     # proportions par ligne
df.pivot_table(values="revenu", index="ville", columns="annee", aggfunc="mean")
df["moy_classe"] = df.groupby("classe")["age"].transform("mean")  # garde la taille de df
```

## 9. Combiner des tables

```python
pd.concat([df1, df2], ignore_index=True)          # empiler des lignes
pd.concat([df1, df2], axis=1)                     # coller des colonnes
df.merge(clients, on="client_id", how="left")     # jointure (inner, left, right, outer)
df.merge(autre, left_on="id", right_on="code")
```

## 10. Dates et séries temporelles

```python
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

df["date"].dt.year, df["date"].dt.month, df["date"].dt.dayofweek   # lundi = 0
df["date"].dt.hour, df.index.day_name()

df.loc["2012-06"]                        # tout juin 2012 (index date)
df.loc["2012-01-01":"2012-03-31"]        # une période

df["ventes"].resample("W").sum()         # hebdomadaire (D, W, MS, QS, YS…)
df["ventes"].rolling(7).mean()           # moyenne mobile sur 7 pas
df["ventes"].shift(1)                    # valeur précédente (création de lags)
df["ventes"].diff(7)                     # différence avec 7 pas avant
df["ventes"].pct_change()                # variation relative
```

## 11. Préparer des données pour scikit-learn

```python
X = df.drop(columns=["cible"])
y = df["cible"]

X = pd.get_dummies(X, columns=["ville", "sexe"], drop_first=True)   # one-hot
X["sexe"] = X["sexe"].astype("category")   # pour XGBoost (enable_categorical=True)

X.corr(numeric_only=True)["cible"].sort_values()   # corrélations avec la cible
y.value_counts(normalize=True)                     # déséquilibre des classes ?
```

## 12. Visualiser rapidement

```python
import matplotlib.pyplot as plt

df["age"].hist(bins=30)
df.plot.scatter(x="age", y="revenu", c="cluster", cmap="viridis")
df.boxplot(column="revenu", by="classe")
df["ventes"].plot(figsize=(12, 4))
df.groupby("contrat")["churn"].mean().plot.bar()
plt.show()
```

## 13. Pièges fréquents

- **`SettingWithCopyWarning`** : vous modifiez peut-être une vue. Faites `sous_df = df[df["a"] > 0].copy()` avant de modifier `sous_df`.
- **`and` / `or` dans un filtre** : utilisez `&` / `|` avec des parenthèses autour de chaque condition.
- **`inplace=True`** : préférez la réaffectation `df = df.dropna()`, plus lisible et chaînable.
- **Comparaison à `NaN`** : `df["a"] == np.nan` est toujours faux, utilisez `df["a"].isna()`.
- **Index décalé après un filtre** : `reset_index(drop=True)` avant de concaténer avec un autre tableau.
- **Fuite de données** : calculez moyennes, médianes ou paramètres de normalisation **sur le jeu d'entraînement uniquement**, puis appliquez-les au jeu de test.
