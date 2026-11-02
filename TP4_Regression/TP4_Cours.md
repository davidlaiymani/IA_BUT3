# TP4 — Régression : cours

[⬅️ Retour au sommaire](../README.md) · Énoncé : [TP4_Enonce.ipynb](TP4_Enonce.ipynb)

## Ce que vous saurez faire à la fin de ce TP

- Formuler un problème de **régression** et choisir une métrique adaptée (MAE, RMSE, R²).
- Entraîner et interpréter une **régression linéaire** simple et multiple.
- Analyser les **résidus** pour diagnostiquer les défauts d'un modèle.
- Enrichir un modèle linéaire (variables polynomiales, transformation logarithmique) et le **régulariser** (Ridge, Lasso).
- Comparer modèles linéaires et modèles à base d'arbres (forêt aléatoire, XGBoost) par **validation croisée**.

---

## 1. Le problème de régression

Même cadre que la classification (TP2), mais la cible `y` est une **quantité continue** : un prix, une consommation, une durée, une température.

| Exemple | `X` | `y` |
|---|---|---|
| Immobilier (ce TP) | revenu du quartier, âge des logements, localisation… | prix médian des logements |
| Énergie | météo, jour, heure | consommation électrique (MWh) |
| Logistique | distance, trafic, météo | durée de livraison (min) |

Les outils du TP2 restent valables : séparation train/test, `Pipeline`, validation croisée, `GridSearchCV`. Seuls changent les **modèles** et les **métriques**.

## 2. Mesurer l'erreur

Pour chaque individu, l'**erreur** (ou **résidu**) vaut `eᵢ = yᵢ − ŷᵢ`.

| Métrique | Formule | Unité | Interprétation |
|---|---|---|---|
| MAE (erreur absolue moyenne) | $\frac{1}{n}\sum \lvert y_i - \hat y_i \rvert$ | celle de `y` | erreur « typique », robuste aux valeurs extrêmes |
| RMSE (racine de l'erreur quadratique moyenne) | $\sqrt{\frac{1}{n}\sum (y_i - \hat y_i)^2}$ | celle de `y` | pénalise fortement les grosses erreurs |
| R² (coefficient de détermination) | $1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$ | sans unité | part de la variance de `y` expliquée ; 1 = parfait, 0 = pas mieux que la moyenne, < 0 = pire que la moyenne |
| MAPE (erreur relative moyenne) | $\frac{1}{n}\sum \frac{\lvert y_i - \hat y_i \rvert}{\lvert y_i \rvert}$ | % | parlante pour le métier, instable si `y` proche de 0 |

- La **MAE** et la **RMSE** s'expriment dans l'unité de la cible : « on se trompe en moyenne de 35 000 $ » parle à tout le monde.
- **RMSE ≥ MAE** toujours ; un grand écart entre les deux signale quelques très grosses erreurs.
- Le **R²** permet de comparer des problèmes différents, mais il ne dit rien de l'ampleur des erreurs en unités métier.

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
mean_absolute_error(y_test, y_pred)
root_mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
```

La **baseline** en régression consiste à prédire toujours la moyenne du jeu d'entraînement (`DummyRegressor()`) : son R² vaut 0.

## 3. La régression linéaire

### 3.1 Régression simple

Avec une seule variable, on cherche la droite `ŷ = w₀ + w₁·x` qui passe « au plus près » des points, au sens des **moindres carrés** : elle minimise la somme des carrés des résidus.

```
  y │            •
    │        •  ╱  •
    │      •  ╱•           ŷ = w₀ + w₁·x
    │    •  ╱   ↕ résidu
    │  •  ╱  •
    │   ╱ •
    └──────────────────► x
```

### 3.2 Régression multiple

Avec `p` variables : `ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₚxₚ`.

**Interprétation d'un coefficient** : `wⱼ` est la variation moyenne de `ŷ` quand `xⱼ` augmente d'une unité, **toutes les autres variables étant fixées**.

- Pour comparer les coefficients entre eux, il faut que les variables soient sur la même échelle : **standardisez**. Le coefficient se lit alors « effet d'une augmentation d'un écart-type ».
- Si deux variables sont très corrélées (**colinéarité**, par exemple nombre de pièces et nombre de chambres), leurs coefficients deviennent instables et difficiles à interpréter : l'un peut devenir négatif pour compenser l'autre.

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

modele = make_pipeline(StandardScaler(), LinearRegression())
modele.fit(X_train, y_train)
modele[-1].coef_, modele[-1].intercept_
```

### 3.3 Les hypothèses et l'analyse des résidus

La régression linéaire suppose que la relation est **linéaire** et que les résidus sont du **bruit** : centrés, de variance constante, sans structure. On le vérifie graphiquement :

1. **Prédit vs réel** : les points doivent s'aligner sur la diagonale `y = ŷ`.
2. **Résidus vs prédit** : on doit voir un nuage sans forme autour de 0.

| Ce que l'on voit | Diagnostic | Remède |
|---|---|---|
| Une courbe (forme en U) | relation non linéaire | variables polynomiales, transformation, modèle non linéaire |
| Un entonnoir (dispersion qui grandit) | variance non constante | transformer la cible (logarithme) |
| Des points très isolés | valeurs aberrantes | vérifier les données, métrique robuste (MAE) |
| Une ligne horizontale ou diagonale nette | cible plafonnée ou censurée | le signaler, éventuellement exclure ces points |

## 4. Enrichir un modèle linéaire

Un modèle linéaire n'est linéaire **que dans ses coefficients** : on peut lui donner des variables transformées.

- **Variables polynomiales** : ajouter `x²`, `x³`, et les **interactions** `x₁·x₂` permet de capturer des courbures et des effets combinés.

  ```python
  from sklearn.preprocessing import PolynomialFeatures
  make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LinearRegression())
  ```

  Avec `p` variables et un degré 2, on obtient `p + p(p+1)/2` variables : 8 variables deviennent 44, et 164 au degré 3. Le risque de **sur-apprentissage** augmente vite.

- **Transformation logarithmique** : pour une variable (ou une cible) très asymétrique, avec une longue traîne vers les grandes valeurs (revenus, prix, populations), `log(1 + x)` resserre les grandes valeurs et rend les relations plus linéaires. Si on transforme la **cible**, il faut revenir à l'échelle d'origine avant de calculer les métriques (`TransformedTargetRegressor` le fait automatiquement).

## 5. La régularisation : Ridge et Lasso

Quand le modèle a beaucoup de variables (par exemple après une expansion polynomiale), les coefficients peuvent devenir énormes pour coller au bruit. La **régularisation** ajoute à la fonction de coût une **pénalité sur la taille des coefficients** :

| Modèle | Fonction minimisée | Effet |
|---|---|---|
| Moindres carrés | $\sum (y_i - \hat y_i)^2$ | aucun frein |
| **Ridge** (L2) | $\sum (y_i - \hat y_i)^2 + \alpha \sum w_j^2$ | rétrécit tous les coefficients vers 0 |
| **Lasso** (L1) | $\sum (y_i - \hat y_i)^2 + \alpha \sum \lvert w_j \rvert$ | met certains coefficients **exactement à 0** : sélection de variables |

- `α = 0` : régression classique ; `α` très grand : tous les coefficients tendent vers 0 (le modèle prédit la moyenne).
- `α` est un **hyperparamètre** : on le choisit par validation croisée (`RidgeCV`, `LassoCV`, ou `GridSearchCV`).
- La pénalité dépend de l'échelle des variables : **standardisez toujours** avant Ridge ou Lasso.

```python
from sklearn.linear_model import RidgeCV
modele = make_pipeline(PolynomialFeatures(2), StandardScaler(),
                       RidgeCV(alphas=np.logspace(-3, 3, 13)))
modele.fit(X_train, y_train)
modele[-1].alpha_     # α retenu par validation croisée
```

## 6. Les modèles non linéaires

Les modèles à base d'arbres vus aux TP2 et TP3 existent aussi en régression : chaque feuille prédit la **moyenne** des cibles des exemples qui y tombent.

- `DecisionTreeRegressor`, `RandomForestRegressor`, `XGBRegressor`, `HistGradientBoostingRegressor`.
- Ils capturent automatiquement les non-linéarités et les interactions, sans standardisation ni variables polynomiales.
- Limite : un arbre ne sait pas **extrapoler**. Il ne prédira jamais une valeur en dehors de l'intervalle des cibles vues à l'entraînement (contrairement à un modèle linéaire).

```python
from xgboost import XGBRegressor
XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6)   # objectif par défaut : erreur quadratique
```

## 7. Validation croisée en régression

Comme en classification (TP2), on compare les modèles par validation croisée sur le jeu d'entraînement. Attention au signe : scikit-learn **maximise** toujours un score, donc les erreurs sont renvoyées en négatif.

```python
from sklearn.model_selection import cross_val_score, KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rmse = -cross_val_score(modele, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
r2 = cross_val_score(modele, X_train, y_train, cv=cv, scoring="r2")
```

> 💡 En régression, `KFold` ne mélange pas les données par défaut : si le fichier est trié (par région, par date…), pensez à `shuffle=True`. Pour des **séries temporelles**, au contraire, il ne faut surtout pas mélanger : voir le TP5.

## À retenir

- Régression = cible continue ; mesurez l'erreur en **unités métier** (MAE, RMSE) et en part de variance expliquée (R²).
- La **régression linéaire** est une baseline interprétable ; lisez ses coefficients sur des variables **standardisées**.
- Regardez toujours les **résidus** : ils révèlent non-linéarités, variance non constante, plafonds et valeurs aberrantes.
- On peut rendre un modèle linéaire plus souple (polynômes, logarithmes) et le **régulariser** (Ridge, Lasso) pour éviter le sur-apprentissage.
- Les **ensembles d'arbres** (forêt aléatoire, XGBoost) sont en général les plus performants sur données tabulaires, mais n'extrapolent pas.

## Pour aller plus loin

- [Guide scikit-learn : modèles linéaires](https://scikit-learn.org/stable/modules/linear_model.html)
- [Exemple scikit-learn : pièges de l'interprétation des coefficients des modèles linéaires](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html)
- La régression **quantile** (`QuantileRegressor`, ou `objective="reg:quantileerror"` dans XGBoost) pour prédire un intervalle plutôt qu'une valeur.
