# TP5 — Séries temporelles : cours

[⬅️ Retour au sommaire](../README.md) · Énoncé : [TP5_Enonce.ipynb](TP5_Enonce.ipynb)

## Ce que vous saurez faire à la fin de ce TP

- Manipuler une série temporelle avec pandas : index de dates, rééchantillonnage, moyennes mobiles, décalages.
- Décomposer une série en **tendance**, **saisonnalité** et **résidu**, et tester sa **stationnarité**.
- Lire les graphes d'**autocorrélation** (ACF et PACF).
- Évaluer une prévision **sans tricher** : découpage temporel et références naïves.
- Prévoir avec un modèle **SARIMA** puis avec une approche **supervisée** (variables retardées + XGBoost), et comparer.

---

## 1. Qu'est-ce qu'une série temporelle ?

Une **série temporelle** est une suite d'observations indexées par le temps, à intervalles réguliers : ventes quotidiennes, consommation électrique horaire, température toutes les 10 minutes, trafic réseau par seconde…

Ce qui la distingue des données des TP précédents : **l'ordre compte**. Les observations ne sont pas indépendantes : la valeur d'aujourd'hui dépend fortement de celle d'hier. Cette dépendance est à la fois une **ressource** (le passé aide à prévoir le futur) et un **piège** (on ne peut plus mélanger les données comme avant).

### Manipulations pandas indispensables

```python
df["date"] = pd.to_datetime(df["date"])
s = df.set_index("date")["ventes"].asfreq("D")   # série quotidienne régulière

s.loc["2012-06"]            # un mois
s.resample("W").sum()       # agrégation hebdomadaire (MS = début de mois, QS, YS…)
s.rolling(30).mean()        # moyenne mobile sur 30 jours : lisse le bruit
s.shift(1)                  # valeur de la veille
s.diff(7)                   # différence avec la même journée la semaine précédente
```

Voir aussi la section 10 du [pense-bête pandas](../Pense_bete_Pandas.md).

## 2. Les composantes d'une série

On décrit classiquement une série comme la combinaison de :

| Composante | Description | Exemple (vélos en libre-service) |
|---|---|---|
| **Tendance** | évolution de fond, à long terme | le service gagne des abonnés d'année en année |
| **Saisonnalité** | motif qui se répète avec une période **fixe** | cycle annuel (été / hiver), cycle hebdomadaire (semaine / week-end) |
| **Résidu** (bruit) | ce qui reste : aléas, événements ponctuels | une journée de tempête, un jour férié |

- **Modèle additif** : `y = tendance + saisonnalité + résidu` — l'amplitude saisonnière est constante.
- **Modèle multiplicatif** : `y = tendance × saisonnalité × résidu` — l'amplitude saisonnière grandit avec le niveau (on s'y ramène en additif en prenant le logarithme).

```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(s, model="additive", period=7)
decomp.plot()
```

> 💡 Une **saisonnalité** a une période fixe et connue (jour, semaine, année). Un **cycle** a une durée variable (cycles économiques) : il est beaucoup plus difficile à prévoir.

## 3. Stationnarité

Une série est **stationnaire** si ses propriétés statistiques (moyenne, variance, autocorrélations) ne changent pas au cours du temps. Une série avec tendance ou saisonnalité n'est **pas** stationnaire.

Pourquoi c'est important : les modèles ARMA supposent la stationnarité. Le passé n'aide à prévoir que si les « règles du jeu » restent les mêmes.

### Rendre une série stationnaire : la différenciation

- **Différence simple** `y'ₜ = yₜ − yₜ₋₁` : supprime une tendance.
- **Différence saisonnière** `y'ₜ = yₜ − yₜ₋ₘ` (ici `m = 7`) : supprime une saisonnalité de période `m`.
- Transformation **logarithmique** : stabilise une variance qui grandit avec le niveau.

### Le test de Dickey-Fuller augmenté (ADF)

- Hypothèse nulle H₀ : la série **n'est pas** stationnaire (elle a une « racine unitaire »).
- Si la **p-value < 0,05**, on rejette H₀ : la série peut être considérée comme stationnaire.

```python
from statsmodels.tsa.stattools import adfuller
stat, p_value, *_ = adfuller(s.dropna())
```

> ⚠️ Un test n'est qu'un indice : regardez toujours le graphique de la série avant de conclure.

## 4. Autocorrélation : ACF et PACF

L'**autocorrélation** au décalage (*lag*) `k` est la corrélation entre la série et elle-même décalée de `k` pas : `corr(yₜ, yₜ₋ₖ)`.

- **ACF** (*autocorrelation function*) : autocorrélation pour chaque `k`. Des pics réguliers à 7, 14, 21… signalent une **saisonnalité hebdomadaire** ; une décroissance très lente signale une **tendance** (série non stationnaire).
- **PACF** (*partial autocorrelation*) : corrélation entre `yₜ` et `yₜ₋ₖ` **une fois retiré l'effet des décalages intermédiaires**. Elle indique quels décalages apportent une information **directe**.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(s, lags=40)
plot_pacf(s, lags=40)
```

La zone bleue est l'intervalle de confiance : les barres qui en sortent sont significatives.

## 5. Évaluer une prévision sans tricher

### 5.1 Découpage temporel

**On ne mélange jamais une série temporelle.** Avec un `train_test_split` aléatoire, le modèle verrait le 15 mars pour prévoir le 14 mars : il « lirait l'avenir », et l'évaluation serait beaucoup trop optimiste. On coupe à une **date** : tout ce qui précède sert à l'entraînement, tout ce qui suit à l'évaluation.

```
 ├──────────────── entraînement ────────────────┤──── test ────┤
 2011-01                                     2012-09       2012-12
```

Pour la validation croisée, `TimeSeriesSplit` agrandit progressivement la fenêtre d'entraînement en validant toujours sur la période **suivante** :

```
 Pli 1 : [train      ][val]
 Pli 2 : [train           ][val]
 Pli 3 : [train                ][val]
 Pli 4 : [train                     ][val]
```

### 5.2 Horizon de prévision

- **Prévision à un pas** (J+1) : chaque jour, on prévoit le lendemain en connaissant tout le passé jusqu'à aujourd'hui.
- **Prévision multi-pas** (J+1 à J+h) : on prévoit d'un coup les `h` prochains jours. L'erreur augmente avec l'horizon.

Les deux ne sont pas comparables : précisez toujours l'horizon.

### 5.3 Références naïves

Tout modèle doit battre ces prévisions triviales :

| Méthode | Prévision de `yₜ` |
|---|---|
| **Naïve** (persistance) | `yₜ₋₁` : « demain comme aujourd'hui » |
| **Naïve saisonnière** | `yₜ₋ₘ` : « lundi prochain comme lundi dernier » |
| **Moyenne mobile** | moyenne des `k` dernières valeurs |

Elles sont souvent étonnamment difficiles à battre. Métriques : MAE, RMSE (voir TP4), et MAPE pour une erreur relative.

## 6. Les modèles ARIMA et SARIMA

### 6.1 AR, MA, ARMA

- **AR(p)** (*autorégressif*) : la valeur est une combinaison linéaire des `p` valeurs précédentes.
  $y_t = c + \varphi_1 y_{t-1} + \dots + \varphi_p y_{t-p} + \varepsilon_t$
- **MA(q)** (*moyenne mobile*, rien à voir avec `rolling`) : la valeur dépend des `q` dernières **erreurs** de prévision.
  $y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}$
- **ARMA(p, q)** combine les deux ; il suppose une série stationnaire.

### 6.2 ARIMA(p, d, q)

Le **I** (*integrated*) intègre la différenciation : on différencie `d` fois la série, on la modélise en ARMA(p, q), puis on « ré-intègre » pour revenir à l'échelle d'origine. `d = 1` suffit en général pour une tendance.

### 6.3 SARIMA(p, d, q)(P, D, Q, m)

Ajoute une partie **saisonnière** de période `m` : les mêmes mécanismes AR, I et MA, mais aux décalages `m, 2m, …`. Par exemple, pour une série quotidienne avec cycle hebdomadaire, `m = 7`.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

modele = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
res = modele.fit(disp=False)
res.summary()
prevision = res.get_forecast(steps=30)       # prévision multi-pas
prevision.predicted_mean, prevision.conf_int()
```

### 6.4 Choisir les ordres

| Indice | Suggère |
|---|---|
| PACF qui se coupe net après `p` décalages, ACF qui décroît | AR(p) |
| ACF qui se coupe net après `q` décalages, PACF qui décroît | MA(q) |
| Série non stationnaire (ADF, graphique) | `d = 1` |
| Pics de l'ACF à `m`, `2m`… | partie saisonnière de période `m` |

En pratique, on compare quelques combinaisons avec un critère d'information (**AIC**, plus petit = meilleur) ou par validation temporelle. `pmdarima.auto_arima` automatise cette recherche.

**Forces** : modèle statistique éprouvé, intervalles de confiance, peu de données nécessaires.
**Limites** : relations linéaires, une seule saisonnalité facilement gérable, prise en compte limitée des variables externes (la version `SARIMAX` en accepte), lent sur de longues séries.

## 7. L'approche supervisée

On transforme la série en un problème de **régression tabulaire** (TP4) : chaque date devient une ligne, avec des variables construites à partir du passé et du calendrier.

| Type de variable | Exemples | Code pandas |
|---|---|---|
| **Retards** (*lags*) | valeur de la veille, d'il y a 7 jours | `s.shift(1)`, `s.shift(7)` |
| **Fenêtres glissantes** | moyenne des 7 derniers jours | `s.shift(1).rolling(7).mean()` |
| **Calendrier** | jour de la semaine, mois, jour férié | `index.dayofweek`, `index.month` |
| **Exogènes** | météo, prix, promotions | colonnes externes |

```
 (valeurs illustratives)
 date        cnt   lag_1  lag_7  moy_7j  jour  temp   → cible : cnt
 2012-06-08  7129  6824   6897   6711    4     0.66
 2012-06-09  6932  7129   5972   6765    5     0.67
```

> ⚠️ **Fuite de données** : chaque variable doit être **connue au moment de la prévision**. Pour prévoir le jour `t`, la moyenne glissante doit s'arrêter à `t − 1` (d'où le `shift(1)` avant `rolling`). De même, la météo du jour `t` n'est pas connue à l'avance : en production, on utiliserait la **prévision** météo.

On peut alors utiliser n'importe quel modèle de régression : forêt aléatoire, XGBoost… avec validation par `TimeSeriesSplit`.

**Forces** : relations non linéaires, nombreuses variables exogènes, très performant en pratique.
**Limites** : un arbre **n'extrapole pas** (voir TP4), donc une tendance forte doit être traitée à part (différenciation, ou prédiction de l'écart à la tendance) ; la prévision multi-pas demande d'enchaîner les prévisions ou d'entraîner un modèle par horizon.

## À retenir

- L'ordre compte : **jamais de mélange**, découpage à une date, `TimeSeriesSplit`.
- Commencez par **visualiser** : tendance, saisonnalités, ruptures, valeurs aberrantes.
- Décomposition, différenciation, test ADF, ACF / PACF : les outils pour comprendre la structure de la série.
- Toujours comparer à une **référence naïve** (persistance, naïve saisonnière).
- **SARIMA** : modèle statistique, linéaire, avec intervalles de confiance.
- **Approche supervisée** : variables retardées + calendrier + exogènes + XGBoost, souvent la plus performante ; attention aux fuites de données.

## Pour aller plus loin

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) (R. Hyndman, G. Athanasopoulos) : la référence, gratuite en ligne.
- [Documentation statsmodels : analyse de séries temporelles](https://www.statsmodels.org/stable/tsa.html)
- Bibliothèques spécialisées : `sktime`, `Nixtla statsforecast`, `Prophet`, `Darts`.
- Lissage exponentiel (Holt-Winters, `ExponentialSmoothing` dans statsmodels) : une alternative simple et robuste à SARIMA.
