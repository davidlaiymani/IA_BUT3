# Projet — Une application web qui combine IA prédictive et IA générative

Le projet clôt le module. En binôme, vous développez une application web qui met en œuvre **deux notions** du cours : une d'IA prédictive et une d'IA générative. Vous présentez ensuite ces deux notions à la classe, sous la forme d'un **cours inversé** : c'est vous qui faites cours, et votre application vous sert de support.

L'objectif est de **comprendre en profondeur deux notions** : savoir les expliquer, savoir quand elles sont utiles, et savoir les utiliser correctement. L'application est le moyen d'y parvenir : c'est en la construisant que vous rencontrerez les vraies difficultés de chaque notion.

## 1. Les notions

| IA prédictive | IA générative |
|---|---|
| **Clustering** (TP1) | **Sorties structurées** : faire produire au LLM un JSON conforme à un schéma (TP6) |
| **Classification** (TP2, TP3) | **Tool use** : laisser le LLM appeler des fonctions de votre code (TP7) |
| **Régression** (TP4, TP3) | |
| **Séries temporelles** (TP5) | |

XGBoost (TP3) n'est pas une notion à part : c'est un modèle que vous pouvez utiliser en classification, en régression ou en prévision, selon votre sujet.

## 2. Le sujet

### Le tirage au sort

Chaque binôme **tire au sort un sujet** parmi les 16 de l'annexe. Chaque sujet associe une notion d'IA prédictive et une notion d'IA générative : il y a deux sujets pour chacune des 8 associations possibles. Un sujet tiré est retiré de l'urne : deux binômes, quel que soit leur groupe, n'ont jamais le même sujet. Le groupe A tire en premier (16 novembre), le groupe B tire ensuite parmi les sujets restants (19 novembre).

Dans chaque sujet, les deux notions ont un **vrai rôle** : si l'on retire l'une des deux, l'application ne fonctionne plus. Deux schémas reviennent souvent :

- **le LLM prépare les données du modèle** : le LLM transforme du texte libre (une annonce, un avis, un ticket) en variables structurées, que le modèle d'IA prédictive utilise ensuite ;
- **le modèle devient un outil du LLM** : votre modèle entraîné (classifieur, régression, prévision) est exposé comme une fonction que le LLM appelle pour répondre à une question posée en langage naturel.

### Les variantes

Vous pouvez adapter votre sujet : changer de ville, de domaine, de produit ou de jeu de données. Une variante doit garder **les deux notions du sujet et leur rôle** dans l'application. Prévenez l'enseignant avant de vous engager dans une variante.

### Les échanges

Deux binômes peuvent échanger leurs sujets, d'un commun accord, **pendant la semaine qui suit leur tirage** (jusqu'au dimanche 22 novembre pour le groupe A, jusqu'au mercredi 25 novembre pour le groupe B), en prévenant l'enseignant.

## 3. Ce qui est attendu

### L'application

- Une **application web** qui fonctionne, dans laquelle on peut utiliser et observer les deux notions. **L'esthétique de l'interface n'est pas notée** (voir les outils recommandés ci-dessous).
- Des **données réelles**, ou des données rédigées par vous lorsque c'est indiqué, avec leur source et leur licence.
- **Une expérience mesurée par notion**, comme dans les TP : comparer deux modèles, deux valeurs d'un hyperparamètre, un LLM et un modèle classique, un prompt avec et sans exemples… Une notion d'IA prédictive s'évalue avec une métrique adaptée et sur des données qui n'ont pas servi à l'entraînement ; une notion d'IA générative s'évalue aussi, par exemple sur un jeu de cas annotés à la main.
- **Aucune clé d'API dans le code ni dans le dépôt** : lisez-la dans une variable d'environnement ou dans un fichier `.env` exclu du dépôt.

### Les outils web recommandés

Votre code d'IA est en Python : le plus simple est de construire l'interface en Python aussi, avec l'un de ces deux outils.

| Outil | Points forts | Pour quels sujets |
|---|---|---|
| **[Streamlit](https://streamlit.io/)** (recommandé) | une page web écrite comme un script Python : widgets (curseurs, listes, envoi de fichiers), tableaux, graphiques, interface de discussion (`st.chat_message`, `st.chat_input`), mise en cache d'un modèle entraîné (`st.cache_resource`) | tous les sujets, en particulier ceux qui affichent des données et des graphiques |
| **[Gradio](https://www.gradio.app/)** | conçu pour les démonstrations de modèles : une fonction Python devient une interface ; `gr.ChatInterface` fournit une interface de discussion complète | les sujets centrés sur un échange avec un assistant (tool use) ou sur la transformation d'un texte (sorties structurées) |

Installation, dans l'environnement du cours :

```bash
uv pip install streamlit gradio
```

```bash
conda install -c conda-forge streamlit gradio
```

Si vous êtes à l'aise en développement web, vous pouvez aussi séparer le modèle et l'interface : une API Python ([FastAPI](https://fastapi.tiangolo.com/) ou [Flask](https://flask.palletsprojects.com/)) qui charge le modèle et appelle le LLM, et une interface en HTML et JavaScript (ou avec le framework de votre choix) qui l'interroge. Cette architecture est plus proche d'une application réelle, mais demande davantage de travail sur la partie web, qui n'est pas ce qui est évalué : faites ce choix en connaissance de cause.

Dans tous les cas, entraînez et enregistrez votre modèle **en dehors de l'application** (notebook ou script, puis `joblib.dump`) et chargez-le au démarrage : l'application ne doit pas réentraîner le modèle à chaque requête.

### Le dépôt

Un dépôt GitHub contenant le code et un `README.md` qui indique :

- comment installer et lancer l'application ;
- les données utilisées, leur source et leur licence ;
- le rôle de chaque notion et les résultats de vos expériences ;
- les limites que vous avez constatées.

### Utiliser une IA pour coder

Vous pouvez vous faire aider par un assistant de programmation. Mais vous devez pouvoir **expliquer chaque choix** de votre application : pourquoi ce modèle, ce prétraitement, cette métrique, ce prompt, ce schéma de sortie. Toute métrique affichée dans l'application doit pouvoir être expliquée. Les questions qui suivent l'exposé portent précisément sur ces choix.

## 4. Le calendrier

| Étape | Date |
|---|---|
| Tirage au sort des sujets | lundi 16 novembre 2026 (groupe A), jeudi 19 novembre (groupe B) |
| Fin des échanges de sujets | dimanche 22 novembre, 23 h 59 (groupe A), mercredi 25 novembre, 23 h 59 (groupe B) |
| Point d'étape : l'application tourne et les deux notions y sont branchées, même de façon incomplète | semaine du 30 novembre |
| Publication de l'ordre de passage | lundi 7 décembre |
| Remise du dépôt (code et README), pour les deux groupes | dimanche 13 décembre, 23 h 59 |
| Soutenances | lundi 14 décembre (groupe A), mardi 15 décembre (groupe B) |

## 5. L'exposé : un cours inversé

Vous disposez de **15 minutes** pour faire comprendre vos deux notions à la classe, suivies de **5 minutes de questions**. Un déroulé possible :

1. présenter le sujet et l'application en deux minutes ;
2. expliquer chaque notion pour vos camarades, comme le ferait un enseignant : l'intuition, le principe, quand s'en servir et les pièges à éviter, en l'illustrant en direct dans votre application ;
3. présenter vos expériences et ce qu'elles montrent ;
4. terminer par une limite honnête : ce qui marche mal, ce que vous feriez autrement.

C'est un **cours** et non une démonstration : faites participer la classe, par exemple avec une question ou un exercice (« à votre avis, que va répondre le modèle si… ? »).

Chaque membre du binôme doit maîtriser **les deux notions** : la répartition de la parole entre vous est tirée au sort au début de l'exposé, et les questions s'adressent à chacun.

## 6. L'évaluation

La note tient compte des points suivants :

- **compréhension des notions** : explications justes, vocabulaire maîtrisé, réponses aux questions ;
- **qualité pédagogique** : clarté, exemples, participation de la classe, respect du temps ;
- **démarche expérimentale** : expériences pertinentes, métriques adaptées et bien interprétées ;
- **application et dépôt** : les deux notions y jouent un vrai rôle, l'application fonctionne, le README est complet ;
- **regard critique** : limites identifiées, honnêteté des résultats.

## Annexe — Les sujets

Les données indiquées ont été vérifiées en novembre 2026 : elles sont disponibles et leur licence permet un usage pédagogique. Lisez tout de même la fiche de votre jeu de données et citez-la dans votre README. Le symbole 🛠️ signale une partie des données à rédiger vous-mêmes.

Plusieurs sujets font passer de nombreux textes par le LLM. Travaillez alors sur un **échantillon** (quelques centaines à quelques milliers de textes) : le coût et le temps des appels augmentent avec le nombre de textes.

### Clustering × Sorties structurées

**1. Cartographie des mécontentements.** Le LLM transforme chaque avis client en fiche structurée : produit ou service, aspect critiqué, ton, gravité. Un clustering regroupe ces fiches en familles de problèmes, et l'application dresse le portrait de chaque groupe.
Données : [BenGaia/avis-clients](https://huggingface.co/datasets/BenGaia/avis-clients) (Hugging Face, CC BY 4.0) : environ 15 000 avis en français (hôtels, commerces, sites web) avec une note de 1 à 10.

**2. Familles de recettes.** Le LLM extrait d'une recette en texte libre ses ingrédients principaux, son temps, son coût estimé et son mode de cuisson. Un clustering regroupe les recettes en familles ; l'utilisateur colle une nouvelle recette et voit à quelle famille elle appartient.
Données : [Food.com – Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) (Kaggle, CC0) : plus de 500 000 recettes en anglais, avec ingrédients, instructions, temps et valeurs nutritionnelles.

### Clustering × Tool use

**3. Assistant de segmentation client.** Une clientèle est segmentée par clustering, puis un assistant répond aux questions d'un responsable marketing en appelant des outils : « décris le groupe 3 », « compare les groupes 1 et 4 », « dans quel groupe tombe ce nouveau client ? ».
Données : les clients du TP1 (*Mall Customers*) ou ceux du TP2 (*Telco Customer Churn*).

**4. Conseiller de destinations.** Des villes sont regroupées selon leur climat sur plusieurs années. Un agent recommande une destination en appelant des outils : recherche dans les groupes, météo prévue via une API.
Données : [Open-Meteo](https://open-meteo.com/) (CC BY 4.0, sans clé, gratuit pour un usage non commercial) : historique depuis 1940 et prévisions. 🛠️ La liste des villes est à constituer (par exemple les préfectures françaises).

### Classification × Sorties structurées

**5. Tri automatique de tickets de support.** Le LLM transforme un ticket rédigé librement en champs structurés : produit concerné, type de problème, ton, urgence exprimée. Ces champs alimentent un classifieur qui prédit le service destinataire ou la priorité. L'application compare cette approche à un LLM qui classerait seul.
Données : [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (Hugging Face, CC BY-NC 4.0, usage non commercial) : tickets en anglais et en allemand, avec le service (`queue`), la priorité et le type.

**6. Détecteur de fausses offres d'emploi.** Le LLM lit chaque offre et en extrait des indices structurés : salaire annoncé, présence d'une description de l'entreprise, précision des exigences, demande d'informations personnelles ou de paiement, ton de l'annonce. Ces indices alimentent un classifieur qui repère les offres frauduleuses ; l'application le compare à un modèle entraîné directement sur le texte.
Données : [Real or Fake Job Posting](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (Kaggle, CC0, Université de l'Égée) : 18 000 offres en anglais, dont environ 800 frauduleuses. Les classes sont très déséquilibrées : choisissez vos métriques en conséquence.

### Classification × Tool use

**7. Assistant anti-départ client.** Un conseiller interroge un agent en langage naturel : « ce client risque-t-il de partir ? pourquoi ? et s'il passait en contrat annuel ? ». L'agent appelle des outils qui prédisent le risque, expliquent les facteurs et simulent un changement.
Données : celles du TP2 (*Telco Customer Churn*, Apache 2.0).

**8. Détecteur de mails d'hameçonnage.** Un classifieur repère les mails suspects. Un agent l'appelle comme outil, ainsi que des outils d'analyse en lecture seule (liens contenus, domaine de l'expéditeur), et explique son verdict.
Données : [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) (Hugging Face, LGPL-3.0) : mails en anglais étiquetés hameçonnage ou légitimes.

### Régression × Sorties structurées

**9. Estimateur d'annonces immobilières.** Le LLM extrait d'une annonce en texte libre la surface, le nombre de pièces, la commune, le type de bien et l'extérieur. Une régression entraînée sur des ventes réelles estime le prix, et l'application signale les annonces sur- ou sous-évaluées.
Données : [Demandes de valeurs foncières géolocalisées](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees) (DVF, data.gouv.fr, Licence Ouverte) : les ventes immobilières en France depuis 2020. Conditions particulières : interdiction de ré-identifier les personnes et d'exposer les données à l'indexation des moteurs de recherche. 🛠️ Les annonces en texte libre sont à rédiger.

**10. Cote des voitures d'occasion.** Même principe sur des annonces de voitures : marque, modèle, année, kilométrage, boîte de vitesses et carburant sont extraits en JSON, puis une régression prédit le prix.
Données : [100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) (Kaggle, CC0) : 100 000 annonces britanniques avec prix, année, kilométrage, boîte, carburant et cylindrée. 🛠️ Les annonces en texte libre sont à rédiger.

### Régression × Tool use

**11. Assistant immobilier.** À la question « combien vaut un T3 de 65 m² à Belfort, et quelle mensualité sur 20 ans ? », l'agent combine plusieurs outils : le modèle de régression, une recherche de ventes comparables et un calcul de crédit.
Données : DVF (voir le sujet 9).

**12. Coach rénovation énergétique.** Une régression prédit le besoin de chauffage d'un logement à partir de ses caractéristiques. Un agent répond à « et si j'isolais les murs ? » en appelant le modèle avec des caractéristiques modifiées, et compare les scénarios.
Données : [Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) (UCI, CC BY 4.0) : 768 bâtiments simulés. Variante plus réaliste : les [DPE des logements existants](https://www.data.gouv.fr/datasets/dpe-logements-existants-depuis-juillet-2021) (ADEME, licence Etalab, API disponible), plus volumineux et plus bruités.

### Séries temporelles × Sorties structurées

**13. Veille des rappels de produits.** Chaque rappel de produit publié en France comporte un motif rédigé librement (« présence de *Listeria monocytogenes* », « morceaux de verre », « allergène non mentionné sur l'étiquette »). Le LLM le transforme en catégorie normalisée : contamination microbiologique, allergène non déclaré, corps étranger, contaminant chimique, défaut de sécurité… L'application construit la série hebdomadaire des rappels par catégorie et prévoit les semaines suivantes.
Données : [RappelConso](https://data.economie.gouv.fr/explore/dataset/rappelconso-v2-gtin-espaces/) (data.economie.gouv.fr, Licence Ouverte) : environ 19 000 fiches de rappel depuis 2018, mises à jour chaque jour et accessibles par API. Le champ `risques_encourus` contient déjà une catégorie de risque : comparez-la à celle que produit le LLM.

**14. Événements et fréquentation cycliste.** Le LLM extrait d'annonces d'événements parisiens (concerts, marchés, manifestations sportives) leur date, leur type et leur ampleur. Ces événements servent de variables explicatives pour prévoir le passage de vélos aux compteurs de la ville.
Données : [comptages horaires des vélos à Paris](https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/) (depuis 2016) et [Que faire à Paris ?](https://www.data.gouv.fr/datasets/que-faire-a-paris-evenements-et-activites) (agenda des événements), en données ouvertes.

### Séries temporelles × Tool use

**15. Assistant vélo.** À la question « combien de cyclistes passeront samedi boulevard de Sébastopol ? », l'agent appelle un outil de prévision, la météo prévue et le calendrier des jours fériés, puis justifie sa réponse.
Données : comptages vélo de Paris (voir le sujet 14), [Open-Meteo](https://open-meteo.com/) pour la météo et l'API des [jours fériés](https://www.data.gouv.fr/datasets/jours-feries-en-france) d'Etalab (`calendrier.api.gouv.fr`).

**16. Le meilleur moment pour consommer.** Un modèle prévoit l'intensité carbone de l'électricité française pour les prochaines heures. Un agent répond à « quand lancer ma machine à laver ? » en appelant la prévision et la météo.
Données : [éCO2mix](https://www.data.gouv.fr/datasets/donnees-eco2mix-nationales-temps-reel-1) de RTE (taux de CO₂ en g/kWh au quart d'heure, téléchargeable ou par API sur la plateforme ODRÉ) et Open-Meteo.
