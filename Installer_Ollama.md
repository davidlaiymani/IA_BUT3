# Utiliser un modèle de langage en local avec Ollama

Dans les TP d'IA générative, le code appelle par défaut l'**API de Mistral** : le modèle tourne sur les serveurs de Mistral et vous lui envoyez vos questions par Internet. Ce document décrit une **alternative facultative** : faire tourner un modèle directement sur votre ordinateur avec [Ollama](https://ollama.com).

Le code des TP reste le même dans les deux cas : seules deux lignes du fichier `.env` changent (voir la partie 5).

---

## 1. Pourquoi (ou pourquoi pas) un modèle local ?

| | API Mistral | Ollama en local |
|---|---|---|
| Compte et clé d'API | nécessaires | aucun |
| Coût | offre gratuite d'expérimentation, avec des limites de débit | gratuit |
| Vos données | envoyées chez Mistral | restent sur votre machine |
| Connexion Internet | nécessaire | inutile, une fois le modèle téléchargé |
| Qualité des réponses | grands modèles, très bonnes | petits modèles, nettement moins bonnes |
| Vitesse | rapide, quelle que soit votre machine | dépend de votre machine : de très rapide à très lent |
| Place sur le disque | aucune | 3 à 10 Go par modèle |

En résumé : Ollama est intéressant si vous ne voulez pas créer de compte, si vous voulez travailler hors ligne, ou si votre projet manipule des données qui ne doivent pas sortir de votre machine. Pour les TP, l'API Mistral donne de meilleurs résultats et fonctionne sur tous les ordinateurs.

> 💡 Vous pouvez installer les deux et passer de l'un à l'autre : c'est même une bonne façon de constater l'écart de qualité entre un petit et un grand modèle.

---

## 2. Votre machine est-elle assez puissante ?

Un modèle de langage doit tenir **entièrement en mémoire vive** (RAM, ou mémoire de la carte graphique) pendant qu'il répond. Sa taille dépend de son nombre de **paramètres** : les « 3b », « 8b », « 14b » des noms de modèles signifient 3, 8 ou 14 milliards (*billion* en anglais) de paramètres.

Pour connaître votre mémoire vive :

- **Windows** : Paramètres → Système → Informations système, ligne « Mémoire RAM installée » ;
- **macOS** : menu  → À propos de ce Mac, ligne « Mémoire » ;
- **Linux** : commande `free -h`, colonne `total`.

Choisissez ensuite le modèle d'après ce tableau. Nous recommandons la famille **Ministral 3**, de Mistral (licence Apache 2.0) : ce sont les mêmes modèles que ceux de l'API utilisée dans les TP (`ministral-8b-latest`), ils parlent bien français et savent appeler des outils, ce qui servira au TP7.

| Mémoire vive | Modèle conseillé | Téléchargement | Remarque |
|---|---|---|---|
| moins de 8 Go | — | — | utilisez l'API Mistral |
| 8 Go | `ministral-3:3b` | 3,0 Go | possible en fermant les autres applications ; réponses lentes |
| 16 Go | `ministral-3:3b` | 3,0 Go | confortable |
| 32 Go ou plus | `ministral-3:8b` | 6,0 Go | meilleures réponses |

Pour les TP sur la recherche sémantique et le RAG, il faut aussi un **modèle de plongements** (*embeddings*), beaucoup plus petit : `embeddinggemma` (622 Mo, multilingue).

Prévoyez **au moins 5 Go libres** sur le disque pour Ollama et ces deux modèles.

> ⚠️ Sans carte graphique, ou avec une carte graphique peu puissante, le modèle tourne sur le processeur. Cela fonctionne, mais une réponse peut prendre plusieurs dizaines de secondes. Les Mac à puce Apple (M1 à M4) utilisent leur processeur graphique automatiquement et s'en sortent bien.

---

## 3. Installer Ollama

### Windows

1. Téléchargez `OllamaSetup.exe` sur [ollama.com/download](https://ollama.com/download).
2. Lancez-le : l'installation se fait dans votre compte, sans droits d'administrateur.
3. Ollama démarre en arrière-plan (icône dans la zone de notification). La commande `ollama` est alors disponible dans PowerShell ou l'invite de commandes.

Il faut Windows 10 (version 22H2) ou plus récent.

### macOS

1. Téléchargez `Ollama.dmg` sur [ollama.com/download](https://ollama.com/download).
2. Ouvrez-le et glissez l'application Ollama dans le dossier Applications.
3. Lancez l'application. Au premier démarrage, elle propose d'installer la commande `ollama` dans le terminal : acceptez.

Il faut macOS 14 (Sonoma) ou plus récent.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Le script installe Ollama comme un service système. S'il n'est pas démarré :

```bash
sudo systemctl start ollama
```

### Vérifier l'installation

Dans un terminal :

```bash
ollama --version
```

La commande doit afficher un numéro de version (0.13.1 ou plus récent pour les modèles Ministral 3).

---

## 4. Télécharger et essayer un modèle

Téléchargez le modèle de langage choisi dans la partie 2, puis le modèle de plongements :

```bash
ollama pull ministral-3:3b
ollama pull embeddinggemma
```

Essayez le modèle directement dans le terminal :

```bash
ollama run ministral-3:3b
```

Posez une question, attendez la réponse, puis tapez `/bye` pour quitter. La première réponse est plus lente : le modèle se charge en mémoire.

Commandes utiles :

| Commande | Effet |
|---|---|
| `ollama list` | liste les modèles téléchargés et leur taille |
| `ollama ps` | liste les modèles chargés en mémoire en ce moment |
| `ollama stop ministral-3:3b` | décharge un modèle de la mémoire |
| `ollama rm ministral-3:3b` | supprime un modèle du disque |

---

## 5. Utiliser Ollama depuis Python

Ollama fait tourner un petit **serveur** sur votre machine, à l'adresse `http://localhost:11434`. Ce serveur imite l'API d'OpenAI : on l'interroge donc avec la bibliothèque `openai`, exactement comme on interroge l'API de Mistral dans les TP. Seules changent l'**adresse** du serveur et la **clé**.

### Installer les bibliothèques

Dans l'environnement du cours :

```bash
uv pip install openai python-dotenv pydantic
```

```bash
conda install -c conda-forge openai python-dotenv pydantic
```

### Le fichier `.env`

Les TP lisent leur configuration dans un fichier `.env`, placé dans le dossier du TP. Ce fichier n'est jamais publié ni partagé, car pour l'API Mistral il contient votre clé secrète.

Pour l'**API Mistral** (configuration par défaut des TP) :

```bash
LLM_BASE_URL=https://api.mistral.ai/v1
LLM_API_KEY=votre-clé-mistral
LLM_MODEL=ministral-8b-latest
EMBEDDING_MODEL=mistral-embed
```

Pour **Ollama** :

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=ministral-3:3b
EMBEDDING_MODEL=embeddinggemma
```

Ollama ne vérifie pas la clé, mais la bibliothèque `openai` en exige une : n'importe quel texte convient.

### Un premier appel

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # lit le fichier .env (dans un notebook : celui du dossier du notebook)
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])

reponse = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[
        {"role": "system", "content": "Tu réponds en une phrase."},
        {"role": "user", "content": "Qu'est-ce qu'un token pour un modèle de langage ?"},
    ],
    temperature=0.2,
)
print(reponse.choices[0].message.content)
```

Le même code fonctionne avec Mistral et avec Ollama : c'est le contenu du `.env` qui décide.

### Un premier plongement

```python
resultat = client.embeddings.create(
    model=os.environ["EMBEDDING_MODEL"],
    input=["Le chat dort.", "Un félin fait la sieste."],
)
vecteur = resultat.data[0].embedding
print(len(vecteur))  # dimension du vecteur : dépend du modèle
```

> ⚠️ Les vecteurs produits par deux modèles de plongements différents ne sont pas comparables : si vous changez de modèle, il faut recalculer tous les vecteurs (par exemple l'index d'un RAG).

---

## 6. Ce qui change par rapport à l'API

- **Les réponses sont moins bonnes.** Un modèle de 3 milliards de paramètres se trompe plus souvent, suit moins bien les consignes et invente davantage. Ce n'est pas un défaut de votre code. L'écart dépend de la tâche : pour classer les 50 critiques du TP6, `ministral-3:3b` en local en a classé correctement 92 %, contre 96 % pour `ministral-8b-latest` par l'API (novembre 2026).
- **Les modèles qui « raisonnent ».** Certains modèles (par exemple `gemma4`, `qwen3`, `gpt-oss`) écrivent d'abord un raisonnement caché avant de répondre. Avec une limite `max_tokens` basse, ce raisonnement peut consommer toute la limite et la réponse arrive **vide**. Les modèles Ministral 3 conseillés ici ne le font pas. Avec un autre modèle, ajoutez `reasoning_effort="none"` à l'appel, ou augmentez `max_tokens`.
- **Quelques paramètres sont ignorés**, comme `tool_choice` (forcer l'appel d'un outil) ou `n` (plusieurs réponses à la fois).
- **La première requête est lente** : le modèle se charge en mémoire. Il est déchargé après quelques minutes d'inactivité.

---

## 7. En cas de problème

| Symptôme | Cause probable | Solution |
|---|---|---|
| `Connection refused` / `APIConnectionError` | le serveur Ollama n'est pas lancé | lancez l'application Ollama (Windows, macOS) ou `sudo systemctl start ollama` (Linux) ; vérifiez avec `ollama list` |
| `model "…" not found` | modèle non téléchargé, ou nom mal écrit dans `.env` | `ollama pull <modèle>` ; comparez avec `ollama list` |
| Réponse vide | modèle qui raisonne et `max_tokens` trop bas | voir la partie 6 |
| Réponses très lentes, ordinateur qui rame | modèle trop gros pour votre mémoire | fermez les autres applications, prenez un modèle plus petit, ou revenez à l'API Mistral |
| Disque plein | modèles volumineux | `ollama rm <modèle>` ; pour stocker les modèles ailleurs, définissez la variable d'environnement `OLLAMA_MODELS` (chemin d'un dossier) puis relancez Ollama |
