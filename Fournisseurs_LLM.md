# Quel fournisseur de LLM utiliser gratuitement ?

[⬅️ Retour au sommaire](README.md) · Modèle local : [Installer_Ollama.md](Installer_Ollama.md)

Les TP6 et TP7 utilisent l'API de Mistral, mais votre code n'en dépend pas : il passe par le SDK `openai` et lit trois lignes du fichier `.env`. Tout fournisseur compatible avec ce format peut donc le remplacer, en changeant **seulement ces trois lignes**. Cette note compare les offres gratuites utilisables pour les TP et pour votre projet.

> ⚠️ **Situation relevée en novembre 2026.** Les offres gratuites changent souvent, sans préavis, et la plupart des fournisseurs ne publient plus leurs limites exactes : vérifiez-les dans la console de votre compte. Seuls Mistral et Ollama ont été testés avec le code du cours ; pour les autres, les informations viennent de leur documentation et de relevés publiés.

## Ce qu'il vous faut

Pour les TP et pour la plupart des sujets du projet, le fournisseur doit offrir :

1. une **API compatible OpenAI** (une adresse `base_url` utilisable avec le SDK `openai`) ;
2. les **sorties structurées** (`client.chat.completions.parse` avec un schéma Pydantic, TP6) ;
3. l'**appel d'outils** (`tools=…`, TP7) ;
4. des **limites de débit** suffisantes pour travailler : nombre de requêtes par minute et par jour.

## Les offres gratuites

| Fournisseur | Inscription | Sorties structurées | Outils | Limites gratuites (indicatives) | Testé dans le cours |
|---|---|---|---|---|---|
| **Mistral** | compte et numéro de téléphone, sans carte bancaire | oui | oui | non publiées ; selon les comptes, de quelques requêtes à plusieurs dizaines de requêtes par minute | ✅ |
| **Groq** | compte, sans carte bancaire | oui (mode strict sur certains modèles) | oui, mais **pas en même temps** que les sorties structurées | environ 30 requêtes par minute et 1 000 par jour | — |
| **Google Gemini** (AI Studio) | compte Google, sans carte bancaire | oui | oui | très basses sur certains modèles (de l'ordre de 20 requêtes par jour pour les modèles Flash, selon des relevés d'octobre 2026) | — |
| **GitHub Models** | compte GitHub | selon le modèle | selon le modèle | environ 150 requêtes par jour pour les petits modèles, 50 pour les gros ; 8 000 tokens en entrée par requête | — |
| **OpenRouter** (modèles `:free`) | compte, sans carte bancaire | selon le modèle | selon le modèle | 20 requêtes par minute et 50 par jour | — |
| **Ollama** (sur votre ordinateur) | aucune | oui | oui (`tool_choice` ignoré) | aucune limite, mais la vitesse dépend de votre machine | ✅ |

Cerebras, souvent cité dans les comparatifs, ne propose plus d'offre gratuite permanente : un crédit d'essai est accordé contre une carte bancaire.

## Nos recommandations

- **Pour les TP** : Mistral, comme indiqué dans les énoncés. Si vous recevez beaucoup d'erreurs 429 (« trop de requêtes »), la fonction `appeler` des TP attend et réessaie : laissez-la faire. Si c'est trop lent, passez à Groq ou à Ollama.
- **Pour le projet** : commencez avec Mistral ou Groq. Groq offre le plus de requêtes par jour sans carte bancaire. Gardez Ollama comme solution de secours si votre ordinateur le permet : il ne dépend d'aucun compte ni d'aucune limite.
- **Ne dépendez pas d'un seul fournisseur** : comme votre code lit le `.env`, prévoyez dès le début de pouvoir changer de fournisseur. Testez l'autre avant la soutenance.

## Les lignes du `.env`

Remplacez les trois lignes du fichier `.env` par celles du fournisseur choisi. La clé se crée dans la console du fournisseur ; elle est **personnelle et secrète** (TP6, section 2.1).

**Mistral** ([console.mistral.ai](https://console.mistral.ai)) :

```bash
LLM_BASE_URL=https://api.mistral.ai/v1
LLM_API_KEY=votre-clé-mistral
LLM_MODEL=ministral-8b-latest
```

**Groq** ([console.groq.com](https://console.groq.com)) :

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=votre-clé-groq
LLM_MODEL=openai/gpt-oss-20b
```

**Google Gemini** ([aistudio.google.com](https://aistudio.google.com)) : choisissez un modèle Flash dans la liste d'AI Studio et recopiez son nom exact.

```bash
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=votre-clé-gemini
LLM_MODEL=nom-du-modèle-flash
```

**GitHub Models** ([github.com/marketplace/models](https://github.com/marketplace/models)) : la clé est un jeton d'accès GitHub (*personal access token*) avec la permission `models` ; le nom du modèle se trouve dans le catalogue.

```bash
LLM_BASE_URL=https://models.github.ai/inference
LLM_API_KEY=votre-jeton-github
LLM_MODEL=nom-du-modèle-du-catalogue
```

**OpenRouter** ([openrouter.ai](https://openrouter.ai)) : choisissez un modèle dont le nom se termine par `:free`, et vérifiez sur sa fiche qu'il accepte les outils (*tools*) et les sorties structurées.

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=votre-clé-openrouter
LLM_MODEL=nom-du-modèle:free
```

**Ollama**, sur votre ordinateur : voir [Installer_Ollama.md](Installer_Ollama.md).

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=ministral-3:3b
```

Après un changement de fournisseur, relancez le noyau du notebook pour que le nouveau `.env` soit lu, puis vérifiez sur une question simple que tout fonctionne : un premier appel, une sortie structurée, un appel d'outil.

## Précautions

- **Vos données** : sur les offres gratuites, les fournisseurs se réservent souvent le droit d'utiliser vos requêtes pour améliorer leurs modèles. N'envoyez ni données personnelles ni données confidentielles. Avec Ollama, rien ne quitte votre ordinateur.
- **Les résultats changent d'un fournisseur à l'autre** : un autre modèle ne donnera pas les mêmes réponses, ni les mêmes scores. Si vous changez de fournisseur en cours de projet, refaites vos mesures et indiquez dans votre README le fournisseur, le modèle et la date.
- **Les limites sont partagées** : chez la plupart des fournisseurs, elles s'appliquent à votre compte entier, pas à chaque clé. Créer plusieurs clés ne donne pas plus de requêtes.
- **Le nom des modèles change** : si un modèle disparaît (erreur 404), cherchez son remplaçant dans la liste du fournisseur.
