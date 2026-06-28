# Rapport — Système Agentic RAG pour le Tourisme au Maroc

**Module :** Systèmes Multi-Agents et Intelligence Artificielle Distribuée  
**Professeur :** Pr. Sara RETAL  
**Étudiant :** Mohammed Souhli  
**Filière :** Master IIBDCC — SDIA  
**Année :** 2024/2025  

---

## 1. Démarche suivie

### 1.1 Choix du domaine

Le domaine choisi est le **tourisme au Maroc**. Ce choix est motivé par la richesse documentaire disponible et la pertinence des questions complexes que l'on peut poser : planification d'itinéraires, comparaisons entre villes, conseils pratiques sur la culture et le budget.

### 1.2 Construction de la base documentaire

Cinq guides touristiques officiels au format PDF ont été utilisés comme base documentaire :

| Document | Contenu |
|----------|---------|
| `Marrakech-FR-Web.pdf` | Attractions, hébergement, gastronomie, médina |
| `Fes FR.pdf` | Médina, tanneries, université Al Quaraouiyine |
| `Casablanca FR.pdf` | Mosquée Hassan II, Art Déco, centre économique |
| `Essaouira FR.pdf` | Médina, port, culture gnaoua |
| `Tanger FR.pdf` | Détroit, Kasbah, histoire méditerranéenne |

Ces documents totalisent **75 pages PDF** découpées en **344 chunks** de 500 caractères avec un chevauchement de 50 caractères afin de préserver le contexte entre les fragments.

### 1.3 Vectorisation et indexation

Les chunks ont été vectorisés avec le modèle **all-MiniLM-L6-v2** de HuggingFace (modèle léger, gratuit, sans clé API). Un index **FAISS** a été construit pour permettre la recherche par similarité sémantique avec retour des 3 documents les plus pertinents (top-k=3).

### 1.4 Développement des outils

Trois outils ont été développés pour l'agent :

- **`recherche_tourisme`** : effectue une recherche vectorielle dans la base FAISS et retourne les chunks les plus pertinents avec leur source.
- **`conseils_pratiques`** : fournit des informations pratiques générales (monnaie, visa, langue, sécurité) sans passer par la base vectorielle.
- **`recherche_web`** : effectue une recherche sur internet via DuckDuckGo pour répondre aux questions sur des destinations non couvertes dans la base documentaire.

### 1.5 Construction du graphe LangGraph

Conformément aux exigences du projet, l'agent a été construit **manuellement avec LangGraph** sans utiliser la méthode `create_agent` de LangChain, afin de contrôler entièrement le cycle de raisonnement de l'agent.

---

## 2. Fonctionnement du système

### 2.1 Architecture du graphe

Le système est composé de quatre noeuds reliés par des edges conditionnels :

```
__start__ → agent → recherche_tourisme → agent → __end__
                  → conseils_pratiques → agent → __end__
                  → recherche_web      → agent → __end__
                  ↘ __end__ (réponse directe sans outil)
```

| Noeud | Rôle |
|-------|------|
| `agent` | Reçoit la question, décide quel outil appeler ou répond directement |
| `recherche_tourisme` | Recherche vectorielle dans la base FAISS (5 guides PDF) |
| `conseils_pratiques` | Informations pratiques générales sur le Maroc |
| `recherche_web` | Recherche internet DuckDuckGo pour les destinations hors base |

### 2.2 État et mémoire

L'état du graphe est défini par `AgentState` contenant un champ `messages` qui accumule l'historique complet de la conversation (question, appels d'outils, résultats, réponse finale). Cette accumulation constitue la **mémoire** de l'agent au fil du raisonnement.

### 2.3 Boucle de raisonnement agentique

1. L'utilisateur soumet une question
2. L'agent (LLM Ollama `llama3.2:3b`, exécuté localement) analyse la question
3. L'agent décide quel outil utiliser via `should_continue`
4. L'outil est exécuté (recherche FAISS, conseils ou recherche web)
5. Le résultat de l'outil est renvoyé à l'agent
6. L'agent génère la réponse finale en français

### 2.4 Structure modulaire du code

Le projet est organisé en fichiers séparés pour assurer la maintenabilité :

| Fichier | Responsabilité |
|---------|---------------|
| `ingest.py` | Chargement PDF, découpage et vectorisation des documents |
| `tools.py` | Définition des 3 outils de l'agent |
| `graph.py` | Architecture LangGraph (state, noeuds, edges) |
| `questions.py` | Banc de questions pour l'évaluation |
| `run_evaluation.py` | Exécution et mesure des performances |
| `generate_graph.py` | Sauvegarde du graphe en image PNG |
| `main.py` | Point d'entrée principal du système |

---

## 3. Résultats de l'évaluation

### 3.1 Questions simples — 10/10 correctes

| # | Question | Temps | Résultat |
|---|----------|-------|---------|
| 1 | Pourquoi Marrakech est-elle appelée la ville rouge ? | 4.64s | ✅ |
| 2 | Quelle est la hauteur du minaret de la mosquée Hassan II ? | 2.71s | ✅ |
| 3 | Quelle est la plus ancienne université du monde à Fès ? | 4.18s | ✅ |
| 4 | Quelle ville est connue pour son festival des vents et son surf ? | 3.42s | ✅ |
| 5 | Quel est le plat typique de Fès ? | 4.27s | ✅ |
| 6 | À quelle distance se trouve l'aéroport Mohammed V ? | 3.98s | ✅ |
| 7 | Pourquoi Essaouira est-elle surnommée la ville des vents ? | 4.05s | ✅ |
| 8 | Quelle est la particularité architecturale de la médina de Fès ? | 4.76s | ✅ |
| 9 | Quel est le style architectural dominant de Casablanca ? | 3.00s | ✅ |
| 10 | Quelle ville du nord est face à l'Espagne sur le détroit ? | 5.83s | ✅ |

**Temps moyen : 4.08s — Taux de réussite : 100%**

### 3.2 Questions complexes — 10/10 correctes

| # | Question | Temps | Résultat |
|---|----------|-------|---------|
| 1 | Marrakech + Fès en une semaine, budget limité ? | 6.76s | ✅ |
| 2 | Marrakech vs Essaouira pour la photographie ? | 15.67s | ✅ |
| 3 | Meilleure période pour éviter chaleur et foules ? | 5.75s | ✅ |
| 4 | Surf + culture : combinaison de villes sur 10 jours ? | 7.08s | ✅ |
| 5 | Différences architecturales Fès vs Marrakech ? | 7.33s | ✅ |
| 6 | Excursions depuis Tanger pour le nord du Maroc ? | 5.38s | ✅ |
| 7 | Circuit 2 semaines — 5 villes ? | 8.22s | ✅ |
| 8 | Plats traditionnels par ville ? | 9.09s | ✅ |
| 9 | Voyage en famille : Marrakech, Essaouira ou Tanger ? | 5.46s | ✅ |
| 10 | Avantages/inconvénients Casablanca vs villes impériales ? | 7.30s | ✅ |

**Temps moyen : 7.80s — Taux de réussite : 100%**

### 3.3 Synthèse des performances

| Métrique | Questions simples | Questions complexes |
|----------|------------------|-------------------|
| Temps moyen | 4.08s | 7.80s |
| Temps minimum | 2.71s | 5.38s |
| Temps maximum | 5.83s | 15.67s |
| Taux de succès | 100% | 100% |

Les questions complexes prennent en moyenne **2x plus de temps** que les questions simples, ce qui s'explique par la longueur des réponses générées et le raisonnement multi-étapes de l'agent. Le temps total pour les 20 questions est de **118.88 secondes**.

---

## 4. Limites et pistes d'amélioration

### 4.1 Limites actuelles

- **Base documentaire limitée** : 5 villes seulement. Des destinations importantes comme Rabat, Agadir ou le désert de Merzouga ne sont pas couvertes dans les PDFs (compensé par l'outil `recherche_web`).
- **Pas de persistance du vector store** : L'index FAISS est recréé à chaque démarrage, ce qui allonge le temps d'initialisation.
- **Modèle local limité** : Le modèle Ollama `llama3.2:3b` utilisé localement est plus compact que les grands modèles cloud, ce qui peut produire des réponses plus courtes sur les questions très complexes.
- **Mémoire conversationnelle limitée** : L'agent ne conserve pas le contexte entre deux sessions distinctes.

### 4.2 Pistes d'amélioration

- **Enrichir la base documentaire** avec des guides supplémentaires (Rabat, Agadir, Merzouga)
- **Sauvegarder l'index FAISS** sur disque pour éviter de le reconstruire à chaque démarrage
- **Implémenter la mémoire persistante** avec `MemorySaver` de LangGraph pour des sessions multi-tours
- **Déployer via Streamlit** pour une interface utilisateur accessible au grand public
- **Évaluation automatique** avec le framework RAGAS (faithfulness, relevancy, context recall)

---

## 5. Conclusion

Ce projet a permis de concevoir et implémenter un système **Agentic RAG complet** dédié au tourisme au Maroc. L'architecture manuelle du graphe LangGraph avec ses noeuds `agent`, `recherche_tourisme`, `conseils_pratiques` et `recherche_web`, sa logique de décision conditionnelle et son état partagé démontre la maîtrise de l'approche agentique demandée. Le système utilise de vrais guides touristiques PDF comme base documentaire et peut également rechercher sur internet pour les destinations non couvertes. Il atteint un taux de réussite de **100% sur les 20 questions testées**.

---

*Rapport rédigé dans le cadre de l'évaluation de fin de module — Master IIBDCC, 2024/2025*
