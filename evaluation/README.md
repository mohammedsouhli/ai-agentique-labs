# Système Agentic RAG — Tourisme au Maroc

**Étudiant :** Mohammed Souhli  
**Module :** Systèmes Multi-Agents et Intelligence Artificielle Distribuée  
**Professeur :** Pr. Sara RETAL — Master IIBDCC, 2025/2026

---

## Description

Système **Agentic RAG** (Retrieval-Augmented Generation) dédié au tourisme au Maroc, construit avec **LangGraph**. L'agent raisonne, choisit ses outils et répond à des questions touristiques en français à partir de vrais guides PDF.

## Architecture

```
__start__ → agent → recherche_tourisme  → agent → __end__
                  → conseils_pratiques  → agent → __end__
                  → recherche_web       → agent → __end__
                  → __end__ (réponse directe)
```

## Technologies

| Composant | Technologie |
|-----------|------------|
| Agent & Graphe | LangGraph |
| LLM | Ollama `llama3.2:3b` (local) |
| Vector Store | FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Recherche web | DuckDuckGo (ddgs) |
| Loader PDF | LangChain PyPDFDirectoryLoader |

## Structure du projet

```
evaluation/
├── documents/          # Guides touristiques PDF (Marrakech, Fès, Casablanca, Essaouira, Tanger)
├── ingest.py           # Chargement PDF, chunking, index FAISS
├── tools.py            # 3 outils : recherche_tourisme, conseils_pratiques, recherche_web
├── graph.py            # Architecture LangGraph (State, noeuds, edges)
├── questions.py        # 10 questions simples + 10 questions complexes
├── run_evaluation.py   # Évaluation et mesure des performances
├── generate_graph.py   # Génération du graphe en PNG
├── main.py             # Point d'entrée principal
├── graph.png           # Visualisation du graphe LangGraph
└── rapport.md          # Rapport individuel (4 pages)
```

## Installation

```bash
# Installer les dépendances
uv pip install langchain langgraph langchain-community langchain-huggingface
uv pip install langchain-ollama faiss-cpu pypdf ddgs

# Installer et lancer Ollama
ollama pull llama3.2:3b
ollama serve
```

## Lancer l'évaluation

```bash
cd evaluation
uv run python main.py
```

## Résultats

| Métrique | Questions simples | Questions complexes |
|----------|-----------------|-------------------|
| Taux de réussite | 100% (10/10) | 100% (10/10) |
| Temps moyen | 4.08s | 7.80s |
| Temps total | | 118.88s pour 20 questions |
