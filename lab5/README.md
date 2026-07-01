# Lab 5 — LangGraph Studio & Multi-Agents

Sujets complets :
[`sujets/lab5/5 TP Langraph Studio.md`](../sujets/lab5/5%20TP%20Langraph%20Studio.md),
[`sujets/lab5/8 TP Multi-Agents.md`](../sujets/lab5/8%20TP%20Multi-Agents.md)

## Fichiers

- `agent_simple.py` — un agent minimal avec un tool `lore_search`, pense pour
  etre visualise noeud par noeud dans LangGraph Studio.
- `langgraph.json` — config Studio/CLI. `root` pointe vers `..` car ce projet
  utilise un seul environnement `uv` partage a la racine du repo (au lieu
  d'un projet `uv` independant par lab).
- `multi_agents.py` — un agent principal qui delegue a deux sous-agents
  (`square_root`, `square`), chacun expose comme un tool. L'orchestrateur
  utilise `ChatGroq` (llama-3.3-70b) plutot que le petit modele local :
  avec `llama3.2:3b` comme orchestrateur, le modele hallucine parfois un
  second appel d'outil inutile apres avoir deja la bonne reponse. Les
  sous-agents, qui n'ont qu'un seul calcul a faire, restent sur Ollama.

## Prerequis

- Ollama lance localement avec le modele `llama3.2:3b`.
- `GROQ_API_KEY` dans le `.env` (utilise par l'orchestrateur de `multi_agents.py`).
- Compte sur https://smith.langchain.com/ pour utiliser Studio (Partie 1 du sujet).

## Lancer

```bash
uv run python lab5/multi_agents.py

# Depuis le dossier lab5/, pour ouvrir LangGraph Studio :
cd lab5
uv run --active langgraph dev
```
