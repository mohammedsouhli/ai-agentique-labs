# Lab 2 — Agents avec LangChain

Sujet complet : [`sujets/lab2/2 TP Agents avec Langchain.md`](../sujets/lab2/2%20TP%20Agents%20avec%20Langchain.md)

## Fichiers

- `01_agents_demo.py` — parcours des concepts du cours : agent sans/avec system
  message, few-shot, reponse structuree (texte puis `BaseModel`), tool custom
  (`meteo_capitale`), tool de recherche web (Tavily), agent sans/avec memoire
  (`InMemorySaver`).
- `chef_personnel_agent.py` — TP final : un agent "chef personnel" qui retient
  les ingredients et preferences de l'utilisateur (memoire), et utilise une
  recherche web pour proposer des recettes adaptees.

## Prerequis

- Ollama lance localement avec le modele `llama3.2:3b` (`ollama pull llama3.2:3b`).
- `TAVILY_API_KEY` renseignee dans le `.env` a la racine du projet.

## Lancer

```bash
uv run python lab2/01_agents_demo.py
uv run python lab2/chef_personnel_agent.py
```
