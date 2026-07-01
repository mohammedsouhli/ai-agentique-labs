# Lab 3 — Retrieval-Augmented Generation (RAG)

Sujet complet : [`sujets/lab3/3 TP Agent RAG.md`](../sujets/lab3/3%20TP%20Agent%20RAG.md)

## Fichiers

- `part1_rag_pdf.py` — Partie 1 : charge `acmecorp-employee-handbook.pdf`,
  decoupe le texte en chunks, genere des embeddings (`sentence-transformers/all-MiniLM-L6-v2`),
  indexe dans un `InMemoryVectorStore`, puis un agent (`search_handbook`) repond
  aux questions en s'appuyant sur le document.
- `part2_sql_agent.py` — Partie 2 : connecte un agent a la base `Chinook.db`
  via un outil `sql_query`, avec un system prompt qui force des reponses en
  langage naturel plutot que du SQL brut.
- `acmecorp-employee-handbook.pdf`, `Chinook.db` — donnees fournies avec le
  sujet du cours.

## Prerequis

- Ollama lance localement avec le modele `llama3.2:3b`.

## Lancer

```bash
uv run python lab3/part1_rag_pdf.py
uv run python lab3/part2_sql_agent.py
```
