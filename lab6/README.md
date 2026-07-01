# Lab 6 — Contexte et État d'un agent

Sujet complet : [`sujets/lab6/6 TP Agent avec état et contexte.md`](../sujets/lab6/6%20TP%20Agent%20avec%20%C3%A9tat%20et%20contexte.md)

## Fichiers

- `agent_context.py` — le **contexte** (`FoodContext`) est immuable pendant
  un `invoke()` : fourni via `context=`, lu par les tools via
  `runtime.context`. Demontre un agent sans acces au contexte (il devine),
  puis avec acces (via tools), puis un changement de contexte entre deux
  appels.
- `agent_state.py` — l'**etat** (`FoodState`, herite de `AgentState`) est
  mutable et persiste entre les tours grace a un `InMemorySaver`. Un tool
  (`update_favourite_dish`) le modifie via `Command`, un autre
  (`read_favourite_dish`) le lit via `runtime.state`.

Exemple utilise : les preferences culinaires (plat prefere), plutot que les
couleurs de l'exemple du cours, pour varier l'illustration tout en couvrant
exactement les memes mecanismes (contexte vs etat).

## Prerequis

- Ollama lance localement avec le modele `llama3.2:3b`.

## Lancer

```bash
uv run python lab6/agent_context.py
uv run python lab6/agent_state.py
```
