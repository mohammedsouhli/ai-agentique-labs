# Lab 7 — Human-In-The-Loop (HITL)

Sujet complet : [`sujets/lab7/7 TP Agent HITL.md`](../sujets/lab7/7%20TP%20Agent%20HITL.md)

## Fichier

- `agent_hitl.py` — un agent lit un email (etat `EmailState`) et propose une
  reponse via le tool `send_email`, gate par `HumanInTheLoopMiddleware`
  (`read_email` n'est pas soumis a approbation, `send_email` l'est). Trois
  scenarios sont demontres, chacun sur un thread independant :
  - **approve** : la proposition est envoyee telle quelle.
  - **reject** : l'envoi est refuse avec un message ; l'agent relance alors
    une nouvelle proposition qui necessite elle aussi une approbation (donc
    un nouveau `__interrupt__`, pas une reponse finale).
  - **edit** : l'humain remplace le corps du mail avant l'envoi ; c'est bien
    le texte modifie qui est envoye, pas la proposition initiale.

## Prerequis

- Ollama lance localement avec le modele `llama3.2:3b`.

## Lancer

```bash
uv run python lab7/agent_hitl.py
```
