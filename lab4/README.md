# Lab 4 — Model Context Protocol (MCP)

Sujet complet : [`sujets/lab4/4 TP Agent MCP.md`](../sujets/lab4/4%20TP%20Agent%20MCP.md)

## Fichiers

- `mcp_local_server.py` — serveur MCP local (stdio) exposant un tool
  `search_web` (Tavily), une resource `github_file` (README distant) et un
  prompt dynamique.
- `agentMCP.py` — Partie 1 : client qui se connecte au serveur local,
  recupere dynamiquement tools/resources/prompt, puis interroge un agent.
- `agentMCPTime.py` — Partie 2 : agent connecte au serveur MCP `mcp-server-time`
  (lance via `uvx`) pour repondre a des questions d'heure/fuseau horaire.
- `agentMCPDistant.py` — Partie 3 : agent connecte au serveur MCP distant
  (`https://mcp.kiwi.com`) via HTTP streaming, pour la recherche de vols.

## Prerequis

- Ollama lance localement avec le modele `llama3.2:3b`.
- `TAVILY_API_KEY` dans le `.env` (utilise par `mcp_local_server.py`).
- `uvx` disponible (fourni par `uv`) pour lancer `mcp-server-time`.

## Lancer

```bash
uv run --active python lab4/agentMCP.py
uv run --active python lab4/agentMCPTime.py
uv run --active python lab4/agentMCPDistant.py
```
