"""
TP Lab 4 - Partie 2 : Agent LLM avec un serveur MCP de temps (mcp-server-time)
Permet de repondre a des questions sur l'heure locale dans un fuseau donne.

Executer avec : uv run --active python lab4/agentMCPTime.py
"""

import asyncio
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-time", "--local-timezone=America/New_York"],
            }
        }
    )

    tools = await client.get_tools()
    print(f"Tools recuperes : {[t.name for t in tools]}")

    model = ChatOllama(model="llama3.2:3b", temperature=0)
    agent = create_agent(model=model, tools=tools)

    question = HumanMessage(content="What time is it in Japan?")
    response = await agent.ainvoke({"messages": [question]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
