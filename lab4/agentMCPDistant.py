"""
TP Lab 4 - Partie 3 : Agent connecte a un serveur MCP distant via HTTP streaming
Se connecte au serveur MCP public de Kiwi.com (recherche de vols) et laisse
un agent utiliser ses tools pour repondre a une demande de voyage.

Executer avec : uv run --active python lab4/agentMCPDistant.py
"""

import asyncio
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Tools recuperes : {[t.name for t in tools]}")

    model = ChatOllama(model="llama3.2:3b", temperature=0)

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt="You are a travel agent. No follow up questions.",
    )

    config = {"configurable": {"thread_id": "1"}}
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Get me a direct flight from Rabat to Agadir on August 31st")]},
        config,
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
