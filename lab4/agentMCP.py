"""
TP Lab 4 - Partie 1 : Agent LLM connecte a un serveur MCP local (stdio)
Le client se connecte a mcp_local_server.py, recupere dynamiquement ses
tools, ses resources et son prompt, puis les fournit a un agent LangChain.

Executer avec : uv run --active python lab4/agentMCP.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)

LOCAL_SERVER_PATH = Path(__file__).parent / "mcp_local_server.py"


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": [str(LOCAL_SERVER_PATH)],
            }
        }
    )

    tools = await client.get_tools()
    print(f"Tools recuperes : {[t.name for t in tools]}")

    resources = await client.get_resources("local_server")
    print(f"Resources recuperees : {len(resources)}")

    prompt_messages = await client.get_prompt("local_server", "prompt")
    system_prompt = prompt_messages[0].content

    model = ChatOllama(model="llama3.2:3b", temperature=0)

    agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)

    config = {"configurable": {"thread_id": "1"}}
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config,
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
