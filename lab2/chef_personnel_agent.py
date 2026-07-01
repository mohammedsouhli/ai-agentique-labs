"""
TP Lab 2 : Chef personnel
Agent qui :
  - recoit les ingredients disponibles dans le frigo,
  - memorise les preferences/infos donnees par l'utilisateur,
  - utilise une recherche web pour completer ses connaissances culinaires,
  - propose un ou plusieurs plats adaptes aux ingredients disponibles.
"""

from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

load_dotenv(override=True)

model = ChatOllama(model="llama3.2:3b", temperature=0)

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Recherche sur le web des recettes, techniques ou associations d'ingredients."""
    return tavily_client.search(query)


SYSTEM_PROMPT = """
Vous etes un chef cuisinier personnel. Votre role :
- Retenir les ingredients que l'utilisateur dit avoir dans son frigo.
- Retenir ses preferences alimentaires (regime, allergies, gouts) au fil de la conversation.
- Utiliser l'outil web_search si vous avez besoin d'idees de recettes,
  de techniques de cuisson ou d'associations d'ingredients que vous ne connaissez pas.
- Proposer un ou plusieurs plats realisables avec les ingredients disponibles,
  en tenant compte des preferences memorisees.
Repondez toujours en francais, de maniere concise et pratique.
"""

chef_agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search],
    checkpointer=InMemorySaver(),
)


def ask(message: str, thread_id: str = "chef-1") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    response = chef_agent.invoke({"messages": [HumanMessage(content=message)]}, config)
    return response["messages"][-1].content


if __name__ == "__main__":
    print(ask("Bonjour, je suis vegetarien et je n'aime pas les champignons."))
    print()
    print(ask("J'ai dans mon frigo : des tomates, de la mozzarella, du basilic et des pates."))
    print()
    print(ask("Qu'est-ce que tu peux me proposer comme plat avec ca ?"))
