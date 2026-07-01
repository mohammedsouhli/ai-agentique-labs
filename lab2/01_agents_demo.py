"""
Lab 2 - Agents avec LangChain
Parcours des concepts du TP : system message, few-shot, reponse structuree,
tools, web search tool, memoire.

Chaque section peut etre commentee/decommentee pour tester un concept a la fois.
"""

from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv(override=True)

model = ChatOllama(model="llama3.2:3b", temperature=0)


def demo_sans_system_message():
    agent = create_agent(model=model)
    question = HumanMessage(content="Quelle est la capitale de la lune ?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


def demo_avec_system_message():
    system_prompt = (
        "Vous etes un auteur de science-fiction ; "
        "inventez une capitale imaginaire a la demande des utilisateurs."
    )
    agent = create_agent(model=model, system_prompt=system_prompt)
    question = HumanMessage(content="Quelle est la capitale de la lune ?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


def demo_few_shot():
    system_prompt = """
Vous etes un auteur de science-fiction et vous devez inventer une capitale
spatiale a la demande d'un utilisateur.

Utilisateur : Quelle est la capitale de Mars ?
Auteur : Marsialis

Utilisateur : Quelle est la capitale de Venus ?
Auteur : Venusovia
"""
    agent = create_agent(model=model, system_prompt=system_prompt)
    question = HumanMessage(content="Quelle est la capitale de la lune ?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


def demo_reponse_structuree_texte():
    system_prompt = """
Vous etes un auteur de science-fiction et vous devez inventer une capitale
spatiale a la demande d'un utilisateur. Respectez la structure ci-dessous.

Nom : nom de la capitale
Localisation : lieu ou elle est situee
Ambiance : description en 2 ou 3 mots
Economie : principaux secteurs d'activite
"""
    agent = create_agent(model=model, system_prompt=system_prompt)
    question = HumanMessage(content="Quelle est la capitale de la lune ?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


class CapitaleInfo(BaseModel):
    nom: str
    localisation: str
    ambiance: str
    economie: str


def demo_reponse_structuree_basemodel():
    system_prompt = """
Vous etes un auteur de science-fiction et vous devez inventer une capitale
spatiale a la demande d'un utilisateur. Respectez la structure ci-dessous.

Nom : nom de la capitale
Localisation : lieu ou elle est situee
Ambiance : description en 2 ou 3 mots
Economie : principaux secteurs d'activite
"""
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        response_format=CapitaleInfo,
    )
    question = HumanMessage(content="Quelle est la capitale de la lune ?")
    response = agent.invoke({"messages": [question]})
    print(response["structured_response"])


@tool("meteo_capitale")
def meteo_capitale(ville: str) -> str:
    """Donne la meteo d'une capitale (valeurs fixes pour test).

    Args:
        ville: nom de la capitale
    """
    print("tool meteo_capitale utilise")
    temperature = 25
    humidite = 60
    pression = 1013
    return (
        f"Meteo a {ville} : "
        f"Temperature = {temperature}°C, "
        f"Humidite = {humidite}%, "
        f"Pression = {pression} hPa"
    )


def demo_tool_meteo():
    system_prompt = "Utilisez les tools disponibles pour repondre aux questions."
    agent = create_agent(model=model, tools=[meteo_capitale], system_prompt=system_prompt)
    question = HumanMessage(content="Quelle est la meteo a Capitole lunaire ?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


def demo_sans_web_search():
    agent = create_agent(model=model)
    question = HumanMessage(content="Vos connaissances sont-elles a jour ?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Recherche une information recente sur le web via Tavily."""
    return tavily_client.search(query)


def demo_avec_web_search():
    agent = create_agent(model=model, tools=[web_search])
    question = HumanMessage(content="Qui est le president de la commune actuelle de Marrakech ?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)


def demo_sans_memoire():
    agent = create_agent(model=model)
    q1 = HumanMessage(content="Bonjour, mon nom est Sami et je suis developpeur.")
    agent.invoke({"messages": [q1]})
    q2 = HumanMessage(content="Quel est mon metier ?")
    response = agent.invoke({"messages": [q2]})
    print(response["messages"][-1].content)


def demo_avec_memoire():
    agent = create_agent(model=model, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    q1 = HumanMessage(content="Bonjour, mon nom est Sami et je suis developpeur.")
    agent.invoke({"messages": [q1]}, config)
    q2 = HumanMessage(content="Quel est mon metier ?")
    response = agent.invoke({"messages": [q2]}, config)
    print(response["messages"][-1].content)


if __name__ == "__main__":
    print("\n--- Sans system message ---")
    demo_sans_system_message()

    print("\n--- Avec system message ---")
    demo_avec_system_message()

    print("\n--- Few-shot ---")
    demo_few_shot()

    print("\n--- Reponse structuree (texte) ---")
    demo_reponse_structuree_texte()

    print("\n--- Reponse structuree (BaseModel) ---")
    demo_reponse_structuree_basemodel()

    print("\n--- Tool meteo_capitale ---")
    demo_tool_meteo()

    print("\n--- Sans web search ---")
    demo_sans_web_search()

    print("\n--- Avec web search ---")
    demo_avec_web_search()

    print("\n--- Sans memoire ---")
    demo_sans_memoire()

    print("\n--- Avec memoire ---")
    demo_avec_memoire()
