"""
TP Lab 5 - Agent simple visualisable dans LangGraph Studio.
Un agent minimal avec un seul tool, pense pour etre inspecte noeud par noeud
dans Studio (nodes, edges, inputs/outputs).
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def lore_search(query: str) -> str:
    """Recherche des informations dans le lore d'une histoire fictive."""
    return (
        "L'heroine, Elena, est une exploratrice qui a decouvert une carte "
        "menant a une cite engloutie remplie de technologies anciennes."
    )


llm = ChatOllama(model="llama3.2:3b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[lore_search],
    system_prompt=(
        "Tu es un assistant specialise dans l'analyse de recits. "
        "Utilise l'outil lore_search pour repondre aux questions en te basant "
        "sur l'histoire. Reponds de maniere precise et cite le contexte trouve."
    ),
)
