"""
graph.py - Architecture du graphe LangGraph
Définit le State, les noeuds et les edges de l'agent agentique.
"""

from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Définition du State partagé entre tous les noeuds du graphe
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(tools):
    """Construit et compile le graphe LangGraph avec les outils fournis."""

    # LLM Groq connecté aux outils
    llm = ChatOllama(model="llama3.2:3b")
    llm_with_tools = llm.bind_tools(tools)

    # Noeud agent : reçoit les messages et décide d'appeler un outil ou de répondre
    def agent_node(state: AgentState):
        system = SystemMessage(content="""Tu es un assistant expert en tourisme au Maroc.
Tu aides les voyageurs à planifier leur séjour en fournissant des informations précises
sur les villes, attractions, hébergements, transports et gastronomie.
Utilise l'outil de recherche pour trouver des informations pertinentes avant de répondre.
Réponds toujours en français de manière claire et structurée.""")
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Fonction de décision : aller vers le bon outil ou terminer
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return last_message.tool_calls[0]['name']
        return END

    # Construction du graphe avec un noeud par outil
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("recherche_tourisme", ToolNode([tools[0]]))
    workflow.add_node("conseils_pratiques", ToolNode([tools[1]]))
    workflow.add_node("recherche_web", ToolNode([tools[2]]))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "recherche_tourisme": "recherche_tourisme",
            "conseils_pratiques": "conseils_pratiques",
            "recherche_web": "recherche_web",
            END: END
        }
    )
    workflow.add_edge("recherche_tourisme", "agent")
    workflow.add_edge("conseils_pratiques", "agent")
    workflow.add_edge("recherche_web", "agent")

    return workflow.compile()
