"""
TP Lab 5 - Multi-Agents : un agent principal delegue a deux sous-agents
specialises, chacun expose comme un tool a l'agent principal.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv(override=True)

# Les sous-agents n'ont qu'un seul calcul simple a faire : le petit modele local suffit.
sub_model = ChatOllama(model="llama3.2:3b", temperature=0)
# L'agent principal doit orchestrer plusieurs tools : un modele plus capable
# evite les boucles/hallucinations qu'on observe avec un modele 3B sur ce role.
orchestrator_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x**0.5


@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x**2


subagent_racine = create_agent(model=sub_model, tools=[square_root])
subagent_carre = create_agent(model=sub_model, tools=[square])


@tool
def call_subagent_racine(x: float) -> str:
    """Call the sub-agent that computes the square root of a number"""
    response = subagent_racine.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square root of {x}")]}
    )
    return response["messages"][-1].content


@tool
def call_subagent_carre(x: float) -> str:
    """Call the sub-agent that computes the square of a number"""
    response = subagent_carre.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square of {x}")]}
    )
    return response["messages"][-1].content


main_agent = create_agent(
    model=orchestrator_model,
    tools=[call_subagent_racine, call_subagent_carre],
    system_prompt=(
        "You are a helpful assistant who delegates to sub-agents to "
        "calculate the square root or the square of a number. Call exactly "
        "one sub-agent per calculation needed, then answer the user directly "
        "with the final result."
    ),
)


if __name__ == "__main__":
    question = "What is the square root of 456?"
    response = main_agent.invoke({"messages": [HumanMessage(content=question)]})
    print(response["messages"][-1].content)
