"""
TP Lab 6 - Contexte d'un agent
Le contexte est immuable pendant un invoke() : il est fourni au moment de
l'appel via `context=`, et lu par les tools via `runtime.context`.
Contrairement a l'etat, il n'est jamais modifie ni persiste par l'agent.
"""

import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)


@dataclass
class FoodContext:
    favourite_dish: str = "pizza"
    least_favourite_dish: str = "liver"


model = ChatOllama(model="llama3.2:3b", temperature=0)


def demo_agent_sans_contexte() -> None:
    """Sans tool, l'agent ne peut pas acceder au contexte : il ne peut que deviner."""
    agent = create_agent(model=model, context_schema=FoodContext)
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite dish?")]},
        context=FoodContext(),
    )
    print(response["messages"][-1].content)


@tool
def get_favourite_dish(runtime: ToolRuntime) -> str:
    """Get the user's favourite dish."""
    return runtime.context.favourite_dish


@tool
def get_least_favourite_dish(runtime: ToolRuntime) -> str:
    """Get the user's least favourite dish."""
    return runtime.context.least_favourite_dish


context_agent = create_agent(
    model=model,
    tools=[get_favourite_dish, get_least_favourite_dish],
    context_schema=FoodContext,
    system_prompt=(
        "Use the tools to answer questions about the user's food preferences. "
        "State the value returned by the tool explicitly and directly."
    ),
)


def demo_agent_avec_contexte(context: FoodContext) -> str:
    response = context_agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite dish?")]},
        context=context,
    )
    return response["messages"][-1].content


if __name__ == "__main__":
    print("--- Sans contexte accessible (l'agent devine) ---")
    demo_agent_sans_contexte()

    print("\n--- Avec contexte par defaut (pizza) ---")
    print(demo_agent_avec_contexte(FoodContext()))

    print("\n--- Changement de contexte (couscous) ---")
    print(demo_agent_avec_contexte(FoodContext(favourite_dish="couscous")))
