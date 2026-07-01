"""
TP Lab 6 - Etat d'un agent
Contrairement au contexte (immuable, fourni a chaque invoke), l'etat est
mutable et persiste entre les tours grace a un checkpointer. Un tool peut
le modifier (Command) et un autre peut le lire (runtime.state).
"""

import sys

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)


class FoodState(AgentState):
    favourite_dish: str


model = ChatOllama(model="llama3.2:3b", temperature=0)


@tool
def update_favourite_dish(favourite_dish: str, runtime: ToolRuntime) -> Command:
    """Update the user's favourite dish in the agent's state once they reveal it."""
    return Command(
        update={
            "favourite_dish": favourite_dish,
            "messages": [
                ToolMessage(
                    "Successfully updated favourite dish",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def read_favourite_dish(runtime: ToolRuntime) -> str:
    """Read the user's favourite dish from the agent's state."""
    try:
        return runtime.state["favourite_dish"]
    except KeyError:
        return "No favourite dish found in state"


state_agent = create_agent(
    model=model,
    tools=[update_favourite_dish, read_favourite_dish],
    checkpointer=InMemorySaver(),
    state_schema=FoodState,
    system_prompt=(
        "Use the tools to update or read the user's favourite dish from state. "
        "Only confirm the action taken or state the value read, in one short "
        "sentence. Do not suggest recipes or give unrelated information."
    ),
)


def ask(message: str, thread_id: str = "1") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    response = state_agent.invoke({"messages": [HumanMessage(content=message)]}, config)
    return response["messages"][-1].content


if __name__ == "__main__":
    print("--- Mise a jour de l'etat ---")
    print(ask("My favourite dish is couscous"))

    print("\n--- Lecture de l'etat (meme thread) ---")
    print(ask("What's my favourite dish?"))
