"""
TP Lab 7 - Agent Human-In-The-Loop (HITL)
L'agent lit un email (etat) et propose une reponse par email, mais s'arrete
avant d'envoyer (send_email) pour demander une validation humaine : le
middleware HumanInTheLoopMiddleware intercepte ce tool et attend une decision
(approve / reject / edit) avant de reprendre l'execution.
"""

import sys

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)


class EmailState(AgentState):
    email: str


@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read the email stored in the agent's state."""
    return runtime.state["email"]


@tool
def send_email(body: str) -> str:
    """Send an email with the given body. (Simule, n'envoie rien reellement.)"""
    print(f"[simulation] Email envoye : {body!r}")
    return "Email sent"


model = ChatOllama(model="llama3.2:3b", temperature=0)

INCOMING_EMAIL = (
    "Bonjour Sara, je vais etre en retard pour notre reunion de demain. "
    "Pouvons-nous la reprogrammer ? Cordialement, Sofia"
)


def build_agent():
    return create_agent(
        model=model,
        tools=[read_email, send_email],
        state_schema=EmailState,
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"read_email": False, "send_email": True},
                description_prefix="Tool execution requires approval",
            )
        ],
    )


def start_flow(agent, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Lis mon email et envoie une reponse tout de suite, "
                        "dans le meme fil de discussion."
                    )
                )
            ],
            "email": INCOMING_EMAIL,
        },
        config=config,
    )
    interrupt = response["__interrupt__"][0]
    proposed_body = interrupt.value["action_requests"][0]["args"]["body"]
    print(f"Reponse proposee par l'agent (en attente d'approbation) : {proposed_body!r}")
    return config


def report_outcome(response) -> None:
    """Une decision peut soit conclure le tour (message final), soit relancer
    l'agent vers une nouvelle proposition qui necessite elle aussi une
    approbation (nouveau __interrupt__) - c'est le cas typique apres un reject."""
    if "__interrupt__" in response:
        new_proposal = response["__interrupt__"][0].value["action_requests"][0]["args"]["body"]
        print(f"L'agent relance une nouvelle proposition (en attente d'approbation) : {new_proposal!r}")
    else:
        print(response["messages"][-1].content)


def scenario_approve(agent):
    print("\n=== Scenario 1 : approve ===")
    config = start_flow(agent, thread_id="hitl-approve")
    response = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )
    report_outcome(response)


def scenario_reject(agent):
    print("\n=== Scenario 2 : reject ===")
    config = start_flow(agent, thread_id="hitl-reject")
    response = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "reject", "message": "J'annule notre rendez-vous."}
                ]
            }
        ),
        config=config,
    )
    report_outcome(response)


def scenario_edit(agent):
    print("\n=== Scenario 3 : edit ===")
    config = start_flow(agent, thread_id="hitl-edit")
    edited_body = "Je suis desolee mais je dois annuler notre rendez-vous, je ne serai pas libre. Sara"
    response = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email",
                            "args": {"body": edited_body},
                        },
                    }
                ]
            }
        ),
        config=config,
    )
    report_outcome(response)


if __name__ == "__main__":
    hitl_agent = build_agent()
    scenario_approve(hitl_agent)
    scenario_reject(hitl_agent)
    scenario_edit(hitl_agent)
