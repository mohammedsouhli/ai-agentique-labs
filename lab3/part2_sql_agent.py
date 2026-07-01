"""
TP Lab 3 - Partie 2 : Agent SQL sur une base SQLite (Chinook)
Un agent interroge la base Chinook via un outil sql_query et renvoie les
resultats en langage naturel.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama

load_dotenv(override=True)

DB_PATH = Path(__file__).parent / "Chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


@tool
def sql_query(query: str) -> str:
    """Execute une requete SQL sur la base Chinook et retourne le resultat brut."""
    try:
        print(f"Requete SQL executee : {query}")
        return db.run(query)
    except Exception as exc:
        return f"Erreur SQL : {exc}"


model = ChatOllama(model="llama3.2:3b", temperature=0)

SYSTEM_PROMPT = f"""Vous etes un expert SQL qui interroge une base de donnees musicale (Chinook).

Regles :
- Utilisez uniquement l'outil sql_query pour obtenir des donnees.
- N'utilisez que les colonnes qui existent reellement dans le schema ci-dessous.
- Si l'information demandee n'existe pas, dites-le clairement, ne devinez pas.
- Repondez toujours en langage naturel, jamais en SQL brut ni en resultat brut de requete.

Schema de la base :
{db.get_table_info()}
"""

sql_agent = create_agent(model=model, tools=[sql_query], system_prompt=SYSTEM_PROMPT)


def ask(question: str) -> str:
    response = sql_agent.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content


if __name__ == "__main__":
    print(ask("Give me the first 5 artists in the database"))
    print()
    print(ask("How many tracks are there in total?"))
