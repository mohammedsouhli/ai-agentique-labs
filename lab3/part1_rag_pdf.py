"""
TP Lab 3 - Partie 1 : RAG sur un document PDF
Charge le manuel employe (PDF), le decoupe en chunks, l'indexe dans une base
vectorielle en memoire, puis expose un agent capable de repondre a des
questions en s'appuyant sur ce document (recherche semantique).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

PDF_PATH = Path(__file__).parent / "acmecorp-employee-handbook.pdf"

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()
print(f"Pages chargees : {len(pages)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
chunks = splitter.split_documents(pages)
print(f"Chunks generes : {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=chunks)


def preview_similarity_search(question: str) -> None:
    results = vector_store.similarity_search(question)
    print(f"\nMeilleur passage trouve pour : {question!r}")
    print(results[0].page_content[:400])


@tool
def search_handbook(query: str) -> str:
    """Cherche dans le manuel employe le passage le plus pertinent pour la requete."""
    results = vector_store.similarity_search(query)
    return results[0].page_content


model = ChatOllama(model="llama3.2:3b", temperature=0)

handbook_agent = create_agent(
    model=model,
    tools=[search_handbook],
    system_prompt=(
        "Vous etes un agent RH qui repond aux questions des employes en vous "
        "basant uniquement sur le manuel employe accessible via l'outil "
        "search_handbook. Si l'information n'y figure pas, dites-le."
    ),
)


def ask(question: str) -> str:
    response = handbook_agent.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content


if __name__ == "__main__":
    preview_similarity_search(
        "How many days of vacation does an employee get in their first year?"
    )

    print("\n--- Reponse de l'agent RAG ---")
    print(ask("How many days of vacation does an employee get in their first year?"))
