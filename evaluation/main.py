"""
main.py - Point d'entrée principal du système Agentic RAG
Lance le pipeline complet : ingestion → outils → graphe → évaluation.
"""

from dotenv import load_dotenv
from ingest import load_and_index_documents
from tools import create_tools
from graph import build_graph
from run_evaluation import run_evaluation

# Chargement des variables d'environnement (clé API Groq)
load_dotenv("../.env")


def main():
    print("=== Système Agentic RAG - Tourisme Maroc ===\n")

    # Étape 1 : Prétraitement et vectorisation des documents
    print("Étape 1 : Chargement et indexation des documents...")
    retriever, n_docs, n_chunks = load_and_index_documents()
    print(f"  → {n_docs} documents, {n_chunks} chunks\n")

    # Étape 2 : Création des outils
    print("Étape 2 : Création des outils...")
    tools = create_tools(retriever)
    print(f"  → {len(tools)} outils : {[t.name for t in tools]}\n")

    # Étape 3 : Construction du graphe LangGraph
    print("Étape 3 : Construction du graphe LangGraph...")
    app = build_graph(tools)
    print("  → Graphe compilé\n")

    # Étape 4 : Évaluation sur 20 questions
    print("Étape 4 : Évaluation du système...\n")
    run_evaluation(app)


if __name__ == "__main__":
    main()
