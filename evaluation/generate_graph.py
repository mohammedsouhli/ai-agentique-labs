"""
generate_graph.py - Génération et sauvegarde du graphe LangGraph
Crée une image PNG du graphe de l'agent pour la documentation.
"""

from dotenv import load_dotenv
from ingest import load_and_index_documents
from tools import create_tools
from graph import build_graph

load_dotenv("../.env")


def save_graph_image(output_path="graph.png"):
    """Génère et sauvegarde le graphe LangGraph en image PNG."""
    retriever, _, _ = load_and_index_documents()
    tools = create_tools(retriever)
    app = build_graph(tools)

    try:
        image_data = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Graphe sauvegardé : {output_path}")
    except Exception as e:
        # Sauvegarde du graphe en format Mermaid si PNG échoue
        mermaid = app.get_graph().draw_mermaid()
        with open("graph.mmd", "w") as f:
            f.write(mermaid)
        print(f"Graphe sauvegardé en format Mermaid : graph.mmd")
        print(mermaid)


if __name__ == "__main__":
    save_graph_image()
