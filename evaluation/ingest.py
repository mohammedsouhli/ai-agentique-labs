"""
ingest.py - Prétraitement et vectorisation des documents touristiques
Charge les documents, les découpe en chunks et crée l'index FAISS.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_and_index_documents(docs_path="documents/"):
    """Charge les documents PDF, les vectorise et retourne le retriever."""

    # Chargement des fichiers PDF
    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()
    print(f"[ingest] {len(documents)} pages PDF chargées")

    # Découpage en chunks de 500 caractères avec 50 de chevauchement
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"[ingest] {len(chunks)} chunks créés")

    # Vectorisation avec le modèle HuggingFace léger all-MiniLM-L6-v2
    print("[ingest] Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Création de l'index FAISS en mémoire
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[ingest] Index FAISS créé avec succès")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever, len(documents), len(chunks)
