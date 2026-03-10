import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb


CHROMA_PERSIST_DIR = None
COLLECTION_NAME = "docmind_collection"


def get_embeddings():
    """Load HuggingFace embeddings (free, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings) -> Chroma:
    """Get or create ChromaDB vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_documents_to_store(chunks: List[Document], embeddings) -> Chroma:
    """Add document chunks to ChromaDB."""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


def similarity_search(vector_store: Chroma, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """Semantic similarity search with scores."""
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def clear_vector_store():
    """Clear all documents from the vector store."""
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
