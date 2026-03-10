import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile

COLLECTION_NAME = "docmind_collection"
_CHROMA_DIR = tempfile.mkdtemp()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def add_documents_to_store(chunks: List[Document], embeddings) -> Chroma:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=_CHROMA_DIR,
    )
    return vector_store

def similarity_search(vector_store: Chroma, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

def clear_vector_store():
    import shutil
    global _CHROMA_DIR
    if os.path.exists(_CHROMA_DIR):
        shutil.rmtree(_CHROMA_DIR)
    _CHROMA_DIR = tempfile.mkdtemp()
