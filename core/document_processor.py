import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def load_document(file_path: str) -> List[Document]:
    """Load a document based on its extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt" or ext == ".md":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def process_uploaded_file(uploaded_file) -> List[Document]:
    """Process a Streamlit uploaded file and return chunks."""
    ext = Path(uploaded_file.name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"File type not supported: {ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)
        # Add source metadata
        for doc in docs:
            doc.metadata["source_name"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks
