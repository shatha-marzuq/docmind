from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile, os, shutil
from pathlib import Path

from core.document_processor import process_uploaded_file, chunk_documents
from core.embeddings import get_embeddings, add_documents_to_store, clear_vector_store
from core.rag_chain import answer_question
from core.hybrid_search import HybridRetriever, BM25Retriever

app = FastAPI(title="DocMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (في الإنتاج استخدم Redis أو database) ──
_state = {
    "embeddings": None,
    "vector_store": None,
    "all_chunks": [],
    "hybrid_retriever": None,
    "file_names": [],
}

class QuestionRequest(BaseModel):
    question: str
    search_mode: str = "Hybrid"   # Hybrid | Semantic | Keyword
    top_k: int = 5

class ResetResponse(BaseModel):
    success: bool

@app.get("/")
def root():
    return {"status": "DocMind API running"}

@app.get("/status")
def status():
    return {
        "documents_loaded": len(_state["file_names"]) > 0,
        "file_names": _state["file_names"],
        "chunks_count": len(_state["all_chunks"]),
    }

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    supported = {".pdf", ".txt", ".docx", ".md"}
    all_chunks = []
    file_names = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in supported:
            raise HTTPException(400, f"نوع الملف غير مدعوم: {ext}")

        # حفظ مؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = process_uploaded_file_path(tmp_path, file.filename)
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
            file_names.append(file.filename)
        finally:
            os.unlink(tmp_path)

    if not all_chunks:
        raise HTTPException(400, "لم يتم معالجة أي مستند")

    # بناء vector store
    if _state["embeddings"] is None:
        _state["embeddings"] = get_embeddings()

    clear_vector_store()
    _state["vector_store"] = add_documents_to_store(all_chunks, _state["embeddings"])
    _state["hybrid_retriever"] = HybridRetriever(_state["vector_store"], all_chunks)
    _state["all_chunks"] = all_chunks
    _state["file_names"] = file_names

    return {
        "success": True,
        "file_names": file_names,
        "chunks_count": len(all_chunks),
    }

@app.post("/ask")
def ask(req: QuestionRequest):
    if not _state["file_names"]:
        raise HTTPException(400, "لم يتم رفع أي مستند بعد")

    vector_store = _state["vector_store"]
    hybrid = _state["hybrid_retriever"]
    all_chunks = _state["all_chunks"]
    top_k = req.top_k

    if req.search_mode == "Hybrid":
        docs_with_scores = hybrid.retrieve(req.question, top_k)
    elif req.search_mode == "Semantic":
        docs_with_scores = vector_store.similarity_search_with_score(req.question, k=top_k)
    else:  # Keyword
        bm25 = BM25Retriever(all_chunks)
        docs_with_scores = bm25.search(req.question, k=top_k)

    result = answer_question(req.question, docs_with_scores)
    return result

@app.post("/reset")
def reset():
    clear_vector_store()
    _state["embeddings"] = None
    _state["vector_store"] = None
    _state["all_chunks"] = []
    _state["hybrid_retriever"] = None
    _state["file_names"] = []
    return {"success": True}


# ── Helper: معالجة الملف من path مباشرة ──
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

def process_uploaded_file_path(file_path: str, original_name: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported: {ext}")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_name"] = original_name
    return docs
