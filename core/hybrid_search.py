"""
Phase 2: Hybrid Search = BM25 (keyword) + Semantic (vectors) + Reranking
هذا اللي يفرق DocMind عن 90% من مشاريع RAG العادية
"""

from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np


# ── BM25 Keyword Search ───────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        tokenized = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer — splits on spaces + lowercases."""
        return text.lower().split()

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Return top-k docs with BM25 scores."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))

        return results


# ── Hybrid Search (RRF Fusion) ────────────────────────────────────────────────

def reciprocal_rank_fusion(
    semantic_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion — يدمج نتيجتين بذكاء.
    RRF Score = Σ weight / (k + rank)
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Score semantic results
    for rank, (doc, _) in enumerate(semantic_results):
        key = doc.page_content[:100]  # unique key
        scores[key] = scores.get(key, 0) + semantic_weight / (k + rank + 1)
        doc_map[key] = doc

    # Score BM25 results
    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + bm25_weight / (k + rank + 1)
        doc_map[key] = doc

    # Sort by fused score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [(doc_map[key], scores[key]) for key in sorted_keys]


# ── Reranker ──────────────────────────────────────────────────────────────────

class SimpleReranker:
    """
    Reranker بسيط بدون API خارجي.
    يعيد ترتيب النتائج بناءً على:
    1. تكرار كلمات السؤال في المقطع
    2. طول المقطع (نفضل المقاطع المتوسطة)
    3. درجة RRF الأصلية
    """

    def rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:

        query_words = set(query.lower().split())
        reranked = []

        for doc, base_score in docs_with_scores:
            content = doc.page_content.lower()
            words = content.split()

            # Feature 1: keyword overlap ratio
            overlap = len(query_words & set(words)) / max(len(query_words), 1)

            # Feature 2: length penalty (favor 100-500 word chunks)
            length = len(words)
            length_score = 1.0 if 100 <= length <= 500 else 0.7

            # Feature 3: exact phrase bonus
            phrase_bonus = 0.2 if query.lower() in content else 0.0

            # Combined score
            final_score = (base_score * 0.5) + (overlap * 0.3) + (length_score * 0.1) + phrase_bonus
            reranked.append((doc, final_score))

        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


# ── Main Hybrid Retriever ─────────────────────────────────────────────────────

class HybridRetriever:
    """الواجهة الرئيسية للـ Hybrid Search + Reranking."""

    def __init__(self, vector_store, all_chunks: List[Document]):
        self.vector_store = vector_store
        self.bm25 = BM25Retriever(all_chunks)
        self.reranker = SimpleReranker()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Full pipeline: Semantic + BM25 → RRF Fusion → Rerank."""

        # Step 1: Semantic search
        semantic_results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)

        # Step 2: BM25 keyword search
        bm25_results = self.bm25.search(query, k=top_k * 2)

        # Step 3: RRF Fusion
        fused = reciprocal_rank_fusion(semantic_results, bm25_results)

        # Step 4: Rerank
        reranked = self.reranker.rerank(query, fused, top_k=top_k)

        return reranked
