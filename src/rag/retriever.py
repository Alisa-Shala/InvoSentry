"""
RAG retriever: embeds policy/fraud-pattern documents into a FAISS index
and retrieves the most relevant ones for a given invoice query.

Uses sentence-transformers (all-MiniLM-L6-v2, small + fast, runs on CPU)
so the whole pipeline runs locally without needing an embeddings API call
per invoice — keeps thesis evaluation runs fast and cheap.
"""
from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from data.policies.policy_docs import POLICY_DOCUMENTS
from src.schemas import RetrievedContext

_MODEL_NAME = "all-MiniLM-L6-v2"


class PolicyRetriever:
    def __init__(self, documents: list[dict] | None = None):
        self.documents = documents or POLICY_DOCUMENTS
        self.model = SentenceTransformer(_MODEL_NAME)
        self._build_index()

    def _build_index(self) -> None:
        texts = [d["text"] for d in self.documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        self.index.add(np.array(embeddings, dtype="float32"))

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedContext]:
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vec, dtype="float32"), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievedContext(
                    source=doc["source"],
                    snippet=doc["text"],
                    relevance_score=float(score),
                )
            )
        return results


# Module-level singleton so the (relatively slow) embedding model and index
# are built once per process, not once per invoice.
_retriever_instance: PolicyRetriever | None = None


def get_retriever() -> PolicyRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PolicyRetriever()
    return _retriever_instance
