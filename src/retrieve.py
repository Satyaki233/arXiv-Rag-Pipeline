"""Stage 4 — RETRIEVAL (+ optional reranking).

Naive vector search returns the top-k by embedding similarity, which is fast but
imprecise. A cross-encoder reranker re-scores each (query, chunk) pair jointly
and reorders them. This two-tier "retrieve wide, rerank narrow" pattern is the
single biggest quality lever in a medium-difficulty RAG system.
"""
from __future__ import annotations

from functools import lru_cache

import config
from src import store


@lru_cache(maxsize=1)
def _reranker():
    """Load the cross-encoder once (it's a few hundred MB)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates with a cross-encoder and keep the best top_k."""
    pairs = [(query, h["text"]) for h in hits]
    scores = _reranker().predict(pairs)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:top_k]


def retrieve(query: str,
             top_k_retrieve: int = config.TOP_K_RETRIEVE,
             top_k_rerank: int = config.TOP_K_RERANK,
             use_reranker: bool = config.USE_RERANKER) -> list[dict]:
    """Full retrieval: wide vector search, narrow rerank, then a relevance gate.

    The gate is what makes off-topic questions return an EMPTY list (→ no
    answer, no sources) instead of the top-5 of an irrelevant corpus.
    """
    hits = store.search(query, top_k=top_k_retrieve)
    if not hits:
        return []
    if use_reranker:
        hits = rerank(query, hits, top_k_rerank)
        return [h for h in hits if h["rerank_score"] >= config.RERANK_SCORE_THRESHOLD]
    # No reranker: fall back to a (fuzzier) vector-distance gate.
    hits = hits[:top_k_rerank]
    return [h for h in hits if h["distance"] <= config.MAX_VECTOR_DISTANCE]


if __name__ == "__main__":
    q = "How do researchers reduce hallucination in large language models?"
    for i, hit in enumerate(retrieve(q), 1):
        print(f"\n[{i}] {hit['metadata']['title']}")
        print(f"    {hit['text'][:180]}...")
