"""Retrieval evaluation — the difference between a demo and an engineered system.

We measure retrieval hit-rate with a self-supervised trick: for each sampled
paper, we turn its title into a query and check whether that paper's own chunks
come back in the top-k. If the retriever can't find the paper its own title
describes, it won't find anything harder.

Run (after building the index):
    python -m eval.evaluate

Compare USE_RERANKER = True vs False in config.py to see the reranker's effect.
"""
from __future__ import annotations

import random

import config
from src import retrieve, store


def build_eval_queries(n: int = 30) -> list[tuple[str, str]]:
    """Sample indexed papers -> (title-as-query, expected paper_id)."""
    collection = store.get_collection()
    data = collection.get(include=["metadatas"])
    # Deduplicate to one entry per paper.
    by_paper = {m["paper_id"]: m["title"] for m in data["metadatas"]}
    papers = list(by_paper.items())
    random.seed(42)
    random.shuffle(papers)
    return [(title, pid) for pid, title in papers[:n]]


def evaluate(n: int = 30) -> None:
    queries = build_eval_queries(n)
    hits_at_k = 0
    reciprocal_ranks = 0.0

    for title, expected_id in queries:
        results = retrieve.retrieve(title)
        retrieved_ids = [h["metadata"]["paper_id"] for h in results]
        if expected_id in retrieved_ids:
            hits_at_k += 1
            rank = retrieved_ids.index(expected_id) + 1
            reciprocal_ranks += 1.0 / rank

    n = len(queries)
    print(f"Evaluated {n} queries "
          f"(reranker={'on' if config.USE_RERANKER else 'off'})")
    print(f"  Hit-rate@{config.TOP_K_RERANK}: {hits_at_k / n:.2%}")
    print(f"  MRR@{config.TOP_K_RERANK}:      {reciprocal_ranks / n:.3f}")


if __name__ == "__main__":
    evaluate()
