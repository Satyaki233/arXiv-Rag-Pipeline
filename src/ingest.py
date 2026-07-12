"""Stage 1 — INGESTION.

Fetch papers from the arXiv API, then split each abstract into overlapping
chunks. Each chunk carries metadata (title, authors, paper id, url) so we can
cite sources later.

This is where the two most important RAG decisions live: what to chunk and how.
"""
from __future__ import annotations

import arxiv

import config


def fetch_papers(category: str = config.ARXIV_CATEGORY,
                 max_papers: int = config.ARXIV_MAX_PAPERS) -> list[dict]:
    """Pull recent papers in a category from the arXiv API."""
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3)

    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.get_short_id(),
            "title": result.title.strip().replace("\n", " "),
            "authors": ", ".join(a.name for a in result.authors[:5]),
            "abstract": result.summary.strip().replace("\n", " "),
            "url": result.entry_id,
            "published": result.published.date().isoformat(),
        })
    return papers


def chunk_text(text: str,
               size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows.

    Overlap keeps a sentence that straddles a boundary retrievable from both
    chunks. This naive splitter is deliberate — swapping in a smarter,
    sentence-aware splitter later is a good exercise and shows up in eval.
    """
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_chunks(papers: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """Turn papers into (documents, metadatas, ids) ready for the vector store."""
    documents, metadatas, ids = [], [], []
    for paper in papers:
        for i, chunk in enumerate(chunk_text(paper["abstract"])):
            documents.append(chunk)
            metadatas.append({
                "paper_id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "url": paper["url"],
                "published": paper["published"],
                "chunk_index": i,
            })
            ids.append(f"{paper['id']}::{i}")
    return documents, metadatas, ids


if __name__ == "__main__":
    papers = fetch_papers(max_papers=5)
    docs, metas, ids = build_chunks(papers)
    print(f"{len(papers)} papers -> {len(docs)} chunks")
    print("\nExample chunk:\n", docs[0][:300])
    print("\nMetadata:\n", metas[0])
