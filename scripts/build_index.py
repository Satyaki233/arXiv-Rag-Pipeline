"""Build the vector index: fetch arXiv papers -> chunk -> embed -> store.

Run once (or whenever you want to refresh the corpus):
    python -m scripts.build_index
"""
import config
from src import ingest, store


def main() -> None:
    print(f"Fetching up to {config.ARXIV_MAX_PAPERS} papers from arXiv "
          f"({config.ARXIV_CATEGORY})...")
    papers = ingest.fetch_papers()
    print(f"Fetched {len(papers)} papers.")

    docs, metas, ids = ingest.build_chunks(papers)
    print(f"Built {len(docs)} chunks. Embedding + indexing "
          f"(first run downloads the embedding model)...")

    store.index(docs, metas, ids)
    print("Done. Ask questions with:  python -m scripts.ask \"your question\"")


if __name__ == "__main__":
    main()
