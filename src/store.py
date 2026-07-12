"""Stages 2 & 3 — EMBEDDING + VECTOR STORE.

ChromaDB handles both: we hand it a local embedding function, and it embeds +
stores + indexes for us. Persisted to disk under data/chroma so you only build
the index once.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import config


def get_collection() -> chromadb.Collection:
    """Open (or create) the persistent collection with our embedding model."""
    # anonymized_telemetry=False disables Chroma's usage analytics, which also
    # silences the harmless "Failed to send telemetry event" warnings.
    client = chromadb.PersistentClient(
        path=config.CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )


def index(documents: list[str], metadatas: list[dict], ids: list[str],
          batch_size: int = 100) -> None:
    """Embed and store chunks. Idempotent: re-running upserts by id."""
    collection = get_collection()
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
    print(f"Indexed {len(documents)} chunks. Collection size: {collection.count()}")


def search(query: str, top_k: int = config.TOP_K_RETRIEVE) -> list[dict]:
    """Embed the query and return the top-k most similar chunks."""
    collection = get_collection()
    res = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits
