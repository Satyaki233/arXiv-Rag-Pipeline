"""Central configuration for the RAG pipeline.

Everything tunable lives here so experiments are one-line changes.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env into environment

# Silence ChromaDB's telemetry: it's off (see store.py) but 0.5.23 still logs a
# harmless posthog version-mismatch error. This mutes that logger for good.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = str(DATA_DIR / "chroma")  # vector DB persists here

# --- Ingestion (arXiv API) ---
ARXIV_CATEGORY = "cs.CL"   # NLP / computational linguistics
ARXIV_MAX_PAPERS = 300     # start small; raise once the pipeline works

# --- Chunking ---
# Abstracts are short, so we chunk lightly. For full-text PDFs later,
# these are the two knobs that most affect quality.
CHUNK_SIZE = 700           # target characters per chunk (~150-180 tokens)
CHUNK_OVERLAP = 100        # characters shared between adjacent chunks

# --- Embedding ---
# all-MiniLM-L6-v2: small, fast, CPU-friendly. Good starting point.
# Upgrade to "BAAI/bge-base-en-v1.5" for better quality later.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K_RETRIEVE = 20        # candidates pulled from the vector store
TOP_K_RERANK = 5           # kept after reranking (fed to the LLM)
USE_RERANKER = True        # the "medium difficulty" quality lever
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Relevance gates: drop chunks that aren't actually relevant, so an off-topic
# question returns NO sources instead of the top-5-of-nothing.
# Calibrated from observed scores: relevant chunks score ~+3 on the reranker
# (~0.54 cosine distance); clearly-irrelevant ones score ~-10 (~0.80+ distance).
RERANK_SCORE_THRESHOLD = -5.0   # keep reranked chunks scoring >= this
MAX_VECTOR_DISTANCE = 0.7       # (no-reranker path) keep chunks with distance <= this

# --- Generation (Claude) ---
# Where Claude runs:
#   "foundry"   -> route through Microsoft/Azure AI Foundry (AnthropicFoundry client)
#   "anthropic" -> direct Anthropic API
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "foundry")

# Azure Foundry credentials (only used when LLM_PROVIDER == "foundry").
AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY")
# Provide EITHER a resource name (builds https://<resource>.services.ai.azure.com/anthropic/)
# OR a full base URL copied verbatim from the Foundry portal. base_url wins if set.
AZURE_FOUNDRY_RESOURCE = os.getenv("AZURE_FOUNDRY_RESOURCE")
AZURE_FOUNDRY_BASE_URL = os.getenv("AZURE_FOUNDRY_BASE_URL")

# On Foundry this is your *deployment* name (often the same as the Claude id).
# Haiku is plenty for grounded Q&A — cheap + fast. Bump to claude-sonnet-5 or
# claude-opus-4-8 only if answer quality actually needs it.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
MAX_TOKENS = 2048          # headroom for a structured, explanatory answer

COLLECTION_NAME = "arxiv_cs_cl"
