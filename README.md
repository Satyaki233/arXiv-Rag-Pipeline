# arXiv RAG Pipeline

A medium-difficulty Retrieval-Augmented Generation pipeline that answers
questions about recent NLP research papers, with citations. Built for learning —
each stage of RAG lives in its own small module.

Generation runs on **Claude (Haiku 4.5) via Azure AI Foundry**; embeddings and
reranking run locally (no cloud, no cost).

## Architecture

```
INDEX-TIME (once)                          QUERY-TIME (every question)
─────────────────                          ───────────────────────────
arXiv API ─▶ chunk ─▶ embed ─▶ Chroma      question ─▶ embed ─▶ vector search
(cs.CL)                        (vectors)              ─▶ rerank ─▶ relevance gate
                                                      ─▶ Claude ─▶ answer + citations
```

| Stage | File | What it does |
|-------|------|--------------|
| 1. Ingest | `src/ingest.py` | Fetch papers from the arXiv API, split abstracts into overlapping chunks |
| 2+3. Embed & store | `src/store.py` | Embed chunks (local model) and store/index in ChromaDB |
| 4. Retrieve | `src/retrieve.py` | Vector search top-20 → cross-encoder rerank top-5 → relevance gate |
| 5. Generate | `src/generate.py` | Claude answers grounded in retrieved chunks, with citations |
| 6. Evaluate | `eval/evaluate.py` | Retrieval hit-rate + MRR |

Entry points: `scripts/build_index.py` (build the index), `scripts/ask.py`
(one-shot question), `chat-app.py` (interactive chat).

## Requirements

- **Python 3.12** — on an Intel Mac, PyTorch has no wheels for 3.13, so the venv
  must be 3.12. `torch==2.2.2` and `numpy<2` are pinned in `requirements.txt` for
  the same reason (drop those pins on Apple Silicon / Linux).
- An **Azure AI Foundry** resource with a Claude model deployed (for generation).

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env        # then fill in your Foundry credentials
```

Configure `.env` (see `.env.example` for details):

```bash
LLM_PROVIDER=foundry
AZURE_FOUNDRY_API_KEY=...
# Provide ONE of these:
AZURE_FOUNDRY_RESOURCE=your-resource-name          # bare name, OR
# AZURE_FOUNDRY_BASE_URL=https://<resource>.services.ai.azure.com/anthropic/
CLAUDE_MODEL=claude-haiku-4-5                       # your Foundry deployment name
```

> To use the direct Anthropic API instead of Foundry, set `LLM_PROVIDER=anthropic`
> and `ANTHROPIC_API_KEY=...`.

## Run

All commands use the venv's Python. Stages 1–4 (build + retrieval) need **no**
API key — only generation calls Foundry.

```bash
# 1. Build the index — fetch papers, embed, store (run once, ~1 min)
.venv/bin/python -m scripts.build_index

# 2. Ask a one-shot question
.venv/bin/python -m scripts.ask "What methods reduce hallucination in LLMs?"

# 3. Interactive chat (remembers the conversation)
.venv/bin/python chat-app.py

# 4. Measure retrieval quality
.venv/bin/python -m eval.evaluate
```

### Chat commands (`chat-app.py`)

| Command | Effect |
|---------|--------|
| `/sources` | toggle the source list under each answer |
| `/reset` | clear the conversation history |
| `/exit` | quit (Ctrl-D / Ctrl-C also work) |

## Tuning knobs (all in `config.py`)

| Knob | Stage | Effect |
|------|-------|--------|
| `ARXIV_CATEGORY`, `ARXIV_MAX_PAPERS` | ingest | corpus topic & size |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | ingest | how abstracts are split (big quality impact) |
| `EMBEDDING_MODEL` | embed | search quality vs. speed (`BAAI/bge-base-en-v1.5` for better) |
| `TOP_K_RETRIEVE`, `TOP_K_RERANK` | retrieve | retrieve-wide, rerank-narrow funnel |
| `USE_RERANKER` | retrieve | the biggest quality lever — A/B it against the eval |
| `RERANK_SCORE_THRESHOLD` | retrieve | relevance gate: lower = more lenient, higher = stricter |
| `MAX_VECTOR_DISTANCE` | retrieve | relevance gate for the no-reranker path |
| `CLAUDE_MODEL`, `MAX_TOKENS` | generate | answer quality vs. cost |
| `LLM_PROVIDER` | generate | `foundry` (Azure) or `anthropic` (direct) |

## Highlights 

1. **Citations** — every answer points back to specific papers.
2. **Reranking** — a cross-encoder re-scores candidates (biggest quality lever).
3. **Relevance gating / abstention** — off-topic questions return *no answer and
   no sources* instead of confidently citing irrelevant papers.
4. **Evaluation** — hit-rate / MRR, so "did it work" isn't just vibes.

## Notes

- ChromaDB telemetry is disabled in `store.py`; its harmless version-mismatch
  warning is muted in `config.py`.
- `chat-app.py` sets `TOKENIZERS_PARALLELISM=false` itself. For `ask`/`evaluate`,
  prefix the command with it (or add it to `.env`) to silence the HF fork warning.

## Learning extensions (next steps)

- Swap the naive character splitter for a sentence-aware one.
- Build a harder eval set (paraphrased questions) so the reranker's value shows.
- Make retrieval history-aware (rewrite follow-ups into standalone queries).
- Ingest full-text PDFs instead of just abstracts (harder chunking).
- Add metadata filtering (e.g. by publication date).
- Then: move to the SEC 10-K project (tables + prose).
