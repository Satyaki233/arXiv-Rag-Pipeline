"""Stage 5 — GENERATION with Claude.

We give Claude the retrieved chunks as numbered context and ask it to answer
using only that context, citing sources by number. Grounding + citations are
what separate a RAG answer from the model just talking from memory.
"""
from __future__ import annotations

import anthropic

import config


def _client():
    """Build the LLM client. Same messages.create() surface either way."""
    if config.LLM_PROVIDER == "foundry":
        from anthropic import AnthropicFoundry
        base_url = config.AZURE_FOUNDRY_BASE_URL
        resource = config.AZURE_FOUNDRY_RESOURCE
        # A resource is a bare NAME. If it looks like a URL (common mistake),
        # treat it as the base_url instead.
        if not base_url and resource and (
            "://" in resource or "/" in resource or "." in resource
        ):
            base_url, resource = resource, None

        if base_url:
            url = base_url.strip().strip('"').strip("'").rstrip("/")
            if not url.startswith(("http://", "https://")):
                url = "https://" + url          # tolerate a missing scheme
            if url.endswith("/v1"):             # SDK appends /v1 itself
                url = url[:-3].rstrip("/")
            return AnthropicFoundry(
                api_key=config.AZURE_FOUNDRY_API_KEY,
                base_url=url + "/",
            )
        return AnthropicFoundry(
            api_key=config.AZURE_FOUNDRY_API_KEY,
            resource=resource,
        )
    return anthropic.Anthropic()  # direct Anthropic API

SYSTEM_PROMPT = """You are a research assistant answering questions about NLP \
papers, grounded in a set of numbered context passages.

Rules:
- Answer using ONLY the numbered context passages provided. Do not use outside \
knowledge. If the passages do not contain the answer, say so plainly and stop.
- Cite the passages you rely on inline with bracketed numbers like [1] or [2, 3], \
placed right after the claim they support.

Style — write a clear, well-structured answer:
- Open with a direct 1–2 sentence answer to the question.
- Then explain: synthesize across the relevant passages, compare or contrast the \
approaches they describe, and note any caveats or disagreements between them.
- Use short paragraphs, and a bulleted list when you're enumerating methods or \
findings. Define jargon briefly the first time it appears.
- Be thorough but do not pad: every sentence should carry information from the \
passages. Never invent detail to fill space."""


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        title = hit["metadata"]["title"]
        blocks.append(f"[{i}] (from \"{title}\")\n{hit['text']}")
    return "\n\n".join(blocks)


def generate_with_history(query: str, hits: list[dict],
                          history: list[dict] | None = None) -> str:
    """Answer the query grounded in `hits`, with prior turns for context.

    `history` is a list of {"role", "content"} dicts holding earlier plain-text
    Q/A turns (no context blocks) so follow-ups stay coherent without bloating
    the prompt. Only the CURRENT turn carries the heavy retrieved context.
    """
    # No relevant chunks passed the retrieval gate → don't invoke the LLM at all
    # (nothing to ground on, and it avoids paying for a guaranteed non-answer).
    if not hits:
        return ("I couldn't find anything relevant to that in the indexed arXiv "
                "papers, so I can't answer it from this corpus.")

    client = _client()
    context = _format_context(hits)

    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the passages above, with citations."
    )
    messages = (history or []) + [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return next(b.text for b in response.content if b.type == "text")


def generate(query: str, hits: list[dict]) -> str:
    """One-shot answer (no conversation history)."""
    return generate_with_history(query, hits, history=None)


def format_sources(hits: list[dict]) -> str:
    """Human-readable source list to print under the answer."""
    lines = ["\nSources:"]
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        lines.append(f"  [{i}] {m['title']} ({m['published']})\n      {m['url']}")
    return "\n".join(lines)
