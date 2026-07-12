"""Interactive terminal chat over the arXiv corpus (RAG, one retrieval per turn).

Run:
    .venv/bin/python chat-app.py

Type a question and press Enter. Each turn retrieves fresh passages and answers
grounded in them, with citations. The conversation is remembered, so follow-ups
like "tell me more about that" work.

Commands:
    /sources   toggle the source list under each answer
    /reset     clear the conversation history
    /exit      quit  (Ctrl-D or Ctrl-C also quit)
"""
import os

# Set before the HF tokenizers library is imported (via retrieve) so it doesn't
# print the parallelism-after-fork warning. Also silences it for the whole run.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src import generate, retrieve  # noqa: E402  (import after env var is set)

BANNER = """\
arXiv RAG chat  —  ask about recent NLP (cs.CL) papers
commands: /sources  /reset  /exit
(first question takes a few seconds while the models load)
"""


def main() -> None:
    history: list[dict] = []
    show_sources = True
    print(BANNER)

    while True:
        try:
            query = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not query:
            continue
        if query in ("/exit", "/quit", "exit", "quit"):
            print("bye")
            break
        if query == "/reset":
            history.clear()
            print("(conversation cleared)")
            continue
        if query == "/sources":
            show_sources = not show_sources
            print(f"(sources {'on' if show_sources else 'off'})")
            continue

        try:
            hits = retrieve.retrieve(query)
            answer = generate.generate_with_history(query, hits, history)
        except Exception as e:  # keep the loop alive on transient API/network errors
            print(f"\n[error] {type(e).__name__}: {e}")
            continue

        print(f"\nclaude > {answer}")
        if show_sources and hits:
            print(generate.format_sources(hits))

        # Store plain-text turns (no context blocks) so follow-ups have memory
        # without the prompt growing by 5 chunks every turn.
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
