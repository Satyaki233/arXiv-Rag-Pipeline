"""Ask a question against the indexed corpus (end-to-end RAG query).

    python -m scripts.ask "What methods reduce hallucination in LLMs?"
"""
import sys

from src import generate, retrieve


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.ask "your question"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    print(f"Q: {query}\n")
    hits = retrieve.retrieve(query)
    answer = generate.generate(query, hits)

    print(answer)
    if hits:
        print(generate.format_sources(hits))


if __name__ == "__main__":
    main()
