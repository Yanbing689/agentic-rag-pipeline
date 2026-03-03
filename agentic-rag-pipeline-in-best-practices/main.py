"""
main.py — Entrypoint for the Agentic RAG Pipeline.

Usage:
    python main.py
"""

import sys
sys.path.insert(0, "src")

from agent.controller import agent_controller
from embeddings.embeddings import load_embedding_model
from retriever.retriever import load_vectorstore, get_retriever
from llm.llm import load_llm, generate
from utils.formatter import format_summary, format_direct, print_result


def build_prompt(query: str, context: str = "") -> str:
    if context:
        return (
            f"You are a helpful assistant. Use the context below to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Task: {query}\n"
            f"Give exactly 5 numbered points. Be concise.\n\n"
            f"Answer:"
        )
    return query


def rag_answer(query: str, retriever, llm) -> str:
    action = agent_controller(query)

    if action == "search":
        if retriever is None:
            print("⚠️  No retriever available — falling back to direct answer.")
            prompt = build_prompt(query)
            raw = generate(llm, prompt)
            return format_direct(raw, query)

        print(f"🕵️  Agent → SEARCH: '{query}'")
        results = retriever.invoke(query)
        context = "\n".join([r.page_content for r in results])
        prompt = build_prompt(query, context)
        raw = generate(llm, prompt)
        return format_summary(raw, query)

    else:
        print(f"🤖 Agent → DIRECT: '{query}'")
        raw = generate(llm, query)
        return format_direct(raw, query)


def main():
    # ── Load models ──────────────────────────────────────────────────────
    embedding_model = load_embedding_model()
    llm = load_llm()

    # ── Load persisted vector store ──────────────────────────────────────
    db = load_vectorstore(embedding_model)
    retriever = get_retriever(db)

    # ── Run queries ───────────────────────────────────────────────────────
    queries = [
        "Give me a 5-point summary from the PDF",
        "What is an Ideal Resume Format? Explain in 50 words.",
    ]

    for query in queries:
        result = rag_answer(query, retriever, llm)
        print_result(query, result)


if __name__ == "__main__":
    main()
