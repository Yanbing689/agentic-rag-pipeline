"""
agent/controller.py — Agentic controller that decides search vs. direct answer.
"""

from config import SEARCH_KEYWORDS


def agent_controller(query: str) -> str:
    """
    Decide whether to retrieve from the vector store or answer directly.

    Returns:
        "search"  — use RAG pipeline
        "direct"  — answer from LLM knowledge only
    """
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in SEARCH_KEYWORDS):
        return "search"
    return "direct"
