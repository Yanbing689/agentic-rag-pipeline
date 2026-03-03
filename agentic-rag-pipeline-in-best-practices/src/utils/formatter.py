"""
utils/formatter.py — Response formatting utilities.
"""

import re
from typing import Optional


def format_summary(response: str, query: Optional[str] = None) -> str:
    """
    Clean and format a numbered-summary response from the LLM.
    - Strips echoed prompt
    - Splits on numbered points
    - Rejoins with blank line separation
    """
    # Strip echoed prompt fragments
    for marker in ["Answer:", "Use this context:", "Task:"]:
        if marker in response:
            response = response.split(marker)[-1]

    if query and query in response:
        response = response.split(query)[-1]

    response = response.strip()

    # Split on patterns like "1." "2." "1)" "2)"
    points = re.split(r"\n?\s*\d+[\.\)]\s+", response)
    points = [p.strip() for p in points if p.strip()][:5]

    if not points:
        return response  # return as-is if no numbered points found

    formatted = "\n\n".join(f"{i + 1}. {point}" for i, point in enumerate(points))
    return formatted


def format_direct(response: str, query: Optional[str] = None) -> str:
    """Clean up a direct LLM response."""
    if query and query in response:
        response = response.split(query)[-1]
    return response.strip()


def print_result(title: str, content: str) -> None:
    """Pretty-print a result block to the console."""
    border = "=" * 55
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    print(content)
    print(border)
