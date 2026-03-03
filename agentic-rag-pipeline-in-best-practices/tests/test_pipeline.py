"""
tests/test_pipeline.py — Unit tests for formatter utilities.
"""

import sys
sys.path.insert(0, "src")

from utils.formatter import format_summary, format_direct


def test_format_summary_numbered():
    raw = "Answer: 1. First point 2. Second point 3. Third point"
    result = format_summary(raw)
    assert "1." in result
    assert "2." in result

def test_format_summary_strips_prompt():
    raw = "Use this context: blah blah Answer: 1. Key insight"
    result = format_summary(raw)
    assert "Use this context" not in result

def test_format_direct_strips_query():
    raw = "What is AI? AI stands for Artificial Intelligence."
    result = format_direct(raw, query="What is AI?")
    assert result.startswith("AI stands for")

def test_format_summary_max_5_points():
    raw = "1. A 2. B 3. C 4. D 5. E 6. F 7. G"
    result = format_summary(raw)
    lines = [l for l in result.split("\n") if l.strip()]
    assert len(lines) <= 5
