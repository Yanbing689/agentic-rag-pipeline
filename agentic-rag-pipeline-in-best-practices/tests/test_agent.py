"""
tests/test_agent.py — Unit tests for the agent controller.
"""

import sys
sys.path.insert(0, "src")

from agent.controller import agent_controller


def test_search_on_pdf_keyword():
    assert agent_controller("Give me a summary from the PDF") == "search"

def test_search_on_document_keyword():
    assert agent_controller("What does the document say about pricing?") == "search"

def test_direct_on_general_question():
    assert agent_controller("What is machine learning?") == "direct"

def test_direct_on_greeting():
    assert agent_controller("Hello, how are you?") == "direct"

def test_search_case_insensitive():
    assert agent_controller("SUMMARIZE the PDF please") == "search"
