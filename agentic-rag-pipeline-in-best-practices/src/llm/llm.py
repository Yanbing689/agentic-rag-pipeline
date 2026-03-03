"""
llm/llm.py — LLM pipeline loader.
"""

from transformers import pipeline
from config import LLM_MODEL, LLM_MAX_NEW_TOKENS, LLM_DEVICE


def load_llm():
    """Load and return a HuggingFace text-generation pipeline."""
    print(f"Loading LLM: {LLM_MODEL}")
    llm = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        device=LLM_DEVICE,
    )
    print(f"✅ LLM ready: {LLM_MODEL}")
    return llm


def generate(llm, prompt: str) -> str:
    """Run inference and return the generated text string."""
    result = llm(prompt)
    return result[0]["generated_text"]
