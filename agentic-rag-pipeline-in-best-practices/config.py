"""
config.py — Central configuration for the Agentic RAG Pipeline.
All tunable parameters live here. No hardcoded values in src/.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CHROMA_DIR      = BASE_DIR / "chroma_db"

# ── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
NORMALIZE_EMBEDDINGS = True

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_MODEL       = "google/flan-t5-base"   # swap to any HF model
LLM_MAX_NEW_TOKENS = 300
LLM_DEVICE      = 0 if os.environ.get("USE_GPU") else -1   # -1 = CPU

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50

# ── Retriever ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_store"
RETRIEVER_K     = 3

# ── Agent keywords (extend as needed) ─────────────────────────────────────
SEARCH_KEYWORDS = [
    "pdf", "document", "summary", "summarize", "according to",
    "in the file", "from the report", "what does it say"
]
