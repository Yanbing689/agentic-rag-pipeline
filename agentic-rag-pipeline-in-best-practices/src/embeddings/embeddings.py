"""
embeddings/embeddings.py — HuggingFace embedding model loader.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, NORMALIZE_EMBEDDINGS


def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load and return the HuggingFace embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
    )
    # Sanity check
    test_vec = model.embed_query("test")
    assert len(test_vec) > 0, "Embedding model returned empty vector!"
    print(f"✅ Embedding model ready — vector dim: {len(test_vec)}")
    return model
