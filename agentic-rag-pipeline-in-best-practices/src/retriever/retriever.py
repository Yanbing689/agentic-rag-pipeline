"""
retriever/retriever.py — Chroma vector store setup and retriever factory.
"""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import COLLECTION_NAME, CHROMA_DIR, RETRIEVER_K


def build_vectorstore(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist: bool = True,
) -> Optional[Chroma]:
    """
    Build a Chroma vector store from document chunks.

    Args:
        chunks:          List of LangChain Document chunks.
        embedding_model: Initialized HuggingFaceEmbeddings instance.
        persist:         Whether to persist the DB to disk.

    Returns:
        Chroma instance, or None if chunks is empty.
    """
    if not chunks:
        print("⚠️  No chunks provided — skipping vector store build.")
        return None

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    persist_dir = str(CHROMA_DIR) if persist else None

    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    print(f"✅ Vector store ready — {db._collection.count()} docs indexed")
    return db


def load_vectorstore(embedding_model: HuggingFaceEmbeddings) -> Optional[Chroma]:
    """Load a persisted Chroma vector store from disk."""
    if not CHROMA_DIR.exists():
        print("⚠️  No persisted vector store found. Run scripts/ingest.py first.")
        return None

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"✅ Loaded persisted vector store — {db._collection.count()} docs")
    return db


def get_retriever(db: Chroma) -> Optional[VectorStoreRetriever]:
    """Return a retriever from an existing Chroma DB instance."""
    if db is None:
        return None
    return db.as_retriever(search_kwargs={"k": RETRIEVER_K})
