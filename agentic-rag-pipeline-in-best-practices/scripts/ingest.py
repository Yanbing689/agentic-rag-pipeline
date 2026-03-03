"""
scripts/ingest.py — One-shot script to load PDFs, chunk, embed, and persist to Chroma.

Usage:
    python scripts/ingest.py
"""

import sys
sys.path.insert(0, "src")

from embeddings.embeddings import load_embedding_model
from retriever.retriever import build_vectorstore
from utils.loader import load_pdfs, chunk_documents


def main():
    print("=" * 55)
    print("  Agentic RAG — Document Ingestion")
    print("=" * 55)

    # Step 1: Load PDFs
    docs = load_pdfs()
    if not docs:
        print("\n❌ No documents to ingest. Add PDFs to the data/ folder.")
        sys.exit(1)

    # Step 2: Chunk
    chunks = chunk_documents(docs)

    # Step 3: Embed + store
    embedding_model = load_embedding_model()
    build_vectorstore(chunks, embedding_model, persist=True)

    print("\n✅ Ingestion complete. Run `python main.py` to query.")


if __name__ == "__main__":
    main()
