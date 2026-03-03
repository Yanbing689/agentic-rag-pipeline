"""
utils/loader.py — PDF loading and text chunking utilities.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdfs(folder_path: Path = DATA_DIR) -> List[Document]:
    """Load all PDFs from a folder and return as LangChain Documents."""
    docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️  No PDF files found in: {folder_path}")
        return []

    for file in pdf_files:
        path = os.path.join(folder_path, file)
        print(f"  Loading: {file}")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    print(f"✅ Loaded {len(docs)} pages from {len(pdf_files)} PDF(s)")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding."""
    if not docs:
        print("⚠️  No documents to chunk.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
