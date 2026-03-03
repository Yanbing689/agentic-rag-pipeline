[README.md](https://github.com/user-attachments/files/25727358/README.md)
# 🤖 Agentic RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) pipeline with an agentic controller that decides whether to search documents or answer directly.

## 📁 Project Structure

```
03-agentic-rag-pipeline/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── controller.py        # Agent logic: search vs. direct answer
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── retriever.py         # Chroma vector DB + retriever setup
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embeddings.py        # HuggingFace embedding model
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm.py               # LLM pipeline (HuggingFace transformers)
│   └── utils/
│       ├── __init__.py
│       ├── loader.py             # PDF loading + chunking
│       └── formatter.py          # Response formatting
├── data/                         # Put your PDFs here (gitignored)
├── chroma_db/                    # Persisted vector DB (gitignored)
├── tests/
│   ├── test_retriever.py
│   ├── test_agent.py
│   └── test_pipeline.py
├── notebooks/
│   └── exploration.ipynb         # Jupyter experimentation
├── scripts/
│   └── ingest.py                 # One-shot ingestion script
├── main.py                       # App entrypoint
├── config.py                     # All config/constants in one place
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🚀 Quickstart

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/03-agentic-rag-pipeline.git
cd 03-agentic-rag-pipeline
pip install -r requirements.txt
```

### 2. Add your PDFs
```bash
cp your_documents.pdf data/
```

### 3. Ingest documents into vector DB
```bash
python scripts/ingest.py
```

### 4. Run the pipeline
```bash
python main.py
```

## ⚙️ Configuration

Edit `config.py` to change:
- PDF data folder path
- Embedding model name
- LLM model name
- Chunk size / overlap
- Retriever top-k

## 🧪 Run Tests
```bash
pytest tests/ -v
```

## 🏗️ Architecture

```
User Query
    │
    ▼
Agent Controller
    │
    ├── "search"  ──► Retriever ──► Chroma DB ──► Context ──► LLM ──► Response
    │
    └── "direct"  ──────────────────────────────► LLM ──► Response
```
