# Bank Assistant

This is a small RAG project for answering questions about Bank products.
It reads the bank's Excel knowledge base, turns it into clean documents, builds a local vector database, and lets you use that data for retrieval.

## What it uses

- Python 3.12
- `uv` for environment and package management
- `openpyxl` for reading the Excel file
- `sentence-transformers` for embeddings
- `ChromaDB` for the local vector store
- `pytest` for tests

## Project flow

There are two main scripts:

- [scripts/preprocess.py](scripts/preprocess.py) — reads the Excel file and writes cleaned documents to [data/processed/documents.json](data/processed/documents.json)
- [scripts/build_vectordb.py](scripts/build_vectordb.py) — chunks those documents, creates embeddings, and stores them in [data/vectorstore](data/vectorstore)

## Setup

Make sure you have `uv` installed.

Then from the project root run:

```bash
uv python install 3.12
uv sync --dev --python 3.12
```

This project is set up for CPU use, so you do not need NVIDIA or CUDA.

## Run the project

### 1. Preprocess the bank data

```bash
uv run python scripts/preprocess.py
```

### 2. Build the vector database

```bash
uv run python scripts/build_vectordb.py
```

## Run tests

```bash
uv run pytest -q
```

## Notes

- The source Excel file is [NUST Bank-Product-Knowledge.xlsx](NUST%20Bank-Product-Knowledge.xlsx)
- Processed output is written under [data/processed](data/processed)
- The vector store is saved under [data/vectorstore](data/vectorstore)
