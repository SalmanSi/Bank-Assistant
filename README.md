# Bank Assistant

A RAG project for answering questions about NUST Bank products.
It reads the bank's Excel knowledge base, turns it into clean documents, builds a local vector database, and answers questions through a Streamlit chat interface powered by a local LLM via Ollama.

## What it uses

- Python 3.12
- `uv` for environment and package management
- `openpyxl` for reading the Excel file
- `sentence-transformers` (`BAAI/bge-small-en-v1.5`) for embeddings
- `ChromaDB` for the local vector store
- `Ollama` + `qwen3:1.7b` as the local LLM
- `Streamlit` for the chat UI
- `pytest` for tests

## Project flow

```
scripts/preprocess.py        reads Excel → data/processed/documents.json
scripts/build_vectordb.py    chunks + embeds → data/vectorstore/
scripts/rag_pipeline.py      retrieves context + calls Ollama
app.py                       Streamlit chat UI
```

## Setup

### 1. Install uv and Python

Make sure you have `uv` installed, then:

```bash
uv python install 3.12
uv sync --dev --python 3.12
```

No NVIDIA or CUDA required — this project runs entirely on CPU.

### 2. Install Ollama

**Linux (requires sudo):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then pull the model:
```bash
ollama pull qwen3:1.7b
```

## Run the project

### 1. Preprocess the bank data

```bash
uv run python scripts/preprocess.py
```

### 2. Build the vector database

```bash
uv run python scripts/build_vectordb.py
```

The embedding model (`BAAI/bge-small-en-v1.5`) is downloaded from HuggingFace on the first run and saved to `data/models/` — subsequent runs load from disk.

### 3. Start Ollama

```bash
ollama serve
```

Keep this running in a separate terminal.

### 4. Launch the chat app

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser and ask questions about NUST Bank products.

## Run tests

```bash
uv run pytest -q
```

## Notes

- Source data: [NUST Bank-Product-Knowledge.xlsx](NUST%20Bank-Product-Knowledge.xlsx)
- Processed documents: [data/processed/](data/processed)
- Vector store: [data/vectorstore/](data/vectorstore)
- Cached embedding model: `data/models/` (excluded from git)
- Terminal logs show the query, retrieved chunks, and context on every request
