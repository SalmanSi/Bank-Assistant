# Bank Assistant

A RAG project for answering questions about NUST Bank products.
It reads the bank's Excel knowledge base, turns it into clean documents, builds a local vector database, and answers questions through a Streamlit chat interface powered by a local LLM via Ollama.
All user input and LLM output passes through a multi-layer security pipeline before anything reaches the model.
The system supports real-time knowledge updates — new FAQ documents can be uploaded and become instantly searchable, and outdated information is automatically replaced.

## System Architecture

The project is built on a clean, modular architecture. The data flow starts from the raw Excel knowledge base, moves through preprocessing and vector indexing, and finally reaches the RAG-powered chat interface.
The main components are:
* **Ingestion (`scripts/preprocess.py`)**: Cleans and extracts text from bank spreadsheets.
* **Embeddings & Vectorstore (`scripts/build_vectordb.py` & `scripts/document_manager.py`)**: Chunks documents, embeds using `bge-small-en-v1.5`, and handles DB state.
* **Security Guardrails (`scripts/guardrails.py`)**: Scans all LLM inputs and outputs using ML and custom regex rules.
* **LLM Integration (`scripts/rag_pipeline.py`)**: Interfaces with Ollama locally with contextual querying.
* **User Interface (`app.py`)**: A Streamlit chat UI for live interaction and database management.

![System Architecture](system_architecture.png)

## What it uses

- Python 3.12
- `uv` for environment and package management
- `openpyxl` for reading the Excel file
- `sentence-transformers` (`BAAI/bge-small-en-v1.5`) for embeddings
- `ChromaDB` for the local vector store
- `Ollama` + `qwen3:1.7b` as the local LLM
- `llm-guard` for ML-based input and output scanning
- `Streamlit` for the chat UI
- `pytest` for tests

## Project flow

```
scripts/preprocess.py        reads Excel -> data/processed/documents.json
scripts/build_vectordb.py    chunks + embeds -> data/vectorstore/
scripts/document_manager.py  real-time CRUD for vector DB (add/update/delete)
scripts/guardrails.py        multi-layer security pipeline (regex + ML scanners)
scripts/rag_pipeline.py      retrieves context + calls Ollama
app.py                       Streamlit chat UI with knowledge management sidebar
```

## Security pipeline

Every query passes through three layers before reaching the LLM, and every response passes through two layers before being returned.

**Input layers:**
1. Sanity checks (type, empty, length limit)
2. Regex patterns for jailbreak attempts, off-topic requests, and prompt injection
3. ML scanners via llm-guard: PromptInjection, Toxicity, BanTopics, Gibberish, TokenLimit, InvisibleText

**Output layers:**
1. Regex patterns for sensitive data leakage (system prompt markers, card numbers, credentials)
2. ML scanner via llm-guard: BanTopics (blocks violence, politics, drugs, hacking)

Blocked inputs and outputs return a fixed safe response without ever reaching the LLM.

## Setup

### 1. Install uv and Python

```bash
uv python install 3.12
uv sync --dev --all-extras --python 3.12
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

## Real-Time Knowledge Management

The app sidebar (`Knowledge Base`) provides a robust, state-persistent tab structure for managing the database dynamically:

- **Overview** — High-level dashboard displaying total chunks, active sources, and disk storage metrics.
- **Upload** — Drops in JSON files for processing. It intelligently auto-detects standard document arrays vs categorized FAQ structures. It leverages `upsert` to deliberately overwrite overlapping collision IDs.
- **Add** — Quickly inject single manual Q&A pairs directly into the store.
- **Browse** — Interactive expanding categorical library of ingested sources with granular chunk deletion.
- **Remove** — Permanently purges entire origin sources from the database.

All changes take effect immediately — no rebuild, app crash, or system restart needed. Each uploaded file acts as a source; re-uploading automatically replaces all chunks connected to that filename.

### Allowed Upload Formats

You can upload knowledge in two JSON formats using the sidebar:

**1. Categorized FAQ Format** (Identified by a `categories` root object)
```json
{
  "categories": [
    {
      "category": "Funds Transfer",
      "questions": [
        {
          "question": "How do I add a beneficiary?",
          "answer": "Go to settings > beneficiaries."
        }
      ]
    }
  ]
}
```

**2. Standard Document Array Format** (Identified by an array of objects)
```json
[
  {
    "id": "DOC_001",
    "content": "NUST Bank branches will be closed on Friday.",
    "product": "General", 
    "category": "Operation Hours"
  }
]
```

## Conversational Memory & Context Compression

To handle "infinite context" without breaking token limits, the application features an active continuous summarization engine.

- **Query Rewriting:** Follow-up questions (like "tell me more" or "how do I open it?") pass through an initial LLM phase parameterized by your chat history to automatically transform them into robust, standalone search queries before hitting the vector store.
- **Active Summarization:** Instead of silently dropping older messages, the backend watches your active token footprint. When limits are exceeded, older interactions are dynamically merged into a persistent conversation summary. This ensures no contextual loss.

### Memory Configuration
Memory retention limits are fully configurable directly in an `.env` file (loaded automatically via `python-dotenv`):
- `MAX_HISTORY_MESSAGES`: Defines how many conversational turns the model holds raw before compressing it. (Default: 20)
- `MAX_HISTORY_TOKENS`: A fail-safe limit; triggers summarization if active raw messages exceed this token length estimation. (Default: 20000)

## Usage Example

Below is an example of the chatbot in action, answering a user question about NUST Bank products:

![Chatbot Interface](interface.jpeg)

## Run tests

All unit tests (no Ollama needed):
```bash
uv run pytest tests/ -v
```

Document manager tests only:
```bash
uv run pytest tests/test_document_manager.py -v
```

Full test suite including real ML models and live Ollama (requires Ollama running):
```bash
uv run pytest tests/ -v --run-e2e
```

## Retrieval Evaluation

The project includes a comprehensive evaluation framework for the retrieval step. Run the evaluation script:

```bash
PYTHONPATH=. uv run python eval/evaluate_retrieval.py
```

### Evaluation Results

| Metric | Value |
|--------|-------|
| **Hit Rate@1** | 37.93% |
| **Hit Rate@3** | 44.83% |
| **Hit Rate@5** | 48.28% |
| **Hit Rate@10** | 58.62% |
| **MRR** (Mean Reciprocal Rank) | 0.4355 |
| **MAP** (Mean Average Precision) | 0.3357 |
| **NDCG@5** | 0.4210 |
| **Precision@5** | 0.1724 |
| **Recall@5** | 0.3417 |

### Distance Analysis
- Relevant docs: Mean distance = 0.2175
- Irrelevant docs: Mean distance = 0.3196

The retrieval system shows good performance on specific factual queries (rates, remittance limits, eligibility criteria) but struggles with queries requiring cross-referencing multiple product categories or app-specific features not well-represented in the Excel knowledge base.

Detailed results are saved to `data/eval_results.json`.

## References

- **Dataset**: `NUST Bank-Product-Knowledge.xlsx`
- **FAQ Dataset Appendix**: `funds_transfer_app_features_faq (1).json`
- **UI Framework Library**: Streamlit (https://streamlit.io/) 
- **Embeddings**: BAAI/bge-small-en-v1.5 model via `sentence-transformers`
- **Vector Database**: ChromaDB (https://www.trychroma.com/)
- **LLM Engine**: Ollama with model `qwen3:1.7b` parameter
- **Security Checkers**: llm-guard library (https://github.com/protectai/llm-guard)