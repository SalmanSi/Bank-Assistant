# Bank Assistant RAG - Implementation Plan

## Project Overview

A RAG (Retrieval-Augmented Generation) chat application built on NUST Bank's product knowledge base.
The system answers user banking queries by retrieving relevant product information and generating responses.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python |
| Package Manager | `uv` |
| Excel Parsing | `openpyxl` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) via `sentence-transformers` |
| Vector DB | `chromadb` (in-process, persisted to disk) |
| RAG Pipeline | LangChain (Script 3, future) |
| Testing | `pytest` + `pytest-cov` |

---

## Dataset Analysis

- **File:** `NUST Bank-Product-Knowledge.xlsx`
- **36 sheets total:**
  - `Main` — Navigation index, lists all products grouped by category
  - `Rate Sheet July 1 2024` — Tabular profit rates for savings accounts and term deposits
  - `Sheet1` — Empty, skipped
  - **33 product sheets** — Q&A style content per product (e.g., `NAA`, `LCA`, `NWA`, `NUST4Car`, etc.)
- **Data patterns:**
  - Product sheets: questions in one row, answers in following rows, heavy merged cells (up to 62 per sheet)
  - Rate sheet: multi-section table with account names, tenors, payout frequencies, rates
  - Some cells contain Excel formula references (e.g., `='Rate Sheet July 1 2024'!D31`)
- **PII findings:** Only public-facing bank URLs and official customer service emails found. No customer CNICs, personal phone numbers, or account numbers. No anonymization required but scan is logged explicitly.

### Product Category Mapping (from `Main` sheet)

| Category | Products |
|----------|----------|
| Liability Products & Services | NAA, LCA, NWA, PWRA, RDA, VPCA, VP-BA, VPBA, NSDA, PLS, CDA, NMA, NADA, NADRA, NSA |
| Consumer Products | NUST4Car, PF, NMC, NMF, NSF, NIF, NUF, NFMF, NFBF, PMYB&ALS, NRF, NHF |
| Insurance / Bancassurance | Nust Life, EFU Life, Jubilee Life |
| Remittance | HOME REMITTANCE |
| Rate Info | Rate Sheet July 1 2024 |

---

## Environment Setup

### Python Version
Pinned to Python 3.12 for compatibility.

### `pyproject.toml`

```toml
[project]
name = "bank-assistant"
version = "0.1.0"
description = "RAG-based Bank Assistant using NUST Bank product knowledge"
requires-python = ">=3.11,<3.13"
dependencies = [
    "openpyxl>=3.1.0",
    "chromadb>=1.0.0",
    "sentence-transformers>=3.0.0",
    "langchain>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-huggingface>=0.1.0",
    "langchain-chroma>=0.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

> **Note on `sentence-transformers`:** This package pulls in PyTorch which can drag in CUDA bindings.
> To avoid this, add CPU-only PyTorch configuration in `[tool.uv]`:
> ```toml
> [tool.uv]
> extra-index-url = ["https://download.pytorch.org/whl/cpu"]
> override-dependencies = [
>     "torch>=2.0.0",
>     "nvidia-* ; python_version < '0'",  # exclude all NVIDIA packages
>     "triton ; python_version < '0'",
> ]
> ```

### Commands

```bash
uv python install 3.12                          # Download Python 3.12 via uv
uv sync --dev --python 3.12                     # Create .venv and install all deps
mkdir -p data/processed data/vectorstore tests  # Create project directories
```

---

## File Structure

```
Bank-Assistant/
├── AGENTS.md
├── opencode.json
├── pyproject.toml
├── PLAN.md                          # This file
├── NUST Bank-Product-Knowledge.xlsx
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py                # Script 1: Data extraction & cleaning
│   └── build_vectordb.py            # Script 2: Chunking, embedding & vector DB
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   └── test_build_vectordb.py
└── data/
    ├── processed/
    │   └── documents.json           # Output of Script 1
    └── vectorstore/                 # Output of Script 2 (ChromaDB persisted collection)
```

---

## Script 1: `scripts/preprocess.py`

### Goal
Extract all content from the Excel file into clean, structured documents per product, handling merged cells, Q&A patterns, and tabular data.

### Steps

#### 1. Load workbook
```python
wb = openpyxl.load_workbook("NUST Bank-Product-Knowledge.xlsx", data_only=True)
```
`data_only=True` resolves Excel formula references (e.g., `='Rate Sheet July 1 2024'!D31` → actual value).

#### 2. PII scan (explicit for rubric compliance)
Scan every cell with regex patterns for:
- CNIC: `\d{5}-\d{7}-\d{1}`
- Personal phone: `\d{4}-\d{7}`
- Personal email: any email not matching `@NUSTbank.com*`

Log findings. Expected result: no customer PII found. Only public bank contacts present.

#### 3. Parse `Main` sheet → product-to-category mapping
Walk rows of `Main` sheet to extract which column/section each product belongs to:
```python
category_map = {
    "NAA": "liability",
    "NUST4Car": "consumer",
    "Nust Life": "insurance",
    "HOME REMITTANCE": "remittance",
    ...
}
```

#### 4. Parse `Rate Sheet July 1 2024` (tabular data)
- Walk rows, detect account name headers (rows with only 1-2 non-null cells in known columns)
- Collect key-value rows (Profit Payment, Profit Rate, Tenor, Payout)
- Convert to natural language sentences per account type:
  ```
  NUST Asaan Account Savings: Profit payment Semi-Annually at 19.00% per annum.
  Term Deposit - Short Notice Deposit Receipt (SNDR): 7-day tenor, paid at Maturity, rate 16.50%.
  ```
- Output: list of `rate_info` documents

#### 5. Parse each product sheet (33 sheets)

**5a. Unmerge cells — propagate values**
```python
# Before iterating rows, unmerge and fill merged cell values
for merged_range in ws.merged_cells.ranges:
    top_left_value = ws.cell(merged_range.min_row, merged_range.min_col).value
    ws.unmerge_cells(str(merged_range))
    for row in range(merged_range.min_row, merged_range.max_row + 1):
        for col in range(merged_range.min_col, merged_range.max_col + 1):
            ws.cell(row, col).value = top_left_value
```

**5b. Q&A pair detection**
A row is a **question** if:
- It contains `?` character, OR
- It starts with a numbered pattern like `1.`, `2.`, `Q:`

Rows following a question (until the next question or empty section) form the **answer**.

Build Q&A pairs:
```python
{"question": "What is the Eligibility Criteria for NAA?",
 "answer": "Resident Pakistani individuals who do not maintain any other account..."}
```

**5c. Inline table detection**
If consecutive rows have data in 3+ columns following a structured pattern (column headers in one row, values in next rows), convert to readable prose:
```
Savings Account: Minimum Age 55 years, Profit Calculation on Monthly Average Balance, Profit Payment Monthly, Currency PKR.
Term Deposit: Tenure 1 year, Minimum Age 55 years, Profit Payment Monthly, Currency PKR.
```

**5d. Text cleaning**
- Replace `\xa0` (non-breaking space) with regular space
- Normalize bullet characters (`·`, `•`, `.`) → `- `
- Strip `\t`, excessive `\n`, leading/trailing whitespace
- Remove navigation labels: cells containing only `"Main"` are skipped
- Remove URLs from content (already logged in PII scan)
- Normalize phone/email fields: replace `CustomerServices@NUSTbank.com.pk` with `[BANK_CONTACT_EMAIL]`

#### 6. Output format
Save to `data/processed/documents.json` as a JSON array:

```json
[
  {
    "id": "NAA_001",
    "product": "NUST Asaan Account (NAA)",
    "sheet": "NAA",
    "type": "qa_pair",
    "category": "liability",
    "question": "What is the Eligibility Criteria for NAA?",
    "content": "Q: What is the Eligibility Criteria for NAA?\nA: Resident Pakistani individuals who do not maintain/have any other account (single or joint) in NUST Bank Limited are eligible to open the NAA in Pak rupees as a single/joint account."
  },
  {
    "id": "RATE_001",
    "product": "Rate Sheet",
    "sheet": "Rate Sheet July 1 2024",
    "type": "rate_info",
    "category": "rate",
    "question": null,
    "content": "NUST Asaan Account Savings: Profit payment Semi-Annually at 19.00% per annum."
  }
]
```

---

## Script 2: `scripts/build_vectordb.py`

### Goal
Chunk preprocessed documents, embed with `BAAI/bge-small-en-v1.5`, build a ChromaDB collection on disk, and expose a `load_vectorstore()` helper for Script 3.

### Steps

#### 1. Load documents
```python
with open("data/processed/documents.json") as f:
    documents = json.load(f)
```

#### 2. Chunking with LangChain
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

- **Q&A pairs** (`type == "qa_pair"`): Most are <512 chars and stay as a single chunk. For longer ones, the splitter splits and the question is prepended to each sub-chunk to preserve retrieval context.
- **Rate info** (`type == "rate_info"`): Each block is typically short, stays as one chunk.
- **Each chunk carries forward all metadata** from the parent document.

#### 3. Embedding model
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
```

- Dimension: **384**
- At **query time**, prepend `"Represent this sentence for searching relevant passages: "` to the query (BGE instruction prefix for retrieval). Document chunks are embedded as-is.

#### 4. ChromaDB collection setup

```python
import chromadb

chroma_client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
collection = chroma_client.get_or_create_collection(
    name="bank_knowledge",
    metadata={"hnsw:space": "cosine"}  # cosine similarity for BGE embeddings
)
```

ChromaDB handles HNSW indexing automatically. Metadata fields (`product`, `category`, `sheet`, `chunk_type`, `question`) are stored alongside documents and support filtering at query time.

#### 5. Build and persist

```python
VECTORSTORE_PATH = "data/vectorstore"

chroma_client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
collection = chroma_client.get_or_create_collection(
    name="bank_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Only build if collection is empty (skip rebuild)
if collection.count() == 0:
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i:05d}")
        embeddings.append(model.encode(chunk["content"]).tolist())
        documents.append(chunk["content"])
        metadatas.append({
            "product":    chunk["product"],
            "sheet":      chunk["sheet"],
            "chunk_type": chunk["type"],
            "question":   chunk.get("question", ""),
            "category":   chunk["category"],
        })
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Built vectorstore: {collection.count()} chunks")
else:
    print("Vectorstore already exists, skipping rebuild.")
```

#### 6. Load helper (used by Script 3)

```python
def load_vectorstore() -> chromadb.Collection:
    """Load the persisted ChromaDB collection."""
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    return client.get_collection(name="bank_knowledge")
```

#### 7. Query interface

**Unfiltered (general queries):**
```python
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
```

**Metadata-filtered (when product/category can be inferred from query):**
```python
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"category": "consumer"},
    include=["documents", "metadatas", "distances"]
)
```

#### 8. Verification (printed on every run)
- Total chunks inserted
- Collection `count()` and disk size
- 3 sample queries with top-3 results printed:
  1. `"account for senior citizens"` (should retrieve NWA)
  2. `"auto finance markup rate"` (should retrieve NUST4Car + Rate Sheet)
  3. `"profit rate savings account"` (should retrieve Rate Sheet entries)

---

## Metadata Fields Summary

| Field | Type | Filterable | Purpose |
|-------|------|-----------|--------|
| `content` | STRING (document) | — | Full chunk text passed to LLM |
| `product` | STRING (metadata) | Yes | Filter by specific product |
| `sheet` | STRING (metadata) | Yes | Source traceability |
| `chunk_type` | STRING (metadata) | Yes | `qa_pair`, `rate_info`, `general_info` |
| `question` | STRING (metadata) | Yes | Original question for Q&A chunks |
| `category` | STRING (metadata) | Yes | Filter by product category |

---

## Test Plan

### `tests/test_preprocess.py`

| Test | What it checks |
|------|---------------|
| `test_pii_scan_finds_no_customer_pii` | No CNIC or personal phone numbers in raw data |
| `test_category_map_built` | `Main` sheet parsed, all 33 product sheets mapped to a category |
| `test_qa_pairs_extracted` | At least 1 Q&A pair per product sheet; each has non-empty question and content |
| `test_rate_sheet_parsed` | Rate sheet produces `rate_info` docs with numeric rates and account names |
| `test_text_cleaning_removes_nbsp` | `\xa0` not present in any output content |
| `test_text_cleaning_removes_navigation` | `"Main"` not present as standalone content in any document |
| `test_output_schema_valid` | Every document has required keys: `id`, `product`, `sheet`, `type`, `category`, `content` |
| `test_output_ids_unique` | All document `id` values are unique |
| `test_no_empty_content` | No document has an empty or whitespace-only `content` field |
| `test_documents_json_written` | `data/processed/documents.json` exists and is valid JSON after run |

### `tests/test_build_vectordb.py`

| Test | What it checks |
|------|---------------|
| `test_chunks_created` | At least as many chunks as input documents (splitting works) |
| `test_chunk_size_within_limit` | No chunk content exceeds 600 chars (splitter respected, with small buffer) |
| `test_chunk_metadata_complete` | Every chunk has all required metadata keys |
| `test_embedding_dimension` | Embedding model produces 384-dim vectors |
| `test_vectorstore_built` | `data/vectorstore/` directory exists after build |
| `test_vectorstore_doc_count` | `collection.count()` matches number of chunks inserted |
| `test_load_vectorstore` | `load_vectorstore()` opens collection without error |
| `test_unfiltered_query_returns_results` | Query `"senior citizen account"` returns >= 1 result |
| `test_filtered_query_by_category` | Query with `where={"category": "liability"}` only returns liability docs |
| `test_filtered_query_by_product` | Query with `where={"product": "NUST Waqaar Account"}` returns relevant docs |
| `test_query_result_fields_present` | Each result has `content`, `product`, `category` fields |
| `test_vectorstore_persistence` | After closing and reopening, `count()` is the same |
| `test_rebuild_skipped_if_exists` | Running `scripts/build_vectordb.py` a second time does not overwrite existing store |

---

## Rubric Alignment

### Data Preprocessing (2 marks)
- **Anonymization check:** Explicit PII scan logged in `scripts/preprocess.py` — confirms no customer data present
- **Clean, reusable pipeline:** Script is idempotent; outputs structured JSON reusable by any downstream consumer
- **Handling banking data carefully:** Merged cell resolution, formula dereferencing (`data_only=True`), proper handling of multi-row answers

### Vector Embeddings & Retrieval (3 marks)
- **Highly relevant retrieval:** BGE instruction prefix at query time maximizes relevance for retrieval tasks
- **Well-tuned search:** ChromaDB HNSW with cosine similarity; metadata filtering on `product` and `category`
- **Diverse query support:** Both unfiltered semantic search and metadata-filtered search supported
- **Chunk quality:** Q&A pairs kept together as single chunks to preserve question-answer coherence; question prepended to sub-chunks when splitting is needed
