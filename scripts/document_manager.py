"""Document manager for real-time vector DB CRUD operations.

Implements source-level full replacement: when a file is re-uploaded,
all existing chunks tagged with that source are deleted before adding
the new data.  Every managed chunk carries ``source`` and ``ingested_at``
metadata so that the system can track provenance and resolve conflicts.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection

from scripts.build_vectordb import (
    VECTORSTORE_PATH,
    _create_client,
    _collection_disk_size,
    chunk_documents,
    encode_texts,
    get_embedding_model,
    load_vectorstore,
    COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_source_tag(filename: str) -> str:
    """Derive a clean source tag from a filename."""
    stem = Path(filename).stem
    # Remove parenthetical suffixes like " (1)"
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    # Replace spaces/special chars with underscores
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return f"faq::{stem}"


# ---------------------------------------------------------------------------
# FAQ JSON parser
# ---------------------------------------------------------------------------


def parse_faq_json(path: Path | str) -> list[dict[str, Any]]:
    """Parse a FAQ-format JSON file into standard document dicts.

    Expected format::

        {
          "categories": [
            {
              "category": "Category Name",
              "questions": [
                {"question": "...", "answer": "..."},
                ...
              ]
            }
          ]
        }

    Returns a list of document dicts compatible with ``chunk_documents``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = []
    counter = 0

    for cat_block in raw.get("categories", []):
        category = cat_block.get("category", "General")
        for qa in cat_block.get("questions", []):
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if not question or not answer:
                continue
            counter += 1
            doc_id = f"FAQ_{counter:04d}"
            documents.append(
                {
                    "id": doc_id,
                    "product": category,
                    "sheet": "faq_upload",
                    "type": "qa_pair",
                    "category": "faq",
                    "question": question,
                    "content": f"Q: {question}\nA: {answer}",
                }
            )
    return documents


def parse_documents_json(path: Path | str) -> list[dict[str, Any]]:
    """Parse a documents.json-format file (array of document dicts)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array of document objects")
    required_keys = {"id", "content"}
    for doc in raw:
        if not required_keys.issubset(doc.keys()):
            raise ValueError(
                f"Document missing required keys {required_keys - doc.keys()}: {doc.get('id', '?')}"
            )
    return raw


def detect_and_parse(path: Path | str) -> list[dict[str, Any]]:
    """Auto-detect file format (FAQ JSON vs documents JSON) and parse."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "categories" in raw:
        return parse_faq_json(path)
    if isinstance(raw, list):
        return parse_documents_json(path)
    raise ValueError(
        "Unrecognized JSON format. Expected either a FAQ JSON "
        '(object with "categories" key) or a documents JSON (array).'
    )


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def add_documents(
    documents: list[dict[str, Any]],
    source: str,
    collection: Collection,
    model: Any,
) -> dict[str, Any]:
    """Delete all existing chunks for *source*, then chunk, embed, and add *documents*.

    Returns a summary dict with keys ``added``, ``deleted``, ``source``.
    """
    # 1. Delete all existing chunks for this source
    deleted = _delete_chunks_by_metadata(collection, "source", source)

    # 2. Chunk
    chunks = chunk_documents(documents)
    if not chunks:
        return {"added": 0, "deleted": deleted, "source": source}

    # 3. Embed
    embeddings = encode_texts(model, [c["content"] for c in chunks])
    now = _now_iso()

    # 4. Add with source + timestamp
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "parent_id": c["parent_id"],
                "product": c["product"],
                "sheet": c["sheet"],
                "chunk_type": c["type"],
                "question": c.get("question") or "",
                "category": c["category"],
                "source": source,
                "ingested_at": now,
            }
            for c in chunks
        ],
    )
    logger.info(
        "add_documents: source=%r  deleted=%d  added=%d",
        source,
        deleted,
        len(chunks),
    )
    return {"added": len(chunks), "deleted": deleted, "source": source}


def add_single_document(
    doc: dict[str, Any],
    source: str,
    collection: Collection,
    model: Any,
) -> dict[str, Any]:
    """Add a single document.  Deletes any existing chunks with the same ``parent_id`` first."""
    parent_id = doc["id"]
    deleted = _delete_chunks_by_metadata(collection, "parent_id", parent_id)

    chunks = chunk_documents([doc])
    if not chunks:
        return {"added": 0, "deleted": deleted, "parent_id": parent_id}

    embeddings = encode_texts(model, [c["content"] for c in chunks])
    now = _now_iso()

    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "parent_id": c["parent_id"],
                "product": c["product"],
                "sheet": c["sheet"],
                "chunk_type": c["type"],
                "question": c.get("question") or "",
                "category": c["category"],
                "source": source,
                "ingested_at": now,
            }
            for c in chunks
        ],
    )
    return {"added": len(chunks), "deleted": deleted, "parent_id": parent_id}


def delete_by_source(source: str, collection: Collection) -> int:
    """Remove all chunks belonging to *source*.  Returns count of deleted chunks."""
    return _delete_chunks_by_metadata(collection, "source", source)


def delete_document(parent_id: str, collection: Collection) -> int:
    """Remove all chunks for a single document by *parent_id*."""
    return _delete_chunks_by_metadata(collection, "parent_id", parent_id)


# ---------------------------------------------------------------------------
# Query / listing helpers
# ---------------------------------------------------------------------------


def list_sources(collection: Collection) -> list[dict[str, Any]]:
    """Return a list of distinct sources with chunk counts and timestamps.

    Each entry: ``{"source": str, "chunk_count": int, "latest_ingested_at": str}``.
    Sources that have no ``source`` metadata (e.g. the initial Excel bulk build)
    are grouped under ``"excel::bulk_build"``.
    """
    all_meta = collection.get(include=["metadatas"])
    metadatas = all_meta.get("metadatas") or []

    source_info: dict[str, dict[str, Any]] = {}
    for meta in metadatas:
        src = (meta or {}).get("source", "excel::bulk_build")
        info = source_info.setdefault(src, {"chunk_count": 0, "latest_ingested_at": ""})
        info["chunk_count"] += 1
        ts = (meta or {}).get("ingested_at", "")
        if ts > info["latest_ingested_at"]:
            info["latest_ingested_at"] = ts

    return [
        {"source": src, **info}
        for src, info in sorted(source_info.items())
    ]


def list_documents(
    collection: Collection,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return per-document summaries for a source (or all sources).

    Each entry: ``{"parent_id", "product", "category", "question",
    "content_preview", "chunk_count", "source", "ingested_at"}``.
    """
    if source is not None:
        result = collection.get(
            where={"source": source},
            include=["metadatas", "documents"],
        )
    else:
        result = collection.get(include=["metadatas", "documents"])

    metadatas = result.get("metadatas") or []
    doc_texts = result.get("documents") or []

    # Group by parent_id
    parent_groups: dict[str, dict[str, Any]] = {}
    for meta, text in zip(metadatas, doc_texts):
        pid = (meta or {}).get("parent_id", "unknown")
        if pid not in parent_groups:
            parent_groups[pid] = {
                "parent_id": pid,
                "product": (meta or {}).get("product", ""),
                "category": (meta or {}).get("category", ""),
                "question": (meta or {}).get("question", ""),
                "source": (meta or {}).get("source", "excel::bulk_build"),
                "ingested_at": (meta or {}).get("ingested_at", ""),
                "chunk_count": 0,
                "content_preview": "",
            }
        parent_groups[pid]["chunk_count"] += 1
        # Keep the longest content preview (first chunk usually has the most context)
        if len(text or "") > len(parent_groups[pid]["content_preview"]):
            parent_groups[pid]["content_preview"] = (text or "")[:150]

    return sorted(parent_groups.values(), key=lambda d: d["parent_id"])


def get_document(parent_id: str, collection: Collection) -> list[dict[str, Any]]:
    """Return all chunks for a specific document, with full content."""
    result = collection.get(
        where={"parent_id": parent_id},
        include=["metadatas", "documents"],
    )
    chunks = []
    for chunk_id, meta, text in zip(
        result.get("ids") or [],
        result.get("metadatas") or [],
        result.get("documents") or [],
    ):
        chunks.append({"id": chunk_id, "content": text, **(meta or {})})
    return chunks


def get_stats(
    collection: Collection,
    vectorstore_path: Path | str = VECTORSTORE_PATH,
) -> dict[str, Any]:
    """Return vectorstore statistics."""
    sources = list_sources(collection)
    return {
        "total_chunks": collection.count(),
        "source_count": len(sources),
        "sources": sources,
        "disk_size_bytes": _collection_disk_size(vectorstore_path),
    }


# ---------------------------------------------------------------------------
# High-level ingestion
# ---------------------------------------------------------------------------


def ingest_file(
    path: Path | str,
    collection: Collection,
    model: Any,
    source_tag: str | None = None,
) -> dict[str, Any]:
    """Ingest a JSON file (auto-detected format).

    If *source_tag* is not given, it is derived from the filename.
    All existing chunks for that source are replaced.
    """
    path = Path(path)
    source = source_tag or _sanitize_source_tag(path.name)
    documents = detect_and_parse(path)
    return add_documents(documents, source, collection, model)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _delete_chunks_by_metadata(
    collection: Collection,
    key: str,
    value: str,
) -> int:
    """Delete all chunks where ``metadata[key] == value``.  Returns count deleted."""
    try:
        existing = collection.get(where={key: value}, include=[])
    except Exception:
        return 0
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)
