"""
Local Vector Database constructor module.

Handles taking processed JSON documents, chunking them optimally for embedding,
calculating embeddings utilizing BAAI/bge-small-en-v1.5, and persisting 
the indices inside a local ChromaDB instance across sessions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "data/processed/documents.json"
VECTORSTORE_PATH = PROJECT_ROOT / "data/vectorstore"
MODEL_CACHE_DIR = PROJECT_ROOT / "data/models"
COLLECTION_NAME = "bank_knowledge"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_documents(documents_path: Path | str = DOCUMENTS_PATH) -> list[dict[str, Any]]:
    """Load JSON array of processed documents."""
    return json.loads(Path(documents_path).read_text(encoding="utf-8"))


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(
    documents: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[dict[str, Any]]:
    """Chunks documents using Langchain Recursive text splitter."""
    splitter = splitter or get_text_splitter()
    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = document["content"].strip()
        if not text:
            continue
            
        if document.get("type") == "qa_pair" and len(text) < 4000:
            parts = [text]
        else:
            parts = splitter.split_text(text)
            
        if not parts:
            parts = [text]
        for index, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if document["type"] == "qa_pair" and len(parts) > 1 and document.get("question"):
                question_prefix = f"Q: {document['question']}\n"
                if not part.startswith(question_prefix):
                    part = f"{question_prefix}{part}"
            chunks.append(
                {
                    "id": f"{document['id']}_chunk_{index:02d}",
                    "parent_id": document["id"],
                    "product": document["product"],
                    "sheet": document["sheet"],
                    "type": document["type"],
                    "question": document.get("question") or "",
                    "category": document["category"],
                    "content": part,
                }
            )
    return chunks


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Instantiate or load cached SentenceTransformer embedding model."""
    local_path = MODEL_CACHE_DIR / model_name.replace("/", "__")
    if local_path.exists():
        return SentenceTransformer(str(local_path))
    model = SentenceTransformer(model_name)
    local_path.mkdir(parents=True, exist_ok=True)
    model.save(str(local_path))
    return model


def encode_texts(model: Any, texts: list[str] | str) -> np.ndarray:
    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
    except TypeError:
        embeddings = model.encode(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if isinstance(texts, list) and embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def _create_client(vectorstore_path: Path | str = VECTORSTORE_PATH) -> chromadb.PersistentClient:
    path = Path(vectorstore_path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collection_disk_size(path: Path | str) -> int:
    base = Path(path)
    if not base.exists():
        return 0
    return sum(file.stat().st_size for file in base.rglob("*") if file.is_file())


def build_vectorstore(
    documents_path: Path | str = DOCUMENTS_PATH,
    vectorstore_path: Path | str = VECTORSTORE_PATH,
    *,
    model: Any | None = None,
    force_rebuild: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    """Orchestrates document reading, chunking, and ChromaDB uploading."""
    model = model or get_embedding_model()
    documents = load_documents(documents_path)
    chunks = chunk_documents(documents)
    client = _create_client(vectorstore_path)

    if force_rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force_rebuild:
        return {
            "collection": collection,
            "chunks": chunks,
            "count": collection.count(),
            "size_bytes": _collection_disk_size(vectorstore_path),
            "skipped": True,
        }

    if collection.count() > 0 and force_rebuild:
        client.delete_collection(collection_name)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    embeddings = encode_texts(model, [chunk["content"] for chunk in chunks])
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk["content"] for chunk in chunks],
        metadatas=[
            {
                "parent_id": chunk["parent_id"],
                "product": chunk["product"],
                "sheet": chunk["sheet"],
                "chunk_type": chunk["type"],
                "question": chunk["question"],
                "category": chunk["category"],
                "source": "excel::bulk_build",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            for chunk in chunks
        ],
    )
    return {
        "collection": collection,
        "chunks": chunks,
        "count": collection.count(),
        "size_bytes": _collection_disk_size(vectorstore_path),
        "skipped": False,
    }


def load_vectorstore(
    vectorstore_path: Path | str = VECTORSTORE_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Collection:
    client = _create_client(vectorstore_path)
    return client.get_collection(collection_name)


def query_vectorstore(
    query: str,
    *,
    vectorstore_path: Path | str = VECTORSTORE_PATH,
    collection: Collection | None = None,
    model: Any | None = None,
    top_k: int = 10,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    collection = collection or load_vectorstore(vectorstore_path)
    model = model or get_embedding_model()
    query_embedding = encode_texts(model, [f"{QUERY_PREFIX}{query}"]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    normalized_results: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        row = {"content": document, "distance": distance}
        row.update(metadata or {})
        normalized_results.append(row)
    return normalized_results


def main() -> None:
    build_result = build_vectorstore()
    print(
        f"Built vectorstore: {build_result['count']} chunks"
        if not build_result["skipped"]
        else f"Vectorstore already exists, skipping rebuild. Existing chunks: {build_result['count']}"
    )
    print(f"Collection size: {build_result['size_bytes']} bytes")
    collection = build_result["collection"]
    model = get_embedding_model()
    for query in (
        "account for senior citizens",
        "auto finance markup rate",
        "profit rate savings account",
    ):
        print(f"\nQuery: {query}")
        for result in query_vectorstore(query, collection=collection, model=model, top_k=3):
            preview = result["content"].replace("\n", " ")[:140]
            print(f"- {result.get('product', 'Unknown')} [{result.get('category', 'n/a')}] :: {preview}")


if __name__ == "__main__":
    main()