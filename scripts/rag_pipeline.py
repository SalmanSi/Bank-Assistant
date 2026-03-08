from __future__ import annotations

import logging
from typing import Any, Generator

import ollama

from scripts.build_vectordb import get_embedding_model, load_vectorstore, query_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RAG] %(message)s",
    datefmt="%H:%M:%S",
)

OLLAMA_MODEL = "qwen3:0.6b"

SYSTEM_PROMPT_TEMPLATE = """You are a helpful bank assistant for NUST Bank. Answer questions about banking products and services using only the context below. Be concise and accurate. If the answer is not in the context, say "I don't have information about that."

Context:
{context}"""


def retrieve(
    query: str,
    *,
    collection: Any | None = None,
    model: Any | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a query from the vectorstore."""
    if collection is None:
        collection = load_vectorstore()
    if model is None:
        model = get_embedding_model()
    return query_vectorstore(query, collection=collection, model=model, top_k=top_k)


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a single context string."""
    parts = []
    for chunk in chunks:
        product = chunk.get("product", "Unknown")
        content = chunk.get("content", "")
        parts.append(f"[{product}]\n{content}")
    return "\n\n---\n\n".join(parts)


def ask(
    query: str,
    *,
    collection: Any | None = None,
    model: Any | None = None,
    top_k: int = 10,
    stream: bool = False,
) -> str | Generator:
    """Run the full RAG pipeline: retrieve context then generate an answer via Ollama.

    Parameters
    ----------
    query:
        The user's question.
    collection:
        Pre-loaded ChromaDB collection. Loaded from disk if not provided.
    model:
        Pre-loaded SentenceTransformer embedding model. Loaded if not provided.
    top_k:
        Number of context chunks to retrieve.
    stream:
        If True, returns an Ollama streaming generator instead of a string.
    """
    chunks = retrieve(query, collection=collection, model=model, top_k=top_k)
    context = build_context(chunks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    logging.info("Query      : %s", query)
    logging.info("Chunks     : %d retrieved", len(chunks))
    for i, chunk in enumerate(chunks, 1):
        logging.info(
            "  [%d] product=%-30s  category=%-12s  dist=%.4f",
            i,
            chunk.get("product", "?"),
            chunk.get("category", "?"),
            chunk.get("distance", 0.0),
        )
    logging.info("Context    :\n%s", context)
    logging.info("Calling    : %s (stream=%s)", OLLAMA_MODEL, stream)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        think=False,
        stream=stream,
    )

    if stream:
        return response  # generator yielding chunks with response["message"]["content"]
    return response["message"]["content"]  # type: ignore[index]
