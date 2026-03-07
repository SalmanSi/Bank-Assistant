from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_vectordb import (
    build_vectorstore,
    chunk_documents,
    encode_texts,
    load_vectorstore,
    query_vectorstore,
)


class FakeEmbeddingModel:
    dimension = 384
    keywords = [
        "senior",
        "citizen",
        "auto",
        "finance",
        "markup",
        "profit",
        "rate",
        "savings",
        "waqaar",
        "consumer",
        "liability",
    ]

    def _vectorize(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.zeros(self.dimension, dtype=np.float32)
        for index, keyword in enumerate(self.keywords):
            vector[index] = lowered.count(keyword)
        vector[len(self.keywords)] = len(lowered) / 1000.0
        return vector

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            return self._vectorize(texts)
        return np.vstack([self._vectorize(text) for text in texts])


@pytest.fixture()
def sample_documents() -> list[dict[str, str | None]]:
    return [
        {
            "id": "NWA_001",
            "product": "NUST Waqaar Account",
            "sheet": "NWA",
            "type": "qa_pair",
            "category": "liability",
            "question": "Does your bank offer any account for senior citizens?",
            "content": "Q: Does your bank offer any account for senior citizens?\nA: NUST Waqaar Account is designed for senior citizens and offers monthly profit.",
        },
        {
            "id": "CAR_001",
            "product": "NUST4Car",
            "sheet": "NUST4Car",
            "type": "qa_pair",
            "category": "consumer",
            "question": "What is the markup rate for auto finance?",
            "content": "Q: What is the markup rate for auto finance?\nA: NUST4Car is the auto finance facility. It includes markup rate guidance and vehicle financing options.",
        },
        {
            "id": "RATE_001",
            "product": "Rate Sheet",
            "sheet": "Rate Sheet July 1 2024",
            "type": "rate_info",
            "category": "rate",
            "question": None,
            "content": "NUST Waqaar Account Savings: Profit payment Monthly at 19.00% per annum. Term deposit one year monthly payout at 16.75%.",
        },
        {
            "id": "LONG_001",
            "product": "Reference",
            "sheet": "Reference",
            "type": "general_info",
            "category": "general",
            "question": None,
            "content": " ".join(["profit rate savings account"] * 80),
        },
    ]


@pytest.fixture()
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture()
def documents_path(tmp_path: Path, sample_documents) -> Path:
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(sample_documents), encoding="utf-8")
    return path


def test_chunks_created(sample_documents):
    chunks = chunk_documents(sample_documents)
    assert len(chunks) >= len(sample_documents)


def test_chunk_size_within_limit(sample_documents):
    chunks = chunk_documents(sample_documents)
    assert max(len(chunk["content"]) for chunk in chunks) <= 600


def test_chunk_metadata_complete(sample_documents):
    chunks = chunk_documents(sample_documents)
    required = {"id", "parent_id", "product", "sheet", "type", "question", "category", "content"}
    for chunk in chunks:
        assert required.issubset(chunk)


def test_embedding_dimension(fake_model):
    embedding = encode_texts(fake_model, ["hello world"])
    assert embedding.shape == (1, 384)


def test_vectorstore_built(documents_path, tmp_path: Path, fake_model):
    result = build_vectorstore(documents_path, tmp_path / "vectorstore", model=fake_model)
    assert (tmp_path / "vectorstore").exists()
    assert result["count"] > 0


def test_vectorstore_doc_count(documents_path, tmp_path: Path, fake_model):
    result = build_vectorstore(documents_path, tmp_path / "vectorstore_doc_count", model=fake_model)
    assert result["count"] == len(result["chunks"])


def test_load_vectorstore(documents_path, tmp_path: Path, fake_model):
    vectorstore_path = tmp_path / "vectorstore_load"
    build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    collection = load_vectorstore(vectorstore_path)
    assert collection.count() > 0


def test_unfiltered_query_returns_results(documents_path, tmp_path: Path, fake_model):
    vectorstore_path = tmp_path / "vectorstore_query"
    build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    results = query_vectorstore("senior citizen account", vectorstore_path=vectorstore_path, model=fake_model)
    assert len(results) >= 1


def test_filtered_query_by_category(documents_path, tmp_path: Path, fake_model):
    vectorstore_path = tmp_path / "vectorstore_category"
    build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    results = query_vectorstore(
        "senior citizen account",
        vectorstore_path=vectorstore_path,
        model=fake_model,
        where={"category": "liability"},
    )
    assert results
    assert all(result["category"] == "liability" for result in results)


def test_filtered_query_by_product(documents_path, tmp_path: Path, fake_model):
    vectorstore_path = tmp_path / "vectorstore_product"
    build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    results = query_vectorstore(
        "senior citizen account",
        vectorstore_path=vectorstore_path,
        model=fake_model,
        where={"product": "NUST Waqaar Account"},
    )
    assert results
    assert all(result["product"] == "NUST Waqaar Account" for result in results)


def test_query_result_fields_present(documents_path, tmp_path: Path, fake_model):
    vectorstore_path = tmp_path / "vectorstore_fields"
    build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    results = query_vectorstore("profit rate savings account", vectorstore_path=vectorstore_path, model=fake_model)
    assert results
    assert {"content", "product", "category"}.issubset(results[0])


def test_vectorstore_persistence(documents_path, tmp_path: Path, fake_model):
    vectorstore_path = tmp_path / "vectorstore_persist"
    first = build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    collection = load_vectorstore(vectorstore_path)
    assert collection.count() == first["count"]


def test_rebuild_skipped_if_exists(documents_path, tmp_path: Path, fake_model):
    vectorstore_path = tmp_path / "vectorstore_skip"
    first = build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    second = build_vectorstore(documents_path, vectorstore_path, model=fake_model)
    assert first["count"] == second["count"]
    assert second["skipped"] is True