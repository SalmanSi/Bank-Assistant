"""Tests for scripts.document_manager — CRUD operations on the vectorstore."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.document_manager import (
    add_documents,
    add_single_document,
    delete_by_source,
    delete_document,
    detect_and_parse,
    get_document,
    get_stats,
    ingest_file,
    list_documents,
    list_sources,
    parse_faq_json,
)
from scripts.build_vectordb import (
    _create_client,
    query_vectorstore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    """Deterministic embedding model for testing."""

    dimension = 384
    keywords = [
        "transfer", "limit", "funds", "app", "rate",
        "profit", "account", "senior", "citizen", "faq",
        "new", "feature",
    ]

    def _vectorize(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.zeros(self.dimension, dtype=np.float32)
        for i, kw in enumerate(self.keywords):
            vector[i] = lowered.count(kw)
        vector[len(self.keywords)] = len(lowered) / 1000.0
        return vector

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            return self._vectorize(texts)
        return np.vstack([self._vectorize(t) for t in texts])


@pytest.fixture()
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture()
def collection(tmp_path: Path):
    """Create a fresh in-process ChromaDB collection."""
    client = _create_client(tmp_path / "vectorstore")
    return client.get_or_create_collection(
        name="test_knowledge",
        metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture()
def sample_docs() -> list[dict]:
    return [
        {
            "id": "DOC_001",
            "product": "Funds Transfer",
            "sheet": "faq_upload",
            "type": "qa_pair",
            "category": "faq",
            "question": "Is there a limit on funds transfer?",
            "content": "Q: Is there a limit on funds transfer?\nA: Yes, 1 million is the current daily limit.",
        },
        {
            "id": "DOC_002",
            "product": "App Features",
            "sheet": "faq_upload",
            "type": "qa_pair",
            "category": "faq",
            "question": "Can I use the app overseas?",
            "content": "Q: Can I use the app overseas?\nA: Yes, the NUST mobile app can be accessed globally.",
        },
    ]


@pytest.fixture()
def faq_json_path(tmp_path: Path) -> Path:
    """Write a sample FAQ JSON file and return its path."""
    data = {
        "categories": [
            {
                "category": "Funds Transfer",
                "questions": [
                    {
                        "question": "Is there a transfer limit?",
                        "answer": "Yes, 1 million daily limit.",
                    },
                    {
                        "question": "How to change transfer limit?",
                        "answer": "Go to My Profile > Manage Limit.",
                    },
                ],
            },
            {
                "category": "App Features",
                "questions": [
                    {
                        "question": "Can I use the app overseas?",
                        "answer": "Yes, with internet connectivity.",
                    },
                ],
            },
        ]
    }
    path = tmp_path / "test_faq.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def docs_json_path(tmp_path: Path, sample_docs) -> Path:
    """Write a documents-format JSON file and return its path."""
    path = tmp_path / "test_docs.json"
    path.write_text(json.dumps(sample_docs), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests: parse_faq_json
# ---------------------------------------------------------------------------


class TestParseFaqJson:
    def test_correct_count(self, faq_json_path):
        docs = parse_faq_json(faq_json_path)
        assert len(docs) == 3

    def test_document_format(self, faq_json_path):
        docs = parse_faq_json(faq_json_path)
        required = {"id", "product", "sheet", "type", "category", "question", "content"}
        for doc in docs:
            assert required.issubset(doc.keys())
            assert doc["type"] == "qa_pair"
            assert doc["content"].startswith("Q: ")
            assert "\nA: " in doc["content"]

    def test_categories_preserved(self, faq_json_path):
        docs = parse_faq_json(faq_json_path)
        products = {d["product"] for d in docs}
        assert "Funds Transfer" in products
        assert "App Features" in products


# ---------------------------------------------------------------------------
# Tests: detect_and_parse
# ---------------------------------------------------------------------------


class TestDetectAndParse:
    def test_detects_faq_format(self, faq_json_path):
        docs = detect_and_parse(faq_json_path)
        assert len(docs) == 3
        assert all(d["type"] == "qa_pair" for d in docs)

    def test_detects_documents_format(self, docs_json_path):
        docs = detect_and_parse(docs_json_path)
        assert len(docs) == 2

    def test_rejects_unknown_format(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"foo": "bar"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Unrecognized"):
            detect_and_parse(path)


# ---------------------------------------------------------------------------
# Tests: add_documents (source-level replacement)
# ---------------------------------------------------------------------------


class TestAddDocuments:
    def test_documents_are_added(self, collection, fake_model, sample_docs):
        result = add_documents(sample_docs, "test::source", collection, fake_model)
        assert result["added"] >= 2
        assert result["source"] == "test::source"
        assert collection.count() >= 2

    def test_source_replacement(self, collection, fake_model, sample_docs):
        """Re-adding same source deletes old chunks and adds new — no duplicates."""
        add_documents(sample_docs, "test::source", collection, fake_model)
        count_after_first = collection.count()

        # Re-add with same source — should replace, not accumulate
        add_documents(sample_docs, "test::source", collection, fake_model)
        count_after_second = collection.count()

        assert count_after_second == count_after_first

    def test_source_replacement_with_different_content(self, collection, fake_model):
        """Replacing source with different content removes old, adds new."""
        old_docs = [
            {
                "id": "OLD_001",
                "product": "Rate Sheet",
                "sheet": "rates",
                "type": "rate_info",
                "category": "rate",
                "question": None,
                "content": "NAA interest rate is 19% per annum.",
            }
        ]
        new_docs = [
            {
                "id": "NEW_001",
                "product": "Rate Sheet",
                "sheet": "rates",
                "type": "rate_info",
                "category": "rate",
                "question": None,
                "content": "All account rates reduced to 13% except Premium Savings at 20%.",
            }
        ]

        add_documents(old_docs, "test::rates", collection, fake_model)
        assert collection.count() >= 1

        # Replace with new content
        result = add_documents(new_docs, "test::rates", collection, fake_model)
        assert result["deleted"] >= 1
        assert result["added"] >= 1

        # Verify old content is gone — get all documents and check
        all_data = collection.get(include=["documents"])
        all_texts = " ".join(all_data["documents"])
        assert "19%" not in all_texts
        assert "13%" in all_texts

    def test_different_sources_coexist(self, collection, fake_model, sample_docs):
        """Chunks from different sources are independent."""
        doc_a = [sample_docs[0]]
        doc_b = [sample_docs[1]]

        add_documents(doc_a, "source_a", collection, fake_model)
        add_documents(doc_b, "source_b", collection, fake_model)

        # Delete source_a — source_b should remain
        delete_by_source("source_a", collection)
        assert collection.count() >= 1

        remaining = collection.get(include=["metadatas"])
        sources = {m.get("source") for m in remaining["metadatas"]}
        assert "source_a" not in sources
        assert "source_b" in sources


# ---------------------------------------------------------------------------
# Tests: add_single_document
# ---------------------------------------------------------------------------


class TestAddSingleDocument:
    def test_add_single(self, collection, fake_model):
        doc = {
            "id": "SINGLE_001",
            "product": "Test Product",
            "sheet": "manual",
            "type": "qa_pair",
            "category": "faq",
            "question": "What is NUST Bank's new feature?",
            "content": "Q: What is NUST Bank's new feature?\nA: Instant payments.",
        }
        result = add_single_document(doc, "manual::single_faq", collection, fake_model)
        assert result["added"] >= 1
        assert collection.count() >= 1

    def test_add_single_replaces_existing(self, collection, fake_model):
        """Re-adding same parent_id replaces old content."""
        doc_v1 = {
            "id": "SINGLE_001",
            "product": "Test",
            "sheet": "manual",
            "type": "qa_pair",
            "category": "faq",
            "content": "Q: Old question?\nA: Old answer.",
        }
        doc_v2 = {
            "id": "SINGLE_001",
            "product": "Test",
            "sheet": "manual",
            "type": "qa_pair",
            "category": "faq",
            "content": "Q: New question?\nA: New answer.",
        }

        add_single_document(doc_v1, "manual::single", collection, fake_model)
        count_v1 = collection.count()

        add_single_document(doc_v2, "manual::single", collection, fake_model)
        count_v2 = collection.count()

        assert count_v2 == count_v1  # replaced, not accumulated

        all_data = collection.get(include=["documents"])
        all_texts = " ".join(all_data["documents"])
        assert "New answer" in all_texts
        assert "Old answer" not in all_texts


# ---------------------------------------------------------------------------
# Tests: delete operations
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_source(self, collection, fake_model, sample_docs):
        add_documents(sample_docs, "test::source", collection, fake_model)
        assert collection.count() > 0

        deleted = delete_by_source("test::source", collection)
        assert deleted > 0
        assert collection.count() == 0

    def test_delete_single_document(self, collection, fake_model, sample_docs):
        add_documents(sample_docs, "test::source", collection, fake_model)
        total_before = collection.count()

        deleted = delete_document("DOC_001", collection)
        assert deleted >= 1
        assert collection.count() < total_before

        # DOC_002 should still be there
        remaining = collection.get(include=["metadatas"])
        parent_ids = {m.get("parent_id") for m in remaining["metadatas"]}
        assert "DOC_001" not in parent_ids
        assert "DOC_002" in parent_ids

    def test_delete_nonexistent_returns_zero(self, collection):
        assert delete_by_source("nonexistent::source", collection) == 0
        assert delete_document("NONEXISTENT_ID", collection) == 0


# ---------------------------------------------------------------------------
# Tests: listing and stats
# ---------------------------------------------------------------------------


class TestListing:
    def test_list_sources(self, collection, fake_model, sample_docs):
        add_documents(sample_docs[:1], "source_a", collection, fake_model)
        add_documents(sample_docs[1:], "source_b", collection, fake_model)

        sources = list_sources(collection)
        source_names = {s["source"] for s in sources}
        assert "source_a" in source_names
        assert "source_b" in source_names
        for s in sources:
            assert s["chunk_count"] >= 1

    def test_list_documents_for_source(self, collection, fake_model, sample_docs):
        add_documents(sample_docs, "test::source", collection, fake_model)

        docs = list_documents(collection, source="test::source")
        assert len(docs) >= 2
        parent_ids = {d["parent_id"] for d in docs}
        assert "DOC_001" in parent_ids
        assert "DOC_002" in parent_ids

        for doc in docs:
            assert "content_preview" in doc
            assert "chunk_count" in doc
            assert doc["chunk_count"] >= 1

    def test_list_documents_all(self, collection, fake_model, sample_docs):
        add_documents(sample_docs[:1], "source_a", collection, fake_model)
        add_documents(sample_docs[1:], "source_b", collection, fake_model)

        docs = list_documents(collection)
        assert len(docs) >= 2

    def test_get_document(self, collection, fake_model, sample_docs):
        add_documents(sample_docs, "test::source", collection, fake_model)

        chunks = get_document("DOC_001", collection)
        assert len(chunks) >= 1
        assert all("content" in c for c in chunks)
        assert all(c.get("parent_id") == "DOC_001" for c in chunks)

    def test_get_stats(self, collection, fake_model, sample_docs, tmp_path):
        add_documents(sample_docs, "test::source", collection, fake_model)

        stats = get_stats(collection, tmp_path / "vectorstore")
        assert stats["total_chunks"] >= 2
        assert stats["source_count"] >= 1


# ---------------------------------------------------------------------------
# Tests: metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_source_and_timestamp_present(self, collection, fake_model, sample_docs):
        add_documents(sample_docs, "test::source", collection, fake_model)

        all_data = collection.get(include=["metadatas"])
        for meta in all_data["metadatas"]:
            assert meta.get("source") == "test::source"
            assert meta.get("ingested_at"), "ingested_at must be non-empty"
            # Verify ISO format
            datetime.fromisoformat(meta["ingested_at"])


# ---------------------------------------------------------------------------
# Tests: ingest_file (end-to-end)
# ---------------------------------------------------------------------------


class TestIngestFile:
    def test_ingest_faq_json(self, collection, fake_model, faq_json_path):
        result = ingest_file(faq_json_path, collection, fake_model)
        assert result["added"] >= 3
        assert collection.count() >= 3

    def test_ingest_docs_json(self, collection, fake_model, docs_json_path):
        result = ingest_file(docs_json_path, collection, fake_model)
        assert result["added"] >= 2

    def test_ingest_re_upload_replaces(self, collection, fake_model, faq_json_path):
        """Re-uploading same file replaces old chunks."""
        ingest_file(faq_json_path, collection, fake_model, source_tag="faq::test")
        count_first = collection.count()

        ingest_file(faq_json_path, collection, fake_model, source_tag="faq::test")
        count_second = collection.count()

        assert count_second == count_first

    def test_updated_doc_immediately_searchable(
        self, collection, fake_model, tmp_path
    ):
        """After update, query finds new content, not old."""
        faq_v1 = {
            "categories": [
                {
                    "category": "Rates",
                    "questions": [
                        {
                            "question": "What is the transfer limit?",
                            "answer": "The daily transfer limit is 1 million.",
                        }
                    ],
                }
            ]
        }
        faq_v2 = {
            "categories": [
                {
                    "category": "Rates",
                    "questions": [
                        {
                            "question": "What is the transfer limit?",
                            "answer": "The daily transfer limit is 500 thousand.",
                        }
                    ],
                }
            ]
        }

        path = tmp_path / "rates.json"
        path.write_text(json.dumps(faq_v1), encoding="utf-8")
        ingest_file(path, collection, fake_model, source_tag="faq::rates")

        # Verify v1 content is there
        all_data = collection.get(include=["documents"])
        assert any("1 million" in d for d in all_data["documents"])

        # Update with v2
        path.write_text(json.dumps(faq_v2), encoding="utf-8")
        ingest_file(path, collection, fake_model, source_tag="faq::rates")

        # Verify v2 content is there and v1 is gone
        all_data = collection.get(include=["documents"])
        all_text = " ".join(all_data["documents"])
        assert "500 thousand" in all_text
        assert "1 million" not in all_text

    def test_existing_data_unaffected(self, collection, fake_model, sample_docs, faq_json_path):
        """CRUD on one source doesn't touch another source's data."""
        add_documents(sample_docs, "source::excel", collection, fake_model)
        excel_count = collection.count()

        # Add FAQ source
        ingest_file(faq_json_path, collection, fake_model, source_tag="faq::test")
        assert collection.count() > excel_count

        # Delete FAQ source
        delete_by_source("faq::test", collection)
        assert collection.count() == excel_count

        # Excel data still intact
        remaining = collection.get(include=["metadatas"])
        sources = {m.get("source") for m in remaining["metadatas"]}
        assert "source::excel" in sources
