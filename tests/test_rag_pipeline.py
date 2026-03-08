from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.rag_pipeline import ask, build_context, retrieve


SAMPLE_CHUNKS = [
    {
        "product": "NUST Asaan Account (NAA)",
        "content": "Resident Pakistani individuals who do not maintain any other account are eligible.",
        "category": "liability",
        "sheet": "NAA",
        "question": "What is the eligibility for NAA?",
        "distance": 0.12,
        "parent_id": "NAA_001",
        "chunk_type": "qa_pair",
    },
    {
        "product": "Rate Sheet",
        "content": "NUST Asaan Account Savings: Profit payment Semi-Annually at 19.00% per annum.",
        "category": "rate",
        "sheet": "Rate Sheet July 1 2024",
        "question": "",
        "distance": 0.25,
        "parent_id": "RATE_001",
        "chunk_type": "rate_info",
    },
]


class TestBuildContext:
    def test_product_label_present(self):
        ctx = build_context(SAMPLE_CHUNKS)
        assert "NUST Asaan Account (NAA)" in ctx
        assert "Rate Sheet" in ctx

    def test_content_present(self):
        ctx = build_context(SAMPLE_CHUNKS)
        assert "Resident Pakistani individuals" in ctx
        assert "19.00%" in ctx

    def test_chunks_separated_by_divider(self):
        ctx = build_context(SAMPLE_CHUNKS)
        assert "---" in ctx

    def test_empty_chunks_returns_empty_string(self):
        assert build_context([]) == ""

    def test_single_chunk_no_divider(self):
        ctx = build_context([SAMPLE_CHUNKS[0]])
        assert "---" not in ctx


class TestRetrieve:
    def test_delegates_to_query_vectorstore(self):
        mock_collection = MagicMock()
        mock_model = MagicMock()

        with patch("scripts.rag_pipeline.query_vectorstore", return_value=SAMPLE_CHUNKS) as mock_qvs:
            result = retrieve("eligibility NAA", collection=mock_collection, model=mock_model, top_k=3)

        mock_qvs.assert_called_once_with(
            "eligibility NAA",
            collection=mock_collection,
            model=mock_model,
            top_k=3,
        )
        assert result == SAMPLE_CHUNKS

    def test_loads_resources_when_not_provided(self):
        fake_collection = MagicMock()
        fake_model = MagicMock()

        with patch("scripts.rag_pipeline.load_vectorstore", return_value=fake_collection) as mock_lvs, \
             patch("scripts.rag_pipeline.get_embedding_model", return_value=fake_model) as mock_gem, \
             patch("scripts.rag_pipeline.query_vectorstore", return_value=[]) as _:
            retrieve("test query")

        mock_lvs.assert_called_once()
        mock_gem.assert_called_once()


class TestAsk:
    def _make_ollama_response(self, text: str) -> dict:
        return {"message": {"content": text, "role": "assistant"}}

    def test_returns_string_when_not_streaming(self):
        mock_collection = MagicMock()
        mock_model = MagicMock()

        with patch("scripts.rag_pipeline.retrieve", return_value=SAMPLE_CHUNKS), \
             patch("scripts.rag_pipeline.ollama.chat", return_value=self._make_ollama_response("Eligible residents.")):
            result = ask("Who is eligible for NAA?", collection=mock_collection, model=mock_model)

        assert isinstance(result, str)
        assert result == "Eligible residents."

    def test_returns_generator_when_streaming(self):
        mock_collection = MagicMock()
        mock_model = MagicMock()

        fake_stream = iter([
            {"message": {"content": "Eligible ", "role": "assistant"}},
            {"message": {"content": "residents.", "role": "assistant"}},
        ])

        with patch("scripts.rag_pipeline.retrieve", return_value=SAMPLE_CHUNKS), \
             patch("scripts.rag_pipeline.ollama.chat", return_value=fake_stream):
            result = ask("Who is eligible?", collection=mock_collection, model=mock_model, stream=True)

        # Should be iterable (the raw generator)
        tokens = list(result)
        assert len(tokens) == 2
        assert tokens[0]["message"]["content"] == "Eligible "

    def test_ollama_called_with_correct_model(self):
        from scripts.rag_pipeline import OLLAMA_MODEL

        mock_collection = MagicMock()
        mock_model = MagicMock()

        with patch("scripts.rag_pipeline.retrieve", return_value=SAMPLE_CHUNKS), \
             patch("scripts.rag_pipeline.ollama.chat", return_value=self._make_ollama_response("OK")) as mock_chat:
            ask("test", collection=mock_collection, model=mock_model)

        assert mock_chat.call_args.kwargs["model"] == OLLAMA_MODEL

    def test_context_injected_into_system_prompt(self):
        mock_collection = MagicMock()
        mock_model = MagicMock()

        with patch("scripts.rag_pipeline.retrieve", return_value=SAMPLE_CHUNKS), \
             patch("scripts.rag_pipeline.ollama.chat", return_value=self._make_ollama_response("OK")) as mock_chat:
            ask("test query", collection=mock_collection, model=mock_model)

        call_kwargs = mock_chat.call_args.kwargs
        messages = call_kwargs["messages"]
        system_content = messages[0]["content"]
        # Context should be embedded in the system prompt
        assert "NUST Asaan Account (NAA)" in system_content
        assert "Resident Pakistani individuals" in system_content
