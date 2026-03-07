from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.preprocess import (
    PII_SCAN_PATH,
    WORKBOOK_PATH,
    load_bank_workbook,
    parse_main_sheet,
    preprocess_workbook,
    scan_pii,
)


@pytest.fixture(scope="module")
def workbook():
    return load_bank_workbook(WORKBOOK_PATH)


@pytest.fixture(scope="module")
def processed_documents(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("processed")
    output_path = tmp_dir / "documents.json"
    pii_path = tmp_dir / "pii_scan.json"
    documents = preprocess_workbook(WORKBOOK_PATH, output_path, pii_path)
    return {
        "documents": documents,
        "output_path": output_path,
        "pii_path": pii_path,
    }


def test_pii_scan_finds_no_customer_pii(workbook):
    findings = scan_pii(workbook)
    assert findings["cnic_hits"] == []
    assert findings["personal_phone_hits"] == []
    assert findings["personal_email_hits"] == []


def test_category_map_built(workbook):
    category_map, _ = parse_main_sheet(workbook)
    product_sheets = [sheet for sheet in workbook.sheetnames if sheet not in {"Main", "Rate Sheet July 1 2024", "Sheet1"}]
    assert len(category_map) == len(product_sheets) == 33
    assert set(category_map) == set(product_sheets)


def test_qa_pairs_extracted(processed_documents):
    documents = processed_documents["documents"]
    qa_pairs = [document for document in documents if document["type"] == "qa_pair"]
    assert qa_pairs
    sheets_with_qas = {document["sheet"] for document in qa_pairs}
    expected_sheets = {document["sheet"] for document in documents if document["sheet"] != "Rate Sheet July 1 2024"}
    assert expected_sheets.issubset(sheets_with_qas | {"Rate Sheet July 1 2024"})
    assert all(document["question"] and document["content"].startswith("Q: ") for document in qa_pairs)


def test_rate_sheet_parsed(processed_documents):
    rate_docs = [document for document in processed_documents["documents"] if document["type"] == "rate_info"]
    assert rate_docs
    assert any("%" in document["content"] for document in rate_docs)
    assert any("Account" in document["content"] or "Deposit" in document["content"] for document in rate_docs)


def test_text_cleaning_removes_nbsp(processed_documents):
    assert all("\xa0" not in document["content"] for document in processed_documents["documents"])


def test_text_cleaning_removes_navigation(processed_documents):
    assert all(document["content"].strip() != "Main" for document in processed_documents["documents"])


def test_output_schema_valid(processed_documents):
    required_keys = {"id", "product", "sheet", "type", "category", "content"}
    for document in processed_documents["documents"]:
        assert required_keys.issubset(document)


def test_output_ids_unique(processed_documents):
    ids = [document["id"] for document in processed_documents["documents"]]
    assert len(ids) == len(set(ids))


def test_no_empty_content(processed_documents):
    assert all(document["content"].strip() for document in processed_documents["documents"])


def test_documents_json_written(processed_documents):
    output_path: Path = processed_documents["output_path"]
    pii_path: Path = processed_documents["pii_path"]
    assert output_path.exists()
    assert pii_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    pii = json.loads(pii_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert "cnic_hits" in pii
