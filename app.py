from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from scripts.build_vectordb import get_embedding_model, load_vectorstore
from scripts.document_manager import (
    add_single_document,
    delete_by_source,
    delete_document,
    get_stats,
    ingest_file,
    list_documents,
    list_sources,
)
from scripts.rag_pipeline import OLLAMA_MODEL, ask

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="NUST Bank Assistant", page_icon="🏦", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_resources() -> tuple[Any, Any]:
    """Load the vectorstore and embedding model once per app session."""
    collection = load_vectorstore()
    model = get_embedding_model()
    return collection, model


collection, embedding_model = get_resources()


def _source_tag_from_name(filename: str) -> str:
    """Derive a stable source tag from an uploaded filename."""
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return f"faq::{stem}"


# ---------------------------------------------------------------------------
# Callbacks — run BEFORE the rerun, so state is updated before rendering
# ---------------------------------------------------------------------------


def _cb_delete_document(pid: str) -> None:
    """on_click callback: delete a document and track it."""
    count = delete_document(pid, collection)
    st.session_state.setdefault("_deleted_pids", set()).add(pid)
    st.session_state["_notify"] = f"✅ Deleted chunk `{pid}` ({count} chunk(s) removed)"


def _cb_delete_source(source: str) -> None:
    """on_click callback: delete all chunks for a source."""
    count = delete_by_source(source, collection)
    st.session_state["_notify"] = f"✅ Deleted {count} chunks from `{source}`"


# ---------------------------------------------------------------------------
# Sidebar: Knowledge Management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📚 Knowledge Management")

    # ── Floating notification banner ──────────────────────────────────────
    # ── Section picker (persists across reruns via key) ──────────────────
    section = st.selectbox(
        "Section",
        ["📊 Stats", "📤 Upload File", "➕ Add FAQ", "📋 Browse Sources", "🗑️ Delete Source"],
        key="_sidebar_section",
        label_visibility="collapsed",
    )

    # ── Notification banner ────────────────────────────────────────────────
    if "_notify" in st.session_state:
        msg = st.session_state.pop("_notify")
        if "✅" in msg:
            st.success(msg)
        elif "❌" in msg:
            st.error(msg)
        else:
            st.info(msg)

    # ── Stats ──────────────────────────────────────────────────────────────
    if section == "📊 Stats":
        stats = get_stats(collection)
        st.metric("Total Chunks", stats["total_chunks"])
        st.metric("Sources", stats["source_count"])
        size_kb = stats["disk_size_bytes"] / 1024
        st.metric("Disk Size", f"{size_kb:.1f} KB")

    # ── Upload File ────────────────────────────────────────────────────────
    elif section == "📤 Upload File":
        st.caption(
            "Upload a `.json` file — FAQ format or documents format."
        )
        upload_counter = st.session_state.get("_upload_counter", 0)
        uploaded = st.file_uploader(
            "Choose a JSON file",
            type=["json"],
            key=f"file_uploader_{upload_counter}",
        )
        if uploaded is not None:
            if st.button("Ingest File", key="btn_ingest", type="primary"):
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".json", mode="wb"
                    ) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name

                    source = _source_tag_from_name(uploaded.name)
                    result = ingest_file(
                        tmp_path, collection, embedding_model,
                        source_tag=source,
                    )
                    Path(tmp_path).unlink(missing_ok=True)
                    st.session_state["_notify"] = (
                        f"✅ Ingested {result['added']} chunks "
                        f"(replaced {result['deleted']} old) "
                        f"— source: {result['source']}"
                    )
                    st.session_state["_upload_counter"] = upload_counter + 1
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")

    # ── Add Single FAQ ─────────────────────────────────────────────────────
    elif section == "➕ Add FAQ":
        with st.form("add_faq_form", clear_on_submit=True):
            faq_category = st.text_input("Category", placeholder="e.g. Funds Transfer")
            faq_product = st.text_input("Product (optional)", placeholder="e.g. Mobile App")
            faq_question = st.text_area("Question", placeholder="Enter the question")
            faq_answer = st.text_area("Answer", placeholder="Enter the answer")
            submitted = st.form_submit_button("Add FAQ", type="primary")

        if submitted:
            if not faq_question.strip() or not faq_answer.strip():
                st.error("❌ Question and Answer are required.")
            else:
                q_hash = hashlib.md5(faq_question.strip().encode()).hexdigest()[:8]
                doc = {
                    "id": f"MANUAL_{q_hash}",
                    "product": faq_product.strip() or faq_category.strip() or "General",
                    "sheet": "manual_entry",
                    "type": "qa_pair",
                    "category": faq_category.strip() or "faq",
                    "question": faq_question.strip(),
                    "content": f"Q: {faq_question.strip()}\nA: {faq_answer.strip()}",
                }
                try:
                    result = add_single_document(
                        doc, "manual::single_faq", collection, embedding_model
                    )
                    st.session_state["_notify"] = (
                        f"✅ Added FAQ ({result['added']} chunk(s)). ID: {doc['id']}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")

    # ── Browse Sources ─────────────────────────────────────────────────────
    elif section == "📋 Browse Sources":
        deleted_pids: set = st.session_state.setdefault("_deleted_pids", set())
        sources = list_sources(collection)

        if not sources:
            st.info("No sources found.")
        else:
            for src_info in sources:
                src_name = src_info["source"]
                # Stable label — no chunk count so it doesn't change on delete
                with st.expander(f"📁 {src_name}", expanded=False):
                    chunk_count = src_info["chunk_count"]
                    ts = src_info.get("latest_ingested_at", "")[:19]
                    st.caption(f"{chunk_count} chunks · last updated {ts}" if ts else f"{chunk_count} chunks")

                    docs = list_documents(collection, source=src_name)
                    docs = [d for d in docs if d["parent_id"] not in deleted_pids]

                    if not docs:
                        st.caption("No documents in this source.")
                        continue

                    for doc_info in docs:
                        pid = doc_info["parent_id"]
                        q = doc_info.get("question") or ""
                        preview = doc_info.get("content_preview", "")[:100]
                        product = doc_info.get("product", "")

                        display = f"**{pid}**"
                        if product:
                            display += f" · {product}"
                        if q:
                            display += f"  \n> {q}"
                        elif preview:
                            display += f"  \n> {preview}..."

                        cols = st.columns([5, 1])
                        with cols[0]:
                            st.markdown(display)
                        with cols[1]:
                            # on_click runs BEFORE rerun — item will be
                            # in deleted_pids by the time we render again,
                            # so it disappears without closing the expander
                            st.button(
                                "🗑️",
                                key=f"del_{pid}",
                                on_click=_cb_delete_document,
                                args=(pid,),
                            )

    # ── Delete Source ───────────────────────────────────────────────────────
    elif section == "🗑️ Delete Source":
        sources = list_sources(collection)
        source_names = [s["source"] for s in sources]
        if not source_names:
            st.info("No sources to delete.")
        else:
            selected_source = st.selectbox(
                "Select source to delete",
                source_names,
                key="delete_source_select",
            )
            st.warning(f"This will delete **all** data from `{selected_source}`.")
            st.button(
                "Delete Source",
                key="btn_delete_source",
                type="primary",
                on_click=_cb_delete_source,
                args=(selected_source,),
            )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("NUST Bank Assistant")
st.caption(f"Powered by {OLLAMA_MODEL} via Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about NUST Bank products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = ask(prompt, collection=collection, model=embedding_model, stream=True)

        def _token_gen():
            for chunk in stream:
                yield chunk["message"]["content"]

        response: str = st.write_stream(_token_gen())

    st.session_state.messages.append({"role": "assistant", "content": response})
