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
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return f"faq::{stem}"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _cb_delete_document(pid: str) -> None:
    count = delete_document(pid, collection)
    st.session_state.setdefault("_deleted_pids", set()).add(pid)
    st.session_state["_notify"] = f"Removed **{pid}** ({count} chunk(s))"


def _cb_delete_source(source: str) -> None:
    count = delete_by_source(source, collection)
    st.session_state["_notify"] = f"Removed **{source}** ({count} chunks)"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Knowledge Base")

    section = st.pills(
        "nav",
        [
            ":material/dashboard: Overview",
            ":material/upload_file: Upload",
            ":material/post_add: Add",
            ":material/folder_open: Browse",
            ":material/delete_sweep: Remove",
        ],
        default=":material/dashboard: Overview",
        key="_nav",
        label_visibility="collapsed",
    )

    # Notification banner
    if "_notify" in st.session_state:
        st.success(st.session_state.pop("_notify"), icon=":material/check_circle:")

    st.divider()

    # ── Overview ──────────────────────────────────────────────────────────
    if section == ":material/dashboard: Overview":
        stats = get_stats(collection)
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats["total_chunks"])
        c2.metric("Sources", stats["source_count"])
        size_kb = stats["disk_size_bytes"] / 1024
        st.caption(f":material/storage: {size_kb:.0f} KB on disk")
        st.divider()
        if stats["sources"]:
            st.markdown("##### Sources")
            for src in stats["sources"]:
                ts = src.get("latest_ingested_at", "")[:10]
                meta = f"{src['chunk_count']} chunks"
                if ts:
                    meta += f" · {ts}"
                st.markdown(f":material/database: **{src['source']}**  \n{meta}")
        else:
            st.caption("No sources ingested yet.")

    # ── Upload ────────────────────────────────────────────────────────────
    elif section == ":material/upload_file: Upload":
        st.caption(":material/info: Accepts FAQ format or documents format (.json)")
        upload_counter = st.session_state.get("_upload_counter", 0)
        uploaded = st.file_uploader(
            "File",
            type=["json"],
            key=f"file_uploader_{upload_counter}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            if st.button("Ingest", key="btn_ingest", type="primary", use_container_width=True):
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
                        f"Ingested **{result['added']}** chunks "
                        f"(replaced {result['deleted']} old) "
                        f"from `{result['source']}`"
                    )
                    st.session_state["_upload_counter"] = upload_counter + 1
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    # ── Add entry ─────────────────────────────────────────────────────────
    elif section == ":material/post_add: Add":
        st.caption(":material/edit_note: Add a single Q&A entry to the knowledge base")
        with st.form("add_faq_form", clear_on_submit=True):
            faq_category = st.text_input("Category", placeholder="e.g. Funds Transfer")
            faq_product = st.text_input("Product *(optional)*", placeholder="e.g. Mobile App")
            faq_question = st.text_area("Question")
            faq_answer = st.text_area("Answer")
            submitted = st.form_submit_button("Add entry", type="primary", use_container_width=True)

        if submitted:
            if not faq_question.strip() or not faq_answer.strip():
                st.warning("Both question and answer are required.")
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
                        f"Added entry **{doc['id']}** ({result['added']} chunk(s))"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    # ── Browse ────────────────────────────────────────────────────────────
    elif section == ":material/folder_open: Browse":
        deleted_pids: set = st.session_state.setdefault("_deleted_pids", set())
        sources = list_sources(collection)

        if not sources:
            st.info("No sources in the knowledge base yet.")
        else:
            for src_info in sources:
                src_name = src_info["source"]
                with st.expander(src_name, expanded=False):
                    chunk_count = src_info["chunk_count"]
                    ts = src_info.get("latest_ingested_at", "")[:19]
                    st.caption(
                        f"{chunk_count} chunks · {ts}" if ts else f"{chunk_count} chunks"
                    )

                    docs = list_documents(collection, source=src_name)
                    docs = [d for d in docs if d["parent_id"] not in deleted_pids]

                    if not docs:
                        st.caption("No documents.")
                        continue

                    for doc_info in docs:
                        pid = doc_info["parent_id"]
                        q = doc_info.get("question") or ""
                        preview = doc_info.get("content_preview", "")[:100]
                        product = doc_info.get("product", "")

                        label = f"**{pid}**"
                        if product:
                            label += f" · {product}"
                        if q:
                            label += f"  \n{q}"
                        elif preview:
                            label += f"  \n{preview}..."

                        cols = st.columns([5, 1])
                        with cols[0]:
                            st.markdown(label)
                        with cols[1]:
                            st.button(
                                "×",
                                key=f"del_{pid}",
                                on_click=_cb_delete_document,
                                args=(pid,),
                                help=f"Remove {pid}",
                            )

    # ── Remove source ─────────────────────────────────────────────────────
    elif section == ":material/delete_sweep: Remove":
        sources = list_sources(collection)
        source_names = [s["source"] for s in sources]
        if not source_names:
            st.info("No sources available.")
        else:
            selected_source = st.selectbox(
                "Source",
                source_names,
                key="delete_source_select",
                label_visibility="collapsed",
            )
            st.caption(f"All data from **{selected_source}** will be permanently removed.")
            st.button(
                "Remove source",
                key="btn_delete_source",
                type="primary",
                use_container_width=True,
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
