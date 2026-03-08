from __future__ import annotations

from typing import Any

import streamlit as st

from scripts.build_vectordb import get_embedding_model, load_vectorstore
from scripts.rag_pipeline import OLLAMA_MODEL, ask

st.set_page_config(page_title="NUST Bank Assistant", page_icon="🏦", layout="centered")
st.title("NUST Bank Assistant")
st.caption(f"Powered by {OLLAMA_MODEL} via Ollama")


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_resources() -> tuple[Any, Any]:
    """Load the vectorstore and embedding model once per app session."""
    collection = load_vectorstore()
    model = get_embedding_model()
    return collection, model


collection, embedding_model = get_resources()

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
