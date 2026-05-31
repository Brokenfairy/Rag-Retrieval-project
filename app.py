from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import get_settings
from src.ingestion import load_documents, split_documents
from src.models import get_chat_model, get_embedding_model
from src.rag import build_rag_chain
from src.vectorstore import get_retriever, has_collection, upsert_documents


st.set_page_config(page_title="LangChain RAG Starter", layout="wide")
st.title("LangChain RAG Starter")
st.caption("Upload files, index to Milvus/Zilliz, then chat with your documents via Ollama.")

upload_dir = Path("data/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

base_settings = get_settings()

with st.sidebar:
    st.header("Configuration")
    collection_name = st.text_input("Collection name", value=base_settings.collection_name)
    top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=4)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=150, step=10)
    st.write("Vector DB URI:", f"`{base_settings.zilliz_uri}`")
    st.write("Ollama model:", f"`{base_settings.ollama_model}`")

settings = get_settings(collection_name=collection_name)
embeddings = get_embedding_model(settings)
llm = get_chat_model(settings)

st.subheader("1) Upload and index")
uploaded_files = st.file_uploader(
    "Select documents",
    accept_multiple_files=True,
    type=["txt", "md", "pdf", "csv", "msg"],
)

if st.button("Index uploaded files", type="primary", disabled=not uploaded_files):
    saved_paths: list[Path] = []
    for uploaded in uploaded_files:
        destination = upload_dir / uploaded.name
        destination.write_bytes(uploaded.getbuffer())
        saved_paths.append(destination)

    with st.spinner("Loading and chunking documents..."):
        raw_docs = load_documents(saved_paths, settings)
        chunks = split_documents(raw_docs, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))

    with st.spinner("Storing embeddings in Milvus/Zilliz..."):
        inserted = upsert_documents(settings, embeddings, chunks)

    st.success(f"Indexed {inserted} chunks from {len(saved_paths)} file(s).")

st.subheader("2) Chat with your data")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your indexed documents")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if not has_collection(settings):
            raise ValueError("No collection found for this name. Index documents first.")

        retriever = get_retriever(settings, embeddings, k=int(top_k))
        rag_chain = build_rag_chain(retriever, llm)
        answer = rag_chain.invoke(prompt)
    except Exception as exc:
        answer = f"Error: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

