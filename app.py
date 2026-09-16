"""Streamlit UI for the fully local FinDER LangChain RAG baseline."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag import DATASET_URL, RAGSettings, build_rag, check_ollama


load_dotenv()

st.set_page_config(
    page_title="FinDER Research Chat",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_rag(settings: RAGSettings):
    return build_rag(settings=settings)


@st.cache_data(ttl=5, show_spinner=False)
def cached_ollama_status(base_url: str, model: str):
    return check_ollama(base_url, model)


def source_caption(metadata: dict) -> str:
    return (
        f"FinDER row {metadata.get('row_id', 'unknown')} · "
        f"{metadata.get('category', 'Uncategorized')} · "
        f"reference {metadata.get('reference_number', '?')}"
    )


st.title("FinDER Research Chat")
st.caption(
    "A free, local LangChain RAG chatbot using Hugging Face embeddings and Ollama."
)

with st.sidebar:
    st.header("Local setup")
    chat_model = st.text_input(
        "Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    ollama_base_url = st.text_input(
        "Ollama address",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    embedding_model = st.text_input(
        "Hugging Face embedding model",
        value=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    embedding_device = st.selectbox(
        "Embedding device",
        options=["cpu", "cuda"],
        index=0 if os.getenv("HF_EMBEDDING_DEVICE", "cpu") == "cpu" else 1,
        help="Choose CUDA only if PyTorch can use your NVIDIA GPU.",
    )
    sample_size = st.slider("Dataset rows", 50, 1_000, 300, step=50)
    top_k = st.slider("Retrieved passages", 2, 8, 4)

    st.divider()
    st.markdown(f"[View the FinDER dataset]({DATASET_URL})")
    st.caption(
        "First use downloads FinDER and the embedding model. After those downloads, "
        "retrieval and generation run locally."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

settings = RAGSettings(
    sample_size=sample_size,
    top_k=top_k,
    chat_model=chat_model.strip() or "llama3.2",
    ollama_base_url=ollama_base_url.strip() or "http://localhost:11434",
    embedding_model=(
        embedding_model.strip() or "sentence-transformers/all-MiniLM-L6-v2"
    ),
    embedding_device=embedding_device,
)

ollama_status = cached_ollama_status(settings.ollama_base_url, settings.chat_model)
if ollama_status.ready:
    st.sidebar.success(ollama_status.message)
else:
    st.sidebar.warning(ollama_status.message)
    st.info(
        "Ollama must be running with the selected model before chatting. "
        f"Install Ollama, open it, run `ollama pull {settings.chat_model}`, then reload."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**S{index} — {source['caption']}**")
                    st.caption(source["text"])

question = st.chat_input(
    "Ask about a company, metric, filing, risk, or calculation...",
    disabled=not ollama_status.ready,
)

if question:
    prior_history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status("Preparing the retrieval index...", expanded=True) as status:
                engine, stats = cached_rag(settings)
                status.write(
                    f"Indexed {stats['chunks']:,} chunks from "
                    f"{stats['references']:,} references."
                )
                status.update(label="Searching and drafting an answer...", state="running")
                response = engine.ask(question, history=prior_history)
                status.update(label="Answer ready", state="complete", expanded=False)

            st.markdown(response.answer)
            serialised_sources = []
            with st.expander("Retrieved sources"):
                for index, source in enumerate(response.sources, start=1):
                    caption = source_caption(source.metadata)
                    st.markdown(f"**S{index} — {caption}**")
                    st.caption(source.page_content)
                    serialised_sources.append(
                        {"caption": caption, "text": source.page_content}
                    )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": serialised_sources,
                }
            )
        except Exception as exc:
            st.error(f"Could not complete the request: {exc}")
