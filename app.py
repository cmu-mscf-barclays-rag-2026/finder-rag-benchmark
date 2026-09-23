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


def show_trace(trace: list[dict]) -> None:
    with st.expander("Why these passages were retrieved"):
        for item in trace:
            if "path" in item:
                st.markdown(f"**S{item['rank']}** - graph score {item['final_score']:.6f}")
                st.code(" -> ".join(item["path"]), language=None)
                st.caption(
                    "One connecting path is shown. The score sums incoming mass from "
                    f"matched concepts ({item['matched_concept_contribution']:.6f}), "
                    f"other concepts ({item['other_concept_contribution']:.6f}), and "
                    f"adjacent passages ({item['structural_contribution']:.6f})."
                )
            else:
                st.markdown(
                    f"**S{item['rank']}** `{item['relation']}` via "
                    f"`{item['seed_id']}` - {', '.join(item['evidence'])}"
                )
                st.caption(
                    f"dense contribution {item['dense_contribution']:.3f} + "
                    f"graph contribution {item['graph_contribution']:.3f} = "
                    f"{item['final_score']:.3f}"
                )


st.title("FinDER Research Chat")
st.caption(
    "Local research chat with independent graph retrieval and cited source passages."
)

with st.sidebar:
    st.header("Local setup")
    retrieval_mode = st.selectbox(
        "Retrieval strategy",
        options=["graph", "dense", "dense_graph"],
        format_func=lambda value: {
            "graph": "Standalone graph (no embeddings)",
            "dense": "Dense MMR baseline",
            "dense_graph": "Dense + graph (previous experiment)",
        }[value],
    )
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
        disabled=retrieval_mode == "graph",
    )
    embedding_device = st.selectbox(
        "Embedding device",
        options=["cpu", "cuda"],
        index=0 if os.getenv("HF_EMBEDDING_DEVICE", "cpu") == "cpu" else 1,
        help="Choose CUDA only if PyTorch can use your NVIDIA GPU.",
        disabled=retrieval_mode == "graph",
    )
    sample_size = st.slider("Dataset rows", 50, 1_000, 300, step=50)
    top_k = st.slider("Retrieved passages", 2, 8, 4)
    graph_seed_k = max(top_k, 10)
    if retrieval_mode == "dense_graph":
        graph_seed_k = st.slider(
            "Dense graph seeds", top_k, 20, graph_seed_k,
            help="Only the previous dense + graph experiment uses dense seeds.",
        )

    st.divider()
    st.markdown(f"[View the FinDER dataset]({DATASET_URL})")
    st.caption(
        "First use downloads FinDER. Standalone graph retrieval needs no embedding "
        "model; answers are generated locally with Ollama."
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
    retrieval_mode=retrieval_mode,
    graph_seed_k=graph_seed_k,
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
        if message.get("retrieval_trace"):
            show_trace(message["retrieval_trace"])

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
                if settings.retrieval_mode in {"graph", "dense_graph"}:
                    status.write(
                        f"Built {stats['graph_edges']:,} interpretable edges across "
                        f"{stats['graph_nodes']:,} passages and "
                        f"{stats['graph_concepts']:,} concepts."
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
            if response.retrieval_trace:
                show_trace(response.retrieval_trace)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": serialised_sources,
                    "retrieval_trace": response.retrieval_trace,
                }
            )
        except Exception as exc:
            st.error(f"Could not complete the request: {exc}")
