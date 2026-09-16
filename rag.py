"""Shared, fully local LangChain RAG pipeline for the FinDER dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATASET_ID = "Linq-AI-Research/FinDER"
DATASET_URL = "https://huggingface.co/datasets/Linq-AI-Research/FinDER"


@dataclass(frozen=True)
class RAGSettings:
    """Tunable settings for a small, understandable baseline."""

    sample_size: int = 300
    seed: int = 42
    chunk_size: int = 1_000
    chunk_overlap: int = 150
    top_k: int = 4
    chat_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.chat_model.strip():
            raise ValueError("chat_model cannot be empty")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        if self.embedding_device not in {"cpu", "cuda", "mps"}:
            raise ValueError("embedding_device must be cpu, cuda, or mps")


@dataclass
class RAGResponse:
    answer: str
    sources: list[Document]


@dataclass(frozen=True)
class OllamaStatus:
    ready: bool
    message: str
    installed_models: tuple[str, ...] = ()


def check_ollama(base_url: str, model: str, timeout: float = 3.0) -> OllamaStatus:
    """Check that the local Ollama server is reachable and has the model."""

    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urlopen(tags_url, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return OllamaStatus(
            ready=False,
            message=(
                "Ollama is not reachable. Install and open Ollama, then run "
                f"`ollama pull {model}`. Details: {exc}"
            ),
        )

    installed = tuple(
        sorted(
            {
                str(item.get("name") or item.get("model"))
                for item in payload.get("models", [])
                if item.get("name") or item.get("model")
            }
        )
    )
    requested = model if ":" in model else f"{model}:latest"
    if requested not in installed:
        return OllamaStatus(
            ready=False,
            message=f"Model `{model}` is not installed. Run `ollama pull {model}`.",
            installed_models=installed,
        )
    return OllamaStatus(
        ready=True,
        message=f"Ollama is ready with `{model}`.",
        installed_models=installed,
    )


def load_finder_records(sample_size: int = 300, seed: int = 42) -> list[dict[str, Any]]:
    """Download FinDER and return a deterministic, shuffled sample."""

    dataset = load_dataset(DATASET_ID, split="train")
    count = min(sample_size, len(dataset))
    return dataset.shuffle(seed=seed).select(range(count)).to_list()


def _reference_texts(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return []
    return (item.strip() for item in value if isinstance(item, str) and item.strip())


def records_to_documents(records: Sequence[dict[str, Any]]) -> list[Document]:
    """Convert each FinDER reference passage into a LangChain document."""

    documents: list[Document] = []
    seen: set[tuple[str, str]] = set()

    for row in records:
        row_id = str(row.get("_id", "unknown"))
        for reference_number, passage in enumerate(
            _reference_texts(row.get("references", [])), start=1
        ):
            identity = (row_id, passage)
            if identity in seen:
                continue
            seen.add(identity)
            documents.append(
                Document(
                    page_content=passage,
                    metadata={
                        "row_id": row_id,
                        "category": str(row.get("category") or "Uncategorized"),
                        "benchmark_question": str(row.get("text") or ""),
                        "reference_number": reference_number,
                        "source": DATASET_URL,
                    },
                )
            )
    return documents


def split_documents(
    documents: Sequence[Document], chunk_size: int = 1_000, chunk_overlap: int = 150
) -> list[Document]:
    """Split references and attach stable identifiers used by retrieval evaluation."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    reference_chunk_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    for chunk_number, chunk in enumerate(chunks, start=1):
        row_id = str(chunk.metadata.get("row_id", "unknown"))
        reference_number = int(chunk.metadata.get("reference_number", 1))
        reference_key = (row_id, reference_number)
        reference_chunk_counts[reference_key] += 1
        within_reference = reference_chunk_counts[reference_key]
        chunk.metadata["chunk_number"] = chunk_number
        chunk.metadata["reference_chunk_number"] = within_reference
        chunk.metadata["chunk_id"] = (
            f"{row_id}:ref{reference_number}:chunk{within_reference}"
        )
    return chunks


def format_context(documents: Sequence[Document]) -> str:
    blocks = []
    for source_number, document in enumerate(documents, start=1):
        metadata = document.metadata
        header = (
            f"[S{source_number}] FinDER row {metadata.get('row_id', 'unknown')} | "
            f"category: {metadata.get('category', 'Uncategorized')}"
        )
        blocks.append(f"{header}\n{document.page_content}")
    return "\n\n".join(blocks)


def format_history(messages: Sequence[dict[str, str]], max_messages: int = 6) -> str:
    recent = messages[-max_messages:]
    if not recent:
        return "No earlier conversation."
    return "\n".join(
        f"{message.get('role', 'user').title()}: {message.get('content', '')}"
        for message in recent
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful financial research assistant. Answer only from the "
            "retrieved FinDER passages. Cite factual claims with [S1], [S2], and so on. "
            "Show arithmetic when the question requires a calculation. If the passages "
            "do not support an answer, say that the indexed sample does not contain "
            "enough evidence. Do not treat the earlier conversation as evidence.",
        ),
        (
            "human",
            "Earlier conversation:\n{history}\n\n"
            "Retrieved passages:\n{context}\n\n"
            "Question: {question}",
        ),
    ]
)


class FinDERRAG:
    """Small retrieval-and-generation facade used by Streamlit and the notebook."""

    def __init__(
        self, vector_store: InMemoryVectorStore, llm: Runnable, top_k: int = 4
    ) -> None:
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(12, top_k * 3), "lambda_mult": 0.7},
        )
        self.chain = PROMPT | llm | StrOutputParser()

    def ask(
        self, question: str, history: Sequence[dict[str, str]] | None = None
    ) -> RAGResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        sources = self.retriever.invoke(question)
        answer = self.chain.invoke(
            {
                "question": question,
                "history": format_history(history or []),
                "context": format_context(sources),
            }
        )
        return RAGResponse(answer=answer, sources=sources)


def build_rag(
    settings: RAGSettings | None = None,
    records: Sequence[dict[str, Any]] | None = None,
) -> tuple[FinDERRAG, dict[str, int]]:
    """Build the local vector index and Ollama generation chain."""

    settings = settings or RAGSettings()
    ollama_status = check_ollama(settings.ollama_base_url, settings.chat_model)
    if not ollama_status.ready:
        raise RuntimeError(ollama_status.message)

    loaded_records = list(records) if records is not None else load_finder_records(
        settings.sample_size, settings.seed
    )
    documents = records_to_documents(loaded_records)
    if not documents:
        raise ValueError("No reference passages were found in the selected records.")
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
    llm = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    engine = FinDERRAG(vector_store=vector_store, llm=llm, top_k=settings.top_k)
    stats = {
        "records": len(loaded_records),
        "references": len(documents),
        "chunks": len(chunks),
    }
    return engine, stats
