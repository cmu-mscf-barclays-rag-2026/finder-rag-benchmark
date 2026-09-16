import hashlib

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from rag import (
    FinDERRAG,
    RAGSettings,
    format_context,
    format_history,
    records_to_documents,
    split_documents,
)
from evaluate_dense import (
    ChunkingConfig,
    build_source_corpus,
    deterministic_split,
    evaluate_rankings,
    query_metrics,
    rank_sources,
)


def test_default_settings_are_local_and_need_no_api_key():
    settings = RAGSettings()

    assert settings.chat_model == "llama3.2"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


class TinyEmbeddings(Embeddings):
    """Deterministic embeddings for an offline retrieval smoke test."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float("revenue" in lowered), float("risk" in lowered), 1.0]


def test_records_become_reference_documents_without_answers():
    records = [
        {
            "_id": "abc12345",
            "text": "What changed?",
            "category": "Financials",
            "references": ["Revenue increased from 10 to 12.", "Costs stayed flat."],
            "answer": "Revenue increased by 2.",
        }
    ]

    documents = records_to_documents(records)

    assert len(documents) == 2
    assert documents[0].metadata["row_id"] == "abc12345"
    assert documents[0].metadata["benchmark_question"] == "What changed?"
    assert all("Revenue increased by 2" not in doc.page_content for doc in documents)


def test_duplicate_reference_in_same_row_is_removed():
    records = [
        {
            "_id": "duplicate",
            "references": ["Same passage", "Same passage"],
        }
    ]

    assert len(records_to_documents(records)) == 1


def test_split_and_context_labels():
    documents = [
        Document(
            page_content="A financial sentence. " * 30,
            metadata={"row_id": "row1", "category": "Accounting"},
        )
    ]

    chunks = split_documents(documents, chunk_size=120, chunk_overlap=20)
    context = format_context(chunks[:2])

    assert len(chunks) > 1
    assert "[S1] FinDER row row1" in context
    assert "[S2] FinDER row row1" in context


def test_history_is_bounded():
    history = [{"role": "user", "content": str(index)} for index in range(10)]
    rendered = format_history(history, max_messages=3)

    assert "User: 7" in rendered
    assert "User: 9" in rendered
    assert "User: 6" not in rendered


def test_end_to_end_retrieval_without_network():
    documents = [
        Document(page_content="Revenue increased by 20 percent.", metadata={"row_id": "r1"}),
        Document(page_content="Cybersecurity is a material risk.", metadata={"row_id": "r2"}),
    ]
    store = InMemoryVectorStore.from_documents(documents, embedding=TinyEmbeddings())
    fake_model = RunnableLambda(lambda _: "Revenue increased by 20 percent [S1].")
    rag = FinDERRAG(store, fake_model, top_k=1)

    response = rag.ask("How did revenue change?")

    assert response.sources[0].metadata["row_id"] == "r1"
    assert "[S1]" in response.answer


def test_split_adds_stable_evaluation_identifiers():
    documents = [
        Document(
            page_content="Revenue increased. " * 30,
            metadata={"row_id": "r1", "reference_number": 2},
        )
    ]

    chunks = split_documents(documents, chunk_size=120, chunk_overlap=20)

    assert chunks[0].metadata["chunk_id"] == "r1:ref2:chunk1"
    assert chunks[1].metadata["chunk_id"] == "r1:ref2:chunk2"


def test_source_corpus_is_deduplicated_without_answers():
    records = [
        {"_id": "a", "text": "qa", "references": ["Same   passage"], "answer": "secret"},
        {"_id": "b", "text": "qb", "references": ["Same passage", "Other evidence"]},
    ]

    corpus, qrels = build_source_corpus(records)

    assert corpus == ["Same passage", "Other evidence"]
    assert qrels == {"a": {0}, "b": {0, 1}}
    assert all("secret" not in passage for passage in corpus)


def test_precision_recall_uses_exact_source_passages_and_fixed_k():
    precision, recall, hit = query_metrics([7, 2, 9], relevant={2, 9, 11, 12}, k=3)

    assert precision == 2 / 3
    assert recall == 2 / 4
    assert hit == 1.0


def test_evaluate_rankings_macro_averages_queries():
    rankings = np.asarray([[0, 2], [2, 0]])
    metrics = evaluate_rankings(
        rankings=rankings,
        qrels=[{0, 1}, {1}],
        top_k_values=[1],
        config=ChunkingConfig("test", 100, 0),
        split="dev",
        source_passages=3,
        chunks=4,
        index_seconds=0.1,
        latency_ms_per_query=2.0,
    )[0]

    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 0.25
    assert metrics.hit_rate_at_k == 0.5
    assert metrics.latency_ms_per_query == 2.0


def test_deterministic_split_matches_sha1_contract():
    row_ids = ["alpha", "beta", "gamma"]
    expected = [
        "dev" if int(hashlib.sha1(value.encode()).hexdigest()[:8], 16) % 5 == 0 else "test"
        for value in row_ids
    ]

    assert deterministic_split(row_ids).tolist() == expected


def test_chunk_scores_are_max_pooled_to_unique_sources():
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    chunks = np.asarray([[0.7, 0.0], [0.9, 0.0], [0.8, 0.0]], dtype=np.float32)
    source_ids = np.asarray([0, 0, 1])

    rankings, _ = rank_sources(queries, chunks, source_ids, source_count=2, depth=2)

    assert rankings.tolist() == [[0, 1]]
