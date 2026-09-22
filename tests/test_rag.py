import hashlib
import math

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from rag import (
    FinDERRAG,
    FinDERGraphRAG,
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
from evaluate_graph import evaluate_method
from graph_rag import (
    InterpretableDocumentGraph,
    InterpretableGraphRetriever,
    format_retrieval_explanations,
)
from legal_data import legal_records_to_documents, legal_records_to_questions


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


def test_legal_records_keep_passage_ids_without_indexing_answers():
    corpus = [
        {
            "id": "2.1-c1-s1",
            "title": "Views",
            "text": "A court inspection is called a view.",
            "footnotes": "Evidence Act 2008 s 53.",
        }
    ]
    qa = [
        {
            "id": 7,
            "question": "What is the procedure called?",
            "answer": "A view.",
            "relevant_passage_id": "2.1-c1-s1",
        }
    ]

    documents = legal_records_to_documents(corpus)
    questions = legal_records_to_questions(qa)

    assert documents[0].metadata["passage_id"] == "2.1-c1-s1"
    assert "A view." not in documents[0].page_content
    assert questions[0].relevant_passage_id == "2.1-c1-s1"


def test_graph_expansion_explains_structural_path_and_score():
    documents = [
        Document(
            page_content="The court may conduct a view.",
            metadata={"passage_id": "2.1-c1-s1", "title": "Views"},
        ),
        Document(
            page_content="Directions are required during the inspection.",
            metadata={"passage_id": "2.1-c1-s2", "title": "Views"},
        ),
        Document(
            page_content="An unrelated sentencing passage.",
            metadata={"passage_id": "9.9-c1-s1", "title": "Sentencing"},
        ),
    ]
    graph = InterpretableDocumentGraph(documents, max_neighbors=4)

    result = graph.expand_seeds(
        [("2.1-c1-s1", 1.0), ("9.9-c1-s1", 0.0)],
        top_k=2,
        graph_weight=0.5,
    )

    assert [item.metadata["passage_id"] for item in result.documents] == [
        "2.1-c1-s1",
        "2.1-c1-s2",
    ]
    assert result.explanations[1].relation == "sequence"
    assert result.explanations[1].seed_id == "2.1-c1-s1"
    assert result.explanations[1].graph_contribution == 0.5
    assert "2.1-c1-s1 -> 2.1-c1-s2" in format_retrieval_explanations(
        result.explanations
    )


def test_graph_metrics_include_mrr_ndcg_and_assisted_gold():
    traces = [
        InterpretableDocumentGraph(
            [
                Document(page_content="alpha", metadata={"passage_id": "a"}),
                Document(page_content="beta", metadata={"passage_id": "b"}),
            ]
        ).expand_seeds([("a", 1.0), ("b", 0.5)], top_k=2).explanations
    ]
    metrics = evaluate_method(
        method="graph",
        rankings=[["a", "b"]],
        gold_ids=["b"],
        top_k_values=[2],
        corpus_size=2,
        latency_ms_per_query=1.0,
        traces=traces,
    )[0]

    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr_at_k == 0.5
    assert metrics.ndcg_at_k == 1 / math.log2(3)
    assert metrics.explanation_coverage == 1.0


def test_end_to_end_graph_rag_returns_auditable_trace():
    documents = [
        Document(
            page_content="Revenue increased by 20 percent.",
            metadata={
                "chunk_id": "r1:ref1:chunk1",
                "row_id": "r1",
                "reference_number": 1,
                "reference_chunk_number": 1,
            },
        ),
        Document(
            page_content="Cybersecurity is a material risk.",
            metadata={
                "chunk_id": "r1:ref1:chunk2",
                "row_id": "r1",
                "reference_number": 1,
                "reference_chunk_number": 2,
            },
        ),
    ]
    store = InMemoryVectorStore.from_documents(documents, embedding=TinyEmbeddings())
    graph = InterpretableDocumentGraph(documents)
    retriever = InterpretableGraphRetriever(store, graph, top_k=1, seed_k=2)
    fake_model = RunnableLambda(lambda _: "Revenue increased by 20 percent [S1].")
    engine = FinDERGraphRAG(retriever, fake_model)

    response = engine.ask("How did revenue change?")

    assert response.sources[0].metadata["chunk_id"] == "r1:ref1:chunk1"
    assert response.retrieval_trace[0]["node_id"] == "r1:ref1:chunk1"
    assert response.retrieval_trace[0]["evidence"]
