import math

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evaluate_standalone_graph import ranking_metrics
from standalone_graph import StandaloneGraphRetriever, StandaloneGraphSettings


def fixture_documents():
    return [
        Document(page_content="alpha bridge", metadata={"passage_id": "a"}),
        Document(page_content="bridge evidence", metadata={"passage_id": "b"}),
        Document(page_content="unrelated island", metadata={"passage_id": "c"}),
    ]


def test_graph_reaches_unmatched_passage_via_shared_concept():
    graph = StandaloneGraphRetriever(fixture_documents(), top_k=3)
    direct = graph.retrieve("alpha", propagate=False)
    result = graph.retrieve("alpha")
    assert [item.node_id for item in direct.explanations] == ["a"]
    assert {item.node_id for item in result.explanations} == {"a", "b"}
    indirect = next(item for item in result.explanations if item.node_id == "b")
    assert indirect.path == ("term:alpha", "passage:a", "term:bridge", "passage:b")
    assert not indirect.matched_concepts
    assert indirect.other_concept_contribution > 0
    assert result.converged
    assert result.residual <= graph.settings.tolerance
    for item in result.explanations:
        assert item.final_score == pytest.approx(
            item.matched_concept_contribution + item.other_concept_contribution + item.structural_contribution
        )
        assert item.final_score == pytest.approx(
            sum(row["contribution"] for row in item.top_incoming) + item.remaining_incoming_contribution
        )


def test_graph_unknown_query_abstains_and_transition_preserves_mass():
    graph = StandaloneGraphRetriever(fixture_documents())
    assert graph.retrieve("zyxwunknown").documents == []
    assert graph.retrieve("the and of").documents == []
    assert np.allclose(np.asarray(graph.transition.sum(axis=1)).ravel(), 1)
    with pytest.raises(ValueError):
        graph.retrieve(" ")


def test_graph_order_and_benchmark_metadata_do_not_change_rankings():
    documents = fixture_documents()
    clean = StandaloneGraphRetriever(documents).retrieve("alpha")
    for document in documents:
        document.metadata.update(benchmark_question="secret_marker", answer="secret_marker", relevant_passage_id="b")
    changed = StandaloneGraphRetriever(list(reversed(documents)))
    assert "term:secret" not in changed.concept_index
    assert changed.retrieve("alpha").explanations == clean.explanations


def test_structural_traversal_and_iteration_limit_are_exposed():
    documents = [
        Document(page_content="alpha", metadata={"passage_id": "1.2-c1-s1"}),
        Document(page_content="beta", metadata={"passage_id": "1.2-c1-s2"}),
    ]
    graph = StandaloneGraphRetriever(documents)
    result = graph.retrieve("alpha")
    neighbor = next(item for item in result.explanations if item.node_id.endswith("s2"))
    assert neighbor.structural_contribution > 0
    assert "consecutive_passage" in neighbor.path_relations
    limited = StandaloneGraphRetriever(documents, settings=StandaloneGraphSettings(max_iterations=1))
    assert not limited.retrieve("alpha").converged


def test_graph_build_and_generation_never_construct_embeddings(monkeypatch):
    import rag

    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone retrieval attempted to create embeddings or a vector store")

    monkeypatch.setattr(rag, "HuggingFaceEmbeddings", forbidden)
    monkeypatch.setattr(rag.InMemoryVectorStore, "from_documents", forbidden)
    monkeypatch.setattr(rag, "check_ollama", lambda *args: rag.OllamaStatus(True, "test"))
    monkeypatch.setattr(rag, "ChatOllama", lambda **kwargs: RunnableLambda(lambda _: "Revenue rose [S1]."))
    engine, stats = rag.build_rag(
        rag.RAGSettings(retrieval_mode="graph", top_k=1),
        records=[{"_id": "a", "references": ["Revenue rose."]}],
    )
    result = engine.ask("revenue")
    assert result.answer == "Revenue rose [S1]."
    assert result.sources
    assert result.retrieval_trace[0]["path"][0] == "term:revenue"
    assert stats["graph_nodes"] == 1
    # Unknown concepts must not trigger ungrounded generation.
    engine.chain = RunnableLambda(forbidden)
    assert not engine.ask("zyxwunknown").sources


def test_single_qrel_metrics():
    values = ranking_metrics(["a", "b"], "b", 5)
    assert values["precision_at_k"] == 0.2
    assert values["recall_at_k"] == 1
    assert values["mrr_at_k"] == 0.5
    assert values["ndcg_at_k"] == 1 / math.log2(3)
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], "a", 5)
