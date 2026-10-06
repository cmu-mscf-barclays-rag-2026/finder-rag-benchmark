import math
from dataclasses import replace

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from legal_protocol import ranking_metrics
from standalone_graph import (
    LEGACY_SETTINGS, CorpusWordForms, StandaloneGraphRetriever, StandaloneGraphSettings,
)


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
    legacy = StandaloneGraphRetriever(documents, settings=LEGACY_SETTINGS).retrieve("alpha")
    neighbor = next(item for item in legacy.explanations if item.node_id.endswith("s2"))
    assert neighbor.structural_contribution > 0
    assert "consecutive_passage" in neighbor.path_relations
    tree = StandaloneGraphRetriever(documents).retrieve("alpha")
    neighbor = next(item for item in tree.explanations if item.node_id.endswith("s2"))
    assert neighbor.structural_contribution > 0
    assert neighbor.path == ("term:alpha", "passage:1.2-c1-s1", "section:1.2", "passage:1.2-c1-s2")
    assert neighbor.path_relations[1:] == ("under_heading", "under_heading")
    limited = StandaloneGraphRetriever(documents, settings=StandaloneGraphSettings(max_iterations=1))
    assert not limited.retrieve("alpha").converged


def heading_documents():
    return [
        Document(page_content="# 1.5 Decide Solely on the Evidence\n\nJurors decide on evidence.",
                 metadata={"passage_id": "1.5-c1-s1"}),
        Document(page_content="## Pre-trial Publicity", metadata={"passage_id": "1.5-c2-s1"}),
        Document(page_content="Warn the jury about media reports.", metadata={"passage_id": "1.5-c2-s2"}),
        Document(page_content="Ignore reports.\n\n## Jury Room Experiments\n\nNo tests.",
                 metadata={"passage_id": "1.5-c2-s3"}),
    ]


def test_passages_inherit_words_of_the_headings_they_sit_under():
    graph = StandaloneGraphRetriever(heading_documents(), top_k=4)
    labels = dict(zip(graph.node_ids, graph.node_labels))
    assert labels["section:1.5"] == "1.5 Decide Solely on the Evidence"
    assert labels["heading:1.5#1"] == "Pre-trial Publicity"
    assert labels["passage:1.5-c2-s2"] == "1.5 Decide Solely on the Evidence > Pre-trial Publicity"
    # A sibling heading that begins inside a passage is not its subheading.
    assert labels["passage:1.5-c2-s3"] == (
        "1.5 Decide Solely on the Evidence > Pre-trial Publicity | Jury Room Experiments"
    )
    # s2 never says "publicity"; its heading does, so one step already reaches it.
    direct = graph.retrieve("publicity", propagate=False)
    under = next(item for item in direct.explanations if item.node_id == "1.5-c2-s2")
    assert under.path == ("term:publicity", "passage:1.5-c2-s2")
    assert under.path_relations == ("heading_mentions_concept",)
    assert under.path_labels[1] == 'passage:1.5-c2-s2 "1.5 Decide Solely on the Evidence > Pre-trial Publicity"'
    assert StandaloneGraphRetriever(
        heading_documents(), settings=replace(StandaloneGraphSettings(), heading_words=False)
    ).retrieve("publicity", propagate=False).explanations[0].node_id == "1.5-c2-s1"
    assert np.allclose(np.asarray(graph.transition.sum(axis=1)).ravel(), 1)
    assert graph.stats["section_nodes"] == 1 and graph.stats["heading_nodes"] == 2


def test_volume_links_make_the_first_step_idf_weighted_mention_scores():
    documents = [
        Document(page_content="juror excused prejudice", metadata={"passage_id": "gold"}),
        Document(page_content="harry island", metadata={"passage_id": "name"}),
        Document(page_content="juror trial", metadata={"passage_id": "x1"}),
        Document(page_content="excused witness", metadata={"passage_id": "x2"}),
        Document(page_content="trial witness", metadata={"passage_id": "x3"}),
    ]
    query = "Harry the juror was excused"
    legacy = StandaloneGraphRetriever(documents, settings=LEGACY_SETTINGS)
    assert legacy.retrieve(query).explanations[0].node_id == "name"
    graph = StandaloneGraphRetriever(documents)
    links = graph.link_query(query)
    columns = {key: graph.concept_index[key] for key in links}
    total = sum(graph.idf[j] * graph.concept_volume[j] for j in columns.values())
    for item in graph.retrieve(query, propagate=False).explanations:
        row = graph.passage_ids.index(item.node_id)
        expected = sum(graph.idf[j] * graph.incidence[row, j] for j in columns.values()) / total
        assert item.final_score == pytest.approx(expected)
    assert graph.retrieve(query).explanations[0].node_id == "gold"


def test_finder_references_group_their_chunks():
    documents = [
        Document(page_content=text, metadata={"chunk_id": f"r1:ref1:chunk{n}", "row_id": "r1",
                                              "reference_number": 1, "reference_chunk_number": n})
        for n, text in enumerate(["revenue rose", "margins fell"], start=1)
    ]
    result = StandaloneGraphRetriever(documents).retrieve("revenue")
    neighbor = next(item for item in result.explanations if item.node_id.endswith("chunk2"))
    assert neighbor.path[2] == "document:r1/ref1"


def test_corpus_word_forms_merge_only_into_corpus_words():
    forms = CorpusWordForms({"excuse", "excused", "excusing", "empanel", "empanelling", "punch",
                             "punches", "christmas", "jones", "kill", "killing", "killings", "recklessly",
                             "reckless", "witnesses", "witness"})
    merged = {word: forms(word) for word in ("excusing", "excused", "empanelling", "punches",
                                             "killings", "recklessly", "witnesses")}
    assert merged == {"excusing": "excuse", "excused": "excuse", "empanelling": "empanel",
                      "punches": "punch", "killings": "kill", "recklessly": "reckless",
                      "witnesses": "witness"}
    assert forms("christmas") == "christmas" and forms("jones") == "jones"


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
