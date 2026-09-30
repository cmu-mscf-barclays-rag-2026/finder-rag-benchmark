import json
import math

import pytest
from langchain_core.documents import Document

from llm_graph import (
    LLMExpandedGraphRetriever, LLMRerankedGraphRetriever, rerank_messages, yes_probability,
)
from standalone_graph import StandaloneGraphRetriever


def fixture_graph():
    return StandaloneGraphRetriever([
        Document(page_content="publicity prejudice directions",
                 metadata={"passage_id": "publicity", "title": "Pre-trial Publicity"}),
        Document(page_content="news story weekend", metadata={"passage_id": "news"}),
        Document(page_content="unrelated island", metadata={"passage_id": "island"}),
    ], top_k=3)


def test_expansion_bridges_vocabulary_gap_with_fixed_question_share():
    graph = fixture_graph()
    assert [d.metadata["passage_id"] for d in graph.invoke("news story")] == ["news"]
    expanded = LLMExpandedGraphRetriever(graph, lambda q: "pre-trial publicity", query_weight=0.5)
    links = expanded.seed_links("news story", "pre-trial publicity")
    assert sum(links.values()) == pytest.approx(1)
    assert links["term:news"] + links["term:story"] == pytest.approx(0.5)
    found = {d.metadata["passage_id"] for d in expanded.invoke("news story")}
    assert {"news", "publicity"} <= found


def test_naive_expansion_concatenates_and_empty_expansion_falls_back():
    graph = fixture_graph()
    naive = LLMExpandedGraphRetriever(graph, lambda q: "publicity")
    assert naive.seed_links("news", "publicity") == graph.link_query("news\npublicity")
    mixed = LLMExpandedGraphRetriever(graph, lambda q: "", query_weight=0.5)
    assert mixed.seed_links("news", "") == graph.link_query("news")
    with pytest.raises(ValueError):
        LLMExpandedGraphRetriever(graph, str, query_weight=1.5)


def test_yes_probability_renormalizes_over_yes_and_no():
    top = [{"token": "Yes", "logprob": math.log(0.6)}, {"token": " no", "logprob": math.log(0.2)},
           {"token": "Maybe", "logprob": math.log(0.1)}]
    assert yes_probability(top) == pytest.approx(0.75)
    assert yes_probability([{"token": "Maybe", "logprob": 0.0}]) == 0.0


def test_reranker_pools_both_walks_and_orders_by_llm_relevance():
    graph = fixture_graph()
    expanded = LLMExpandedGraphRetriever(graph, lambda q: "publicity", query_weight=0.5)
    judged = []

    def score(question, document):
        judged.append(document.metadata["passage_id"])
        return 0.9 if document.metadata["passage_id"] == "publicity" else 0.1

    reranker = LLMRerankedGraphRetriever(graph, score, expanded=expanded, pool_depth=3)
    result = reranker.retrieve("news story")
    assert [item.node_id for item in result.explanations][:2] == ["publicity", "news"]
    assert sorted(judged) == ["news", "publicity"]  # each pooled passage judged once
    top = result.explanations[0]
    assert top.found_by == ("expansion",) and top.llm_relevance == 0.9
    trace = top.to_dict()
    assert trace["rank"] == 1 and trace["path"] and trace["llm_relevance"] == 0.9
    assert result.documents[0].metadata["passage_id"] == "publicity"


def test_reranker_breaks_llm_ties_with_fused_graph_rank():
    graph = fixture_graph()
    reranker = LLMRerankedGraphRetriever(graph, lambda q, d: 0.5, pool_depth=3)
    assert ([item.node_id for item in reranker.retrieve("news story").explanations]
            == [item.node_id for item in graph.retrieve("news story").explanations])


def test_rerank_prompt_includes_title_and_truncates_passage():
    document = Document(page_content="x" * 10_000, metadata={"title": "Excusing Jurors"})
    user = rerank_messages("q?", document)[1]["content"]
    assert "Passage title: Excusing Jurors" in user and len(user) < 3_000


def test_links_api_validates_concepts_and_top_k():
    graph = fixture_graph()
    with pytest.raises(ValueError):
        graph.retrieve_from_links({"term:missing": 1.0})
    assert len(graph.retrieve("news publicity island", top_k=1).documents) == 1


def test_graph_llm_mode_expands_reranks_and_never_constructs_embeddings(monkeypatch):
    import rag
    from langchain_core.runnables import RunnableLambda

    def forbidden(*args, **kwargs):
        raise AssertionError("graph_llm attempted to create embeddings or a vector store")

    prompts = []

    def fake_llm(prompt):
        text = prompt.to_string()
        prompts.append(text)
        return "net sales" if "search phrases" in text else "Revenue rose [S1]."

    monkeypatch.setattr(rag, "HuggingFaceEmbeddings", forbidden)
    monkeypatch.setattr(rag.InMemoryVectorStore, "from_documents", forbidden)
    monkeypatch.setattr(rag, "check_ollama", lambda *args: rag.OllamaStatus(True, "test"))
    monkeypatch.setattr(rag, "ChatOllama", lambda **kwargs: RunnableLambda(fake_llm))
    monkeypatch.setattr(rag, "OllamaRelevanceScorer", lambda *args: (
        lambda question, document: 0.9 if "Net sales" in document.page_content else 0.1))
    engine, _ = rag.build_rag(
        rag.RAGSettings(retrieval_mode="graph_llm", top_k=1, llm_rerank_pool=2),
        records=[{"_id": "a", "references": ["Revenue guidance."]},
                 {"_id": "b", "references": ["Net sales rose."]},
                 {"_id": "c", "references": ["Operating costs fell."]}],
    )
    result = engine.ask("revenue")
    assert result.answer == "Revenue rose [S1]."
    assert result.sources[0].page_content == "Net sales rose."  # found only via expansion
    trace = result.retrieval_trace[0]
    assert trace["found_by"] == ["expansion"] and trace["llm_relevance"] == 0.9
    assert sum("search phrases" in text for text in prompts) == 1  # one expansion call


def test_team_protocol_text_only_documents_split_groups_and_bootstrap():
    from legal_protocol import paired_bootstrap, split_queries
    from legal_data import legal_records_to_documents

    record = {"id": "1.2-c1-s1", "title": "Excusing Jurors", "text": "Body.", "footnotes": "Note."}
    plain = legal_records_to_documents([record], text_only=True)[0]
    assert plain.page_content == "Body." and "title" not in plain.metadata
    assert "Footnotes" in legal_records_to_documents([record])[0].page_content
    # Questions whose gold passages share exact text stay on one side of the split.
    corpus = [{"id": f"p{i}", "text": "same" if i < 2 else f"text {i}"} for i in range(10)]
    qa = [{"id": i, "relevant_passage_id": f"p{i}"} for i in range(10)]
    dev, test = split_queries(qa, corpus)
    assert len(dev) >= 2 and not dev & test and len(dev | test) == 10
    assert {"0", "1"} <= dev or {"0", "1"} <= test
    low, high = paired_bootstrap([1.0] * 5 + [0.0] * 5, [f"g{i}" for i in range(10)])
    assert 0.0 <= low <= 0.5 <= high <= 1.0


def test_team_result_files_use_legal_rag_columns(tmp_path):
    import csv

    from legal_data import LegalQuestion
    from legal_protocol import write_results

    questions = [LegalQuestion(str(i), f"q{i}", "", f"p{i}") for i in (1, 2, 3)]
    rankings = {"base": {"1": ["p1"], "2": ["x"], "3": ["x", "p3"]},
                "new": {"1": ["p1"], "2": ["p2"], "3": ["p3"]}}
    write_results(tmp_path, questions=questions, dev_ids={"1"}, test_ids={"2", "3"},
                  rankings=rankings, config_ids={"base": "b", "new": "n"},
                  latency={"base": [1.0, 3.0]}, baseline="base")

    def header(name):
        with (tmp_path / name).open(encoding="utf-8") as handle:
            return next(csv.reader(handle))

    # Column names and order from legal_rag/run.py on feature/person2-bm25.
    assert header("per_query.csv") == ["method", "query_id", "k", "recall", "hit_rate",
                                       "all_evidence_hit", "evidence_coverage", "mrr", "ndcg"]
    assert header("test_metrics.csv") == ["method", "config_id", "k", "n_queries", "recall",
                                          "hit_rate", "all_evidence_hit", "evidence_coverage",
                                          "mrr", "ndcg", "latency_mean_ms", "latency_p95_ms"]
    with (tmp_path / "test_metrics.csv").open(encoding="utf-8") as handle:
        rows = {(r["method"], r["k"]): r for r in csv.DictReader(handle)}
    assert float(rows["new", "5"]["recall"]) == 1.0 and float(rows["base", "5"]["recall"]) == 0.5
    assert float(rows["base", "1"]["mrr"]) == 0.0 and rows["new", "5"]["latency_mean_ms"] == ""
    assert json.loads((tmp_path / "split.json").read_text()) == {"dev": [1], "test": [2, 3]}
