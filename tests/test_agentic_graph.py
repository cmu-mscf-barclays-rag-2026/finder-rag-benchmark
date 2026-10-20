import pytest
from langchain_core.documents import Document

from agentic_graph import OneRetryGraphRetriever, coverage, peakedness
from standalone_graph import StandaloneGraphRetriever


def fixture_graph():
    return StandaloneGraphRetriever([
        Document(page_content="publicity prejudice directions", metadata={"passage_id": "publicity"}),
        Document(page_content="news story weekend", metadata={"passage_id": "news"}),
        Document(page_content="news prejudice island", metadata={"passage_id": "island"}),
    ], top_k=3)


def counting(text):
    calls = []

    def reformulate(question):
        calls.append(question)
        return text

    return reformulate, calls


def test_confident_first_search_makes_no_llm_call():
    graph = fixture_graph()
    reformulate, calls = counting("pre-trial publicity")
    agent = OneRetryGraphRetriever(graph, reformulate, threshold=0.0)
    result = agent.retrieve("news story")
    assert not result.retried and calls == [] and result.reformulation is None
    assert [d.metadata["passage_id"] for d in result.documents] == [
        d.metadata["passage_id"] for d in graph.invoke("news story")]


def test_unconfident_first_search_retries_once_with_the_reformulation():
    graph = fixture_graph()
    reformulate, calls = counting("pre-trial publicity")
    first = graph.retrieve("news story", top_k=20)
    agent = OneRetryGraphRetriever(graph, reformulate, threshold=peakedness(first) + 1)
    result = agent.retrieve("news story")
    assert result.retried and calls == ["news story"]
    assert "publicity" in [d.metadata["passage_id"] for d in result.documents]
    assert result.query_links["term:news"] + result.query_links["term:story"] == pytest.approx(0.5)
    trace = result.explanations[0].to_dict()
    assert trace["retried"] and trace["reformulation"] == "pre-trial publicity" and trace["path"]
    assert trace["confidence"] == pytest.approx(peakedness(first))


def test_confidence_signals():
    graph = fixture_graph()
    result = graph.retrieve("news story", top_k=20)
    scores = [item.final_score for item in result.explanations]
    assert peakedness(result) == pytest.approx(scores[0] * len(scores) / sum(scores))
    assert peakedness(result) >= 1
    assert coverage(graph, result) == pytest.approx(1.0)  # top passage mentions both words
    with pytest.raises(ValueError):
        OneRetryGraphRetriever(graph, str, threshold=-1)
