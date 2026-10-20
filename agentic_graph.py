"""One-retry agentic search around the standalone graph.

1. Search the graph once.
2. Inspect retrieval confidence (peakedness of the PageRank scores).
3. If it is below a threshold, reformulate the question with the LLM keyword
   expansion and search a second time, with the question keeping a fixed
   share of the seed mass. The second walk combines both query versions, so
   its ranking is the answer.

At most one extra search step is taken. Confident queries cost one graph walk
and no LLM call. No relevance labels, answers, or embeddings are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from langchain_core.documents import Document

from llm_graph import LLMExpandedGraphRetriever
from standalone_graph import StandaloneExplanation, StandaloneGraphRetriever, StandaloneResult


def peakedness(result: StandaloneResult) -> float:
    """Top score over the mean of the returned scores (1 = flat, higher = one
    passage stands out). A standard score-based query-performance predictor."""
    scores = np.asarray([item.final_score for item in result.explanations])
    return float(scores[0] / scores.mean()) if len(scores) and scores.mean() > 0 else 0.0


def coverage(graph: StandaloneGraphRetriever, result: StandaloneResult) -> float:
    """Share of the question's link mass on concepts the top passage mentions."""
    if not result.explanations:
        return 0.0
    row = graph.passage_ids.index(result.explanations[0].node_id)
    return float(sum(weight for concept, weight in result.query_links.items()
                     if graph.incidence[row, graph.concept_index[concept]] > 0))


@dataclass(frozen=True)
class AgenticExplanation:
    rank: int
    node_id: str
    retried: bool
    confidence: float
    threshold: float
    reformulation: str | None
    graph: StandaloneExplanation

    def to_dict(self) -> dict[str, Any]:
        # Keep the graph trace keys (path, contributions) for existing displays.
        return {**self.graph.to_dict(), "retried": self.retried, "confidence": self.confidence,
                "confidence_threshold": self.threshold, "reformulation": self.reformulation}


@dataclass(frozen=True)
class AgenticResult:
    documents: list[Document]
    explanations: list[AgenticExplanation]
    query_links: dict[str, float]
    confidence: float
    retried: bool
    reformulation: str | None


class OneRetryGraphRetriever:
    """Graph search with at most one confidence-gated reformulated retry."""

    def __init__(
        self,
        graph: StandaloneGraphRetriever,
        reformulate: Callable[[str], str],
        *,
        threshold: float,
        query_weight: float = 0.5,
        depth: int = 20,
    ) -> None:
        if threshold < 0 or depth < 1:
            raise ValueError("threshold must be non-negative and depth positive")
        self.graph, self.threshold, self.depth = graph, threshold, depth
        self.reformulate = reformulate
        self.expanded = LLMExpandedGraphRetriever(graph, reformulate, query_weight=query_weight)

    def retrieve(self, query: str, *, top_k: int | None = None) -> AgenticResult:
        top_k = self.graph.top_k if top_k is None else top_k
        # Confidence is read from the same depth whatever top_k is requested.
        first = self.graph.retrieve(query, top_k=max(top_k, self.depth))
        confidence = peakedness(first) if first.explanations else 0.0
        result, reformulation = first, None
        if confidence < self.threshold:
            reformulation = self.reformulate(query)
            result = self.graph.retrieve_from_links(
                self.expanded.seed_links(query, reformulation), top_k=top_k
            )
        retried = reformulation is not None
        return AgenticResult(
            result.documents[:top_k],
            [AgenticExplanation(item.rank, item.node_id, retried, confidence, self.threshold,
                                reformulation, item)
             for item in result.explanations[:top_k]],
            result.query_links, confidence, retried, reformulation,
        )

    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query).documents
