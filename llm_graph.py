"""LLM-augmented standalone graph retrieval.

The graph, its concept vocabulary, and the Personalized PageRank walk are
unchanged. A local LLM is used in two optional places:

1. Keyword expansion: it rewrites the question as heading-style search phrases
   for the underlying issue, and their corpus concepts join the question's
   concepts as walk seeds.
2. Reranking: it judges whether each pooled graph candidate helps answer the
   question; P(Yes) from the first answer token orders the final passages.

No relevance labels, answers, or dense embeddings reach the LLM or the graph.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable
from urllib.request import Request, urlopen

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from standalone_graph import StandaloneExplanation, StandaloneGraphRetriever, StandaloneResult


EXPANSION_PROMPT_VERSION = "keywords-v1"
# Issue keywords: short heading-style phrases, without scenario filler.
EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You help a keyword search engine find the passage of a reference document "
        "that answers a question. First identify the underlying issue, ignoring "
        "people's names and incidental story details. Then output 6 to 10 search "
        "phrases using the precise technical terms and section-heading wording "
        "such a document would use. Output only the phrases, comma-separated.",
    ),
    ("human", "Question: {question}"),
])

RERANK_PROMPT_VERSION = "yesno-v1"
RERANK_SYSTEM = (
    "You are a relevance judge for a search engine. Decide whether the passage "
    "contains information that helps answer the question. Answer with exactly one "
    "word: Yes or No."
)
RERANK_MAX_CHARS = 2_500


def expansion_chain(llm: Runnable) -> Runnable:
    return EXPANSION_PROMPT | llm | StrOutputParser()


def rerank_messages(question: str, document: Document) -> list[dict[str, str]]:
    title = str(document.metadata.get("title") or "").strip()
    header = f"Passage title: {title}\n" if title else ""
    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": (
            f"Question: {question}\n\n{header}Passage:\n"
            f"{document.page_content[:RERANK_MAX_CHARS]}\n\n"
            "Does this passage help answer the question?"
        )},
    ]


def yes_probability(top_logprobs: list[dict[str, Any]]) -> float:
    """P(Yes) renormalized over Yes/No among the first token's alternatives."""
    mass = {"yes": 0.0, "no": 0.0}
    for item in top_logprobs:
        word = str(item.get("token", "")).strip().lower()
        if word in mass:
            mass[word] += math.exp(float(item["logprob"]))
    total = mass["yes"] + mass["no"]
    return mass["yes"] / total if total else 0.0


class OllamaRelevanceScorer:
    """Score (question, passage) with one constrained Ollama token and logprobs.

    LangChain's ChatOllama does not expose token logprobs, so this calls the
    local Ollama chat endpoint directly.
    """

    def __init__(self, model: str, base_url: str, timeout: float = 120.0) -> None:
        self.model, self.timeout = model, timeout
        self.url = f"{base_url.rstrip('/')}/api/chat"

    def __call__(self, question: str, document: Document) -> float:
        body = json.dumps({
            "model": self.model, "messages": rerank_messages(question, document),
            "stream": False, "logprobs": True, "top_logprobs": 10,
            "options": {"temperature": 0, "num_predict": 1},
        }).encode("utf-8")
        request = Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        tokens = payload.get("logprobs") or []
        return yes_probability(tokens[0].get("top_logprobs", [])) if tokens else 0.0


class LLMExpandedGraphRetriever:
    """Seed the standalone graph walk with question + LLM-expansion concepts.

    ``query_weight=None`` is the naive variant: the expansion is appended to the
    question and linked as one text. A float in [0, 1] instead mixes the two
    IDF-normalized seed distributions, reserving that share for the question.
    """

    def __init__(
        self,
        graph: StandaloneGraphRetriever,
        expand: Callable[[str], str],
        *,
        query_weight: float | None = None,
    ) -> None:
        if query_weight is not None and not 0 <= query_weight <= 1:
            raise ValueError("query_weight must be in [0, 1]")
        self.graph = graph
        self.expand = expand
        self.query_weight = query_weight

    def seed_links(self, query: str, expansion: str) -> dict[str, float]:
        if self.query_weight is None:
            return self.graph.link_query(f"{query}\n{expansion}")
        question = self.graph.link_query(query)
        phrases = self.graph.link_query(expansion) if expansion.strip() else {}
        if not question or not phrases:
            return question or phrases
        mixed = {key: self.query_weight * value for key, value in question.items()}
        for key, value in phrases.items():
            mixed[key] = mixed.get(key, 0.0) + (1 - self.query_weight) * value
        return mixed

    def retrieve(self, query: str, *, top_k: int | None = None) -> StandaloneResult:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return self.graph.retrieve_from_links(
            self.seed_links(query, self.expand(query)), top_k=top_k
        )

    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query).documents


@dataclass(frozen=True)
class RerankedExplanation:
    rank: int
    node_id: str
    llm_relevance: float
    fused_graph_score: float
    found_by: tuple[str, ...]
    graph: StandaloneExplanation

    def to_dict(self) -> dict[str, Any]:
        # Keep the graph trace keys (path, contributions) for existing displays.
        return {**self.graph.to_dict(), "rank": self.rank, "llm_relevance": self.llm_relevance,
                "fused_graph_score": self.fused_graph_score, "found_by": list(self.found_by),
                "graph_rank": self.graph.rank}


@dataclass(frozen=True)
class RerankedResult:
    documents: list[Document]
    explanations: list[RerankedExplanation]
    query_links: dict[str, float]


class LLMRerankedGraphRetriever:
    """Pool graph candidates, then order them by LLM relevance.

    The pool is the top ``pool_depth`` passages of the question-only walk and,
    when ``expanded`` is given, of the expanded walk. Reciprocal-rank fusion of
    those graph rankings orders the pool and breaks ties in LLM relevance.
    """

    RRF_K = 60

    def __init__(
        self,
        graph: StandaloneGraphRetriever,
        score: Callable[[str, Document], float],
        *,
        expanded: LLMExpandedGraphRetriever | None = None,
        pool_depth: int = 20,
    ) -> None:
        if pool_depth < 1:
            raise ValueError("pool_depth must be positive")
        self.graph, self.score, self.expanded = graph, score, expanded
        self.pool_depth = pool_depth

    def retrieve(self, query: str, *, top_k: int | None = None) -> RerankedResult:
        top_k = self.graph.top_k if top_k is None else top_k
        runs = {"question": self.graph.retrieve(query, top_k=self.pool_depth)}
        if self.expanded is not None:
            runs["expansion"] = self.expanded.retrieve(query, top_k=self.pool_depth)
        fused: dict[str, float] = {}
        found_by: dict[str, list[str]] = {}
        best: dict[str, tuple[Document, StandaloneExplanation]] = {}
        for name, run in runs.items():
            for document, explanation in zip(run.documents, run.explanations):
                node = explanation.node_id
                fused[node] = fused.get(node, 0.0) + 1 / (self.RRF_K + explanation.rank)
                found_by.setdefault(node, []).append(name)
                if node not in best or explanation.rank < best[node][1].rank:
                    best[node] = (document, explanation)
        relevance = {node: self.score(query, best[node][0]) for node in fused}
        ranked = sorted(fused, key=lambda node: (-relevance[node], -fused[node], node))[:top_k]
        return RerankedResult(
            [best[node][0] for node in ranked],
            [RerankedExplanation(rank, node, relevance[node], fused[node],
                                 tuple(found_by[node]), best[node][1])
             for rank, node in enumerate(ranked, start=1)],
            runs.get("expansion", runs["question"]).query_links,
        )

    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query).documents
