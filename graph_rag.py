"""Interpretable, dependency-light graph retrieval for passage RAG.

The graph is constructed only from corpus text and document structure.  It does
not use questions, answers, or relevance labels.  Dense retrieval supplies seed
nodes and graph traversal adds passages connected by explicit, inspectable
relations such as adjacent chunks, shared citations, and rare shared terms.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]{3,}")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9&.-]{1,8}\b")
_LEGAL_ID_RE = re.compile(r"^(?P<section>.+?)-c(?P<chunk>\d+)-s(?P<segment>\d+)$")
_ACT_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z'’.-]*\s+){0,6}Act\s+(?:18|19|20)\d{2}"
    r"(?:\s*\([A-Za-z]+\))?(?:\s+s{1,2}\.?\s*\d+[A-Za-z0-9().-]*)?"
)
_CASE_RE = re.compile(
    r"\b(?:R|[A-Z][A-Za-z'’.-]+)\s+v\s+[A-Z][A-Za-z'’.-]+"
    r"(?:\s+\[(?:18|19|20)\d{2}\]\s+[A-Z]{2,8}\s+\d+)?"
)

_STOPWORDS = {
    "about", "above", "after", "again", "against", "almost", "along", "also",
    "among", "another", "answer", "because", "before", "being", "below", "between",
    "both", "could", "does", "doing", "during", "each", "either", "enough", "every",
    "following", "from", "further", "given", "having", "however", "into", "itself",
    "judge", "least", "might", "more", "most", "must", "neither", "other", "otherwise",
    "over", "passage", "question", "rather", "same", "should", "since", "some", "such",
    "than", "that", "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "under", "unless", "until", "upon", "using", "very", "what", "when",
    "where", "whether", "which", "while", "whose", "with", "within", "without", "would",
    "year", "years", "your",
}

_RELATION_PRIORITY = {
    "sequence": 0,
    "same_title": 1,
    "shared_citation": 2,
    "shared_title_term": 3,
    "shared_acronym": 4,
    "shared_term": 5,
}


@dataclass(frozen=True)
class GraphEdge:
    """One undirected, human-readable connection between two passage nodes."""

    source: str
    target: str
    relation: str
    weight: float
    evidence: tuple[str, ...]

    def other(self, node_id: str) -> str:
        if node_id == self.source:
            return self.target
        if node_id == self.target:
            return self.source
        raise ValueError(f"Node {node_id!r} is not incident to this edge")


@dataclass(frozen=True)
class RetrievalExplanation:
    """Score decomposition and strongest graph path for one result."""

    rank: int
    node_id: str
    final_score: float
    dense_score: float | None
    dense_contribution: float
    graph_contribution: float
    seed_id: str
    relation: str
    evidence: tuple[str, ...]
    hop: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphStats:
    nodes: int
    edges: int
    concepts: int
    edges_by_relation: dict[str, int]


@dataclass(frozen=True)
class GraphRetrievalResult:
    documents: list[Document]
    explanations: list[RetrievalExplanation]


def _normalise_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _document_base_id(document: Document) -> str:
    metadata = document.metadata
    for key in ("chunk_id", "passage_id", "document_id", "id"):
        value = str(metadata.get(key, "")).strip()
        if value:
            return value
    digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()[:16]
    return f"passage:{digest}"


def _candidate_concepts(document: Document) -> dict[str, str]:
    """Extract deterministic lexical concepts without an opaque NER model."""

    title = str(document.metadata.get("title", ""))
    text = f"{title}\n{document.page_content}"
    concepts: dict[str, str] = {}

    for match in itertools.chain(_ACT_RE.finditer(text), _CASE_RE.finditer(text)):
        display = " ".join(match.group(0).split())
        concepts[f"citation:{_normalise_label(display)}"] = display

    for acronym in _ACRONYM_RE.findall(text):
        if acronym.casefold() not in _STOPWORDS:
            concepts[f"acronym:{acronym.casefold()}"] = acronym

    title_tokens = [
        token.casefold()
        for token in _TOKEN_RE.findall(title)
        if token.casefold() not in _STOPWORDS
    ]
    for token in title_tokens:
        concepts[f"title:{token}"] = token
    for left, right in zip(title_tokens, title_tokens[1:]):
        phrase = f"{left} {right}"
        concepts[f"title:{phrase}"] = phrase

    for token in set(_TOKEN_RE.findall(text)):
        normalised = token.casefold().strip("-'’")
        if len(normalised) >= 5 and normalised not in _STOPWORDS:
            concepts[f"term:{normalised}"] = normalised
    return concepts


class InterpretableDocumentGraph:
    """Sparse passage graph built from document structure and lexical evidence."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        max_concept_df: int = 24,
        max_concepts_per_node: int = 20,
        max_neighbors: int = 12,
    ) -> None:
        if not documents:
            raise ValueError("At least one document is required to build a graph")
        if max_concept_df < 2:
            raise ValueError("max_concept_df must be at least 2")
        if max_concepts_per_node < 1 or max_neighbors < 1:
            raise ValueError("concept and neighbor limits must be positive")

        self.documents: dict[str, Document] = {}
        self._content_to_ids: defaultdict[str, list[str]] = defaultdict(list)
        duplicate_counts: Counter[str] = Counter()
        for document in documents:
            base_id = _document_base_id(document)
            duplicate_counts[base_id] += 1
            suffix = duplicate_counts[base_id]
            node_id = base_id if suffix == 1 else f"{base_id}#{suffix}"
            self.documents[node_id] = document
            self._content_to_ids[document.page_content].append(node_id)

        concept_df: Counter[str] = Counter()
        for document in self.documents.values():
            concept_df.update(_candidate_concepts(document).keys())

        inverted: defaultdict[str, list[str]] = defaultdict(list)
        concept_labels: dict[str, str] = {}
        for node_id, document in self.documents.items():
            candidates = _candidate_concepts(document)
            eligible = [
                key for key in candidates if 2 <= concept_df[key] <= max_concept_df
            ]
            eligible.sort(
                key=lambda key: (
                    0 if key.startswith("citation:") else
                    1 if key.startswith("title:") else
                    2 if key.startswith("acronym:") else 3,
                    concept_df[key],
                    key,
                )
            )
            selected = eligible[:max_concepts_per_node]
            for key in selected:
                inverted[key].append(node_id)
                concept_labels[key] = candidates[key]

        pair_relations: defaultdict[
            tuple[str, str], dict[str, tuple[float, set[str]]]
        ] = defaultdict(dict)

        def add_pair(
            left: str, right: str, relation: str, weight: float, evidence: str
        ) -> None:
            if left == right:
                return
            pair = tuple(sorted((left, right)))
            prior_weight, prior_evidence = pair_relations[pair].get(
                relation, (0.0, set())
            )
            pair_relations[pair][relation] = (
                max(prior_weight, weight),
                prior_evidence | {evidence},
            )

        self._add_structural_edges(add_pair)
        corpus_size = len(self.documents)
        max_idf = math.log(corpus_size + 1)
        for concept, node_ids in inverted.items():
            document_frequency = len(node_ids)
            idf = math.log((corpus_size + 1) / (document_frequency + 1)) / max_idf
            if concept.startswith("citation:"):
                relation, weight = "shared_citation", max(0.90, idf)
            elif concept.startswith("title:"):
                relation, weight = "shared_title_term", max(0.65, idf)
            elif concept.startswith("acronym:"):
                relation, weight = "shared_acronym", max(0.70, idf)
            else:
                relation, weight = "shared_term", 0.30 + 0.50 * idf
            evidence = concept_labels[concept]
            for left, right in itertools.combinations(sorted(node_ids), 2):
                add_pair(left, right, relation, min(weight, 1.0), evidence)

        candidate_edges: list[GraphEdge] = []
        for (left, right), relations in pair_relations.items():
            relation = min(relations, key=lambda item: _RELATION_PRIORITY[item])
            relation_weight, relation_evidence = relations[relation]
            all_evidence = sorted(
                set().union(*(value[1] for value in relations.values()))
            )
            weight = min(
                1.0,
                relation_weight + 0.025 * (len(relations) - 1) + 0.01 * (len(all_evidence) - 1),
            )
            candidate_edges.append(
                GraphEdge(left, right, relation, weight, tuple(all_evidence[:5]))
            )

        candidate_edges.sort(
            key=lambda edge: (
                _RELATION_PRIORITY[edge.relation], -edge.weight, edge.source, edge.target
            )
        )
        adjacency: defaultdict[str, list[GraphEdge]] = defaultdict(list)
        accepted: list[GraphEdge] = []
        for edge in candidate_edges:
            if (
                len(adjacency[edge.source]) >= max_neighbors
                or len(adjacency[edge.target]) >= max_neighbors
            ):
                continue
            adjacency[edge.source].append(edge)
            adjacency[edge.target].append(edge)
            accepted.append(edge)

        self.adjacency = dict(adjacency)
        relation_counts = Counter(edge.relation for edge in accepted)
        self.stats = GraphStats(
            nodes=len(self.documents),
            edges=len(accepted),
            concepts=len(inverted),
            edges_by_relation=dict(sorted(relation_counts.items())),
        )

    def _add_structural_edges(self, add_pair: Any) -> None:
        source_groups: defaultdict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        legal_groups: defaultdict[str, list[tuple[int, int, str]]] = defaultdict(list)
        title_groups: defaultdict[str, list[str]] = defaultdict(list)

        for node_id, document in self.documents.items():
            metadata = document.metadata
            if "reference_chunk_number" in metadata:
                group = (
                    str(metadata.get("row_id", "unknown")),
                    str(metadata.get("reference_number", "1")),
                )
                source_groups[group].append(
                    (int(metadata["reference_chunk_number"]), node_id)
                )

            passage_id = str(metadata.get("passage_id", metadata.get("id", "")))
            legal_match = _LEGAL_ID_RE.match(passage_id)
            if legal_match:
                legal_groups[legal_match.group("section")].append(
                    (
                        int(legal_match.group("chunk")),
                        int(legal_match.group("segment")),
                        node_id,
                    )
                )

            title = _normalise_label(str(metadata.get("title", "")))
            if title:
                title_groups[title].append(node_id)

        for (row_id, reference_number), members in source_groups.items():
            ordered = [node_id for _, node_id in sorted(members)]
            for left, right in zip(ordered, ordered[1:]):
                add_pair(
                    left,
                    right,
                    "sequence",
                    1.0,
                    f"adjacent chunks in FinDER {row_id} reference {reference_number}",
                )

        for section, members in legal_groups.items():
            ordered = [node_id for _, _, node_id in sorted(members)]
            for left, right in zip(ordered, ordered[1:]):
                add_pair(
                    left,
                    right,
                    "sequence",
                    1.0,
                    f"adjacent passages in legal section {section}",
                )

        for title, members in title_groups.items():
            if len(members) < 2:
                continue
            for left, right in zip(sorted(members), sorted(members)[1:]):
                add_pair(left, right, "same_title", 0.85, f"shared title: {title}")

    def resolve_document(self, document: Document) -> str:
        """Resolve a vector-store result back to its graph node ID."""

        base_id = _document_base_id(document)
        if base_id in self.documents:
            return base_id
        matches = self._content_to_ids.get(document.page_content, [])
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Retrieved document {base_id!r} is not present in the graph")

    def expand_seeds(
        self,
        seeds: Sequence[tuple[str, float]],
        *,
        top_k: int = 5,
        graph_weight: float = 0.35,
    ) -> GraphRetrievalResult:
        """Rerank dense seeds and their one-hop neighbors with score traces."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError("graph_weight must be between 0 and 1")
        valid: dict[str, float] = {}
        for node_id, score in seeds:
            if node_id in self.documents:
                valid[node_id] = max(valid.get(node_id, -math.inf), float(score))
        if not valid:
            return GraphRetrievalResult([], [])

        raw_values = list(valid.values())
        low, high = min(raw_values), max(raw_values)
        if math.isclose(low, high):
            normalised = {node_id: 1.0 for node_id in valid}
        else:
            normalised = {
                node_id: (score - low) / (high - low)
                for node_id, score in valid.items()
            }

        dense_part = {
            node_id: (1.0 - graph_weight) * score
            for node_id, score in normalised.items()
        }
        graph_part: defaultdict[str, float] = defaultdict(float)
        graph_paths: dict[str, tuple[str, GraphEdge]] = {}
        for seed_id, seed_score in normalised.items():
            for edge in self.adjacency.get(seed_id, []):
                neighbor = edge.other(seed_id)
                contribution = graph_weight * seed_score * edge.weight
                if contribution > graph_part[neighbor]:
                    graph_part[neighbor] = contribution
                    graph_paths[neighbor] = (seed_id, edge)

        candidates = set(dense_part) | set(graph_part)
        ranked = sorted(
            candidates,
            key=lambda node_id: (
                -(dense_part.get(node_id, 0.0) + graph_part.get(node_id, 0.0)),
                -dense_part.get(node_id, 0.0),
                node_id,
            ),
        )[:top_k]

        documents: list[Document] = []
        explanations: list[RetrievalExplanation] = []
        for rank, node_id in enumerate(ranked, start=1):
            documents.append(self.documents[node_id])
            if node_id in graph_paths:
                seed_id, edge = graph_paths[node_id]
                relation = edge.relation
                evidence = edge.evidence
                hop = 1
            else:
                seed_id = node_id
                relation = "dense_seed"
                evidence = ("selected by dense cosine similarity",)
                hop = 0
            dense_contribution = dense_part.get(node_id, 0.0)
            graph_contribution = graph_part.get(node_id, 0.0)
            explanations.append(
                RetrievalExplanation(
                    rank=rank,
                    node_id=node_id,
                    final_score=dense_contribution + graph_contribution,
                    dense_score=valid.get(node_id),
                    dense_contribution=dense_contribution,
                    graph_contribution=graph_contribution,
                    seed_id=seed_id,
                    relation=relation,
                    evidence=evidence,
                    hop=hop,
                )
            )
        return GraphRetrievalResult(documents, explanations)


class InterpretableGraphRetriever:
    """Vector-store adapter that exposes documents plus graph explanations."""

    def __init__(
        self,
        vector_store: Any,
        graph: InterpretableDocumentGraph,
        *,
        top_k: int = 5,
        seed_k: int = 10,
        graph_weight: float = 0.35,
    ) -> None:
        if seed_k < top_k:
            raise ValueError("seed_k must be at least top_k")
        self.vector_store = vector_store
        self.graph = graph
        self.top_k = top_k
        self.seed_k = seed_k
        self.graph_weight = graph_weight

    def retrieve(self, query: str) -> GraphRetrievalResult:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
        dense_results = self.vector_store.similarity_search_with_score(
            query, k=self.seed_k
        )
        seeds = [
            (self.graph.resolve_document(document), float(score))
            for document, score in dense_results
        ]
        return self.graph.expand_seeds(
            seeds, top_k=self.top_k, graph_weight=self.graph_weight
        )

    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query).documents


def format_retrieval_explanations(
    explanations: Iterable[RetrievalExplanation],
) -> str:
    """Render compact, stable trace text for logs, notebooks, and prompts."""

    lines = []
    for item in explanations:
        path = item.node_id if item.hop == 0 else f"{item.seed_id} -> {item.node_id}"
        evidence = ", ".join(item.evidence)
        lines.append(
            f"G{item.rank} {path} | {item.relation} ({evidence}) | "
            f"dense={item.dense_contribution:.3f}, graph={item.graph_contribution:.3f}, "
            f"final={item.final_score:.3f}"
        )
    return "\n".join(lines)
