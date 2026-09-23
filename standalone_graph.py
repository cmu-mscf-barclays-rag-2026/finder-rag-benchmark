"""Standalone concept/passage Graph RAG: no embeddings or dense candidates.

Queries attach directly to corpus concept nodes. Personalized PageRank ranks
passages through concept incidence and structural edges. A one-step ablation
uses exactly the same query links without any subsequent propagation.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from langchain_core.documents import Document
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_WORDS = re.compile(r"[a-zA-Z]+(?:['\u2019][a-zA-Z]+)?|\d+(?:\.\d+)*")
_PASSAGE_ID = re.compile(r"^(.+)-c(\d+)-s(\d+)$")
_STOP = ENGLISH_STOP_WORDS | {"said", "say", "says", "mr", "mrs", "ms", "insert"}


def tokens(text: str) -> list[str]:
    """Case-fold, strip possessives, and apply a small explicit plural rule."""
    output = []
    for match in _WORDS.finditer(text.casefold()):
        token = match.group().replace("\u2019", "'")
        if token.endswith("'s"):
            token = token[:-2]
        if token in _STOP or len(token) < 2:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        output.append(token)
    return output


def concepts(text: str) -> Counter[str]:
    """For this simple baseline, concepts are normalized keywords."""
    return Counter(f"term:{word}" for word in tokens(text))


@dataclass(frozen=True)
class StandaloneGraphSettings:
    restart_probability: float = 0.35
    structural_mass: float = 0.15
    title_weight: float = 2.0
    max_term_df_fraction: float = 0.75
    tolerance: float = 1e-9
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if not 0 < self.restart_probability < 1:
            raise ValueError("restart_probability must be strictly between 0 and 1")
        if not 0 <= self.structural_mass < 1:
            raise ValueError("structural_mass must be in [0, 1)")
        if self.title_weight <= 0 or not math.isfinite(self.title_weight):
            raise ValueError("title_weight must be finite and positive")
        if not 0 < self.max_term_df_fraction <= 1:
            raise ValueError("max_term_df_fraction must be in (0, 1]")
        if not 0 < self.tolerance < 1 or self.max_iterations < 1:
            raise ValueError("tolerance must be in (0, 1) and max_iterations positive")


@dataclass(frozen=True)
class StandaloneExplanation:
    rank: int
    node_id: str
    final_score: float
    matched_concept_contribution: float
    other_concept_contribution: float
    structural_contribution: float
    path: tuple[str, ...]
    path_relations: tuple[str, ...]
    matched_concepts: tuple[str, ...]
    top_incoming: tuple[dict[str, Any], ...]
    remaining_incoming_contribution: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StandaloneResult:
    documents: list[Document]
    explanations: list[StandaloneExplanation]
    query_links: dict[str, float]
    iterations: int
    residual: float
    converged: bool


class StandaloneGraphRetriever:
    """Weighted heterogeneous graph, with direct concept personalization.

    Only passage text, title, and source/chunk identifiers are consumed. The
    constructor does not accept an embedding model, vector store, or qrels.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        top_k: int = 5,
        settings: StandaloneGraphSettings | None = None,
    ) -> None:
        if not documents or top_k < 1:
            raise ValueError("non-empty documents and positive top_k are required")
        self.settings = settings or StandaloneGraphSettings()
        self.top_k = top_k
        by_id: dict[str, Document] = {}
        for document in documents:
            metadata = document.metadata
            node_id = str(metadata.get("chunk_id") or metadata.get("passage_id") or
                          metadata.get("id") or hashlib.sha256(
                              document.page_content.encode("utf-8")
                          ).hexdigest())
            if node_id in by_id:
                raise ValueError(f"Duplicate passage ID: {node_id}")
            by_id[node_id] = document
        self.passage_ids = sorted(by_id)
        self.documents = [by_id[node_id] for node_id in self.passage_ids]
        self.passage_count = len(self.documents)
        counts = []
        df: Counter[str] = Counter()
        for document in self.documents:
            content_counts = concepts(document.page_content)
            title_counts = concepts(str(document.metadata.get("title") or ""))
            for key, value in title_counts.items():
                content_counts[key] += self.settings.title_weight * value
            counts.append(content_counts)
            df.update(content_counts.keys())
        max_df = max(1, int(self.settings.max_term_df_fraction * self.passage_count))
        self.concept_ids = sorted(
            key for key, count in df.items() if count <= max_df
        )
        self.concept_index = {key: index for index, key in enumerate(self.concept_ids)}
        self.idf = np.asarray([
            math.log(1 + (self.passage_count - df[key] + 0.5) / (df[key] + 0.5))
            for key in self.concept_ids
        ])
        rows, cols, weights = [], [], []
        for passage_index, row in enumerate(counts):
            for concept_id, frequency in sorted(row.items()):
                column = self.concept_index.get(concept_id)
                if column is None:
                    continue
                # Repeated mentions strengthen an edge with diminishing returns.
                weight = 1.0 + math.log(frequency)
                rows.append(passage_index)
                cols.append(column)
                weights.append(weight)
        self.incidence = sparse.csr_matrix(
            (weights, (rows, cols)), shape=(self.passage_count, len(self.concept_ids)),
            dtype=np.float64,
        )
        # Concept -> passage normalizes each concept's outgoing evidence mass.
        self.concept_to_passage = self._row_normalize(self.incidence.T.tocsr())
        # Passage -> concept prefers locally distinctive concepts.
        passage_to_concept = self._row_normalize(self.incidence.multiply(self.idf).tocsr())
        structural = self._structural_edges()
        has_concepts = np.asarray(passage_to_concept.sum(axis=1)).ravel() > 0
        has_structure = np.asarray(structural.sum(axis=1)).ravel() > 0
        structural_share = np.where(
            has_structure, np.where(has_concepts, self.settings.structural_mass, 1.0), 0.0
        )
        passage_to_concept = sparse.diags(1 - structural_share) @ passage_to_concept
        structure_transition = sparse.diags(structural_share) @ self._row_normalize(structural)
        self.transition = sparse.bmat([
            [structure_transition, passage_to_concept],
            [self.concept_to_passage, None],
        ], format="csr")
        self.transition.sort_indices()
        self.incoming = self.transition.T.tocsr()
        self.dangling = np.asarray(self.transition.sum(axis=1)).ravel() == 0
        self.node_ids = [f"passage:{value}" for value in self.passage_ids] + self.concept_ids
        self.stats = {
            "passage_nodes": self.passage_count,
            "concept_nodes": len(self.concept_ids),
            "total_nodes": len(self.node_ids),
            "incidence_edges": self.incidence.nnz,
            "structural_edges": structural.nnz // 2,
            "directed_transition_edges": self.transition.nnz,
        }

    @staticmethod
    def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        total = np.asarray(matrix.sum(axis=1)).ravel()
        inverse = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
        return (sparse.diags(inverse) @ matrix).tocsr()

    def _structural_edges(self) -> sparse.csr_matrix:
        groups: defaultdict[tuple[str, ...], list[tuple[int, int, int]]] = defaultdict(list)
        for index, document in enumerate(self.documents):
            metadata = document.metadata
            match = _PASSAGE_ID.fullmatch(str(metadata.get("passage_id", "")))
            if match:
                groups[("legal", match[1])].append((int(match[2]), int(match[3]), index))
            elif "reference_chunk_number" in metadata and "row_id" in metadata:
                key = ("finder", str(metadata["row_id"]), str(metadata.get("reference_number", 1)))
                groups[key].append((0, int(metadata["reference_chunk_number"]), index))
        pairs: set[tuple[int, int]] = set()
        for members in groups.values():
            ordered = sorted(members)
            for left, right in zip(ordered, ordered[1:]):
                pairs.add((left[2], right[2]))
                pairs.add((right[2], left[2]))
        ordered_pairs = sorted(pairs)
        return sparse.csr_matrix(
            ([1.0] * len(ordered_pairs),
             ([pair[0] for pair in ordered_pairs], [pair[1] for pair in ordered_pairs])),
            shape=(self.passage_count, self.passage_count),
        )

    def link_query(self, query: str) -> dict[str, float]:
        """Normalize IDF weights over exact normalized corpus-concept matches."""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        matched = sorted(set(concepts(query)) & self.concept_index.keys())
        weights = {key: float(self.idf[self.concept_index[key]]) for key in matched}
        total = sum(weights.values())
        return {key: weight / total for key, weight in weights.items()} if total else {}

    def _representative_path(self, target: int, seeds: set[int]) -> tuple[int, ...]:
        """Shortest positive-transition path; illustrative, not total attribution."""
        queue = deque([target])
        toward_target: dict[int, int | None] = {target: None}
        while queue:
            node = queue.popleft()
            if node in seeds:
                path = [node]
                while toward_target[path[-1]] is not None:
                    path.append(toward_target[path[-1]])
                return tuple(path)
            start, end = self.incoming.indptr[node:node + 2]
            for predecessor, weight in zip(
                self.incoming.indices[start:end], self.incoming.data[start:end]
            ):
                previous = int(predecessor)
                if weight > 0 and previous not in toward_target:
                    toward_target[previous] = node
                    queue.append(previous)
        return ()

    def retrieve(self, query: str, *, propagate: bool = True) -> StandaloneResult:
        links = self.link_query(query)
        if not links:
            return StandaloneResult([], [], {}, 0, 0.0, True)
        personalization = np.zeros(len(self.node_ids), dtype=np.float64)
        for concept_id, weight in links.items():
            personalization[self.passage_count + self.concept_index[concept_id]] = weight
        seed_nodes = set(np.flatnonzero(personalization))
        previous = personalization
        iterations, residual, converged = 1, 0.0, True
        alpha = 1 - self.settings.restart_probability
        if propagate:
            converged = False
            for iterations in range(1, self.settings.max_iterations + 1):
                scores = alpha * (self.incoming @ previous)
                scores += (self.settings.restart_probability + alpha * previous[self.dangling].sum()) * personalization
                residual = float(np.abs(scores - previous).sum())
                if residual <= self.settings.tolerance:
                    converged = True
                    break
                if iterations < self.settings.max_iterations:
                    previous = scores
        else:
            alpha = 1.0
            scores = self.incoming @ personalization

        # Only positive-mass passages are eligible; no arbitrary zero-score fill.
        candidates = np.flatnonzero(scores[:self.passage_count] > 0)
        ranked = sorted(candidates, key=lambda i: (-scores[i], self.passage_ids[i]))[:self.top_k]
        explanations = []
        for rank, target in enumerate(ranked, start=1):
            start, end = self.incoming.indptr[target:target + 2]
            sources = self.incoming.indices[start:end]
            incoming_weights = alpha * self.incoming.data[start:end] * previous[sources]
            matched, other, structure = 0.0, 0.0, 0.0
            contributions = []
            for source, contribution in zip(sources, incoming_weights):
                if contribution <= 0:
                    continue
                if source < self.passage_count:
                    structure += float(contribution)
                    relation = "consecutive_passage"
                else:
                    if source in seed_nodes:
                        matched += float(contribution)
                    else:
                        other += float(contribution)
                    relation = "mentions_concept"
                contributions.append({
                    "source": self.node_ids[source], "relation": relation,
                    "contribution": float(contribution),
                })
            contributions.sort(key=lambda row: (-row["contribution"], row["source"]))
            path = self._representative_path(int(target), seed_nodes)
            relations = tuple(
                "consecutive_passage" if left < self.passage_count and right < self.passage_count
                else "mentions_concept" for left, right in zip(path, path[1:])
            )
            direct_matches = tuple(
                self.node_ids[source] for source in sources if source in seed_nodes
            )
            explanations.append(StandaloneExplanation(
                rank, self.passage_ids[target], float(scores[target]), matched, other,
                structure, tuple(self.node_ids[index] for index in path), relations,
                direct_matches, tuple(contributions[:5]),
                float(sum(row["contribution"] for row in contributions[5:])),
            ))
        return StandaloneResult(
            [self.documents[index] for index in ranked], explanations,
            links, iterations, residual, converged,
        )

    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query).documents
