"""Standalone concept/passage Graph RAG: no embeddings or dense candidates.

Nodes are passages, keyword concepts, and the corpus's own sections and
markdown headings. Passages link to the concepts they mention, including the
words of the headings they sit under, and to those headings; headings link to
their parent. Queries attach directly to concept nodes, and Personalized
PageRank ranks passages. A one-step ablation uses exactly the same query links
without any subsequent propagation.

``LEGACY_SETTINGS`` rebuilds the original keyword graph: no heading words,
consecutive-passage edges, and IDF-only query links.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from langchain_core.documents import Document
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_WORDS = re.compile(r"[a-zA-Z]+(?:['’][a-zA-Z]+)?|\d+(?:\.\d+)*")
_PASSAGE_ID = re.compile(r"^(.+)-c(\d+)-s(\d+)$")
_STOP = ENGLISH_STOP_WORDS | {"said", "say", "says", "mr", "mrs", "ms", "insert"}
_HEADING = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def _words(text: str) -> list[str]:
    """Case-folded words without possessives, stop words, or 1-character tokens."""
    output = []
    for match in _WORDS.finditer(text.casefold()):
        token = match.group().replace("’", "'")
        if token.endswith("'s"):
            token = token[:-2]
        if token not in _STOP and len(token) >= 2:
            output.append(token)
    return output


def _plural(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokens(text: str) -> list[str]:
    """Case-fold, strip possessives, and apply a small explicit plural rule."""
    return [_plural(token) for token in _words(text)]


def concepts(text: str) -> Counter[str]:
    """Legacy concepts: plural-normalized keywords."""
    return Counter(f"term:{word}" for word in tokens(text))


class CorpusWordForms:
    """Merge inflections into a base word that itself occurs in the corpus.

    Plural, -ing, -ed, and -ly endings are removed only when the shorter word
    is in the corpus vocabulary, so every concept is a real corpus word
    ("excusing" -> "excuse", "empanelling" -> "empanel", "punches" -> "punch")
    and words without a corpus base stay whole ("christmas", "jones").
    """

    def __init__(self, vocabulary: set[str]) -> None:
        self.vocabulary = vocabulary
        self._base: dict[str, str] = {}

    def _candidates(self, word: str) -> list[str]:
        candidates = []
        if len(word) > 4 and word.endswith("ies"):
            candidates.append(word[:-3] + "y")
        if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            candidates += [word[:-1], word[:-2]] if word.endswith("es") else [word[:-1]]
        for suffix in ("ing", "ed"):
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                candidates += [stem + "e", stem]
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    candidates.append(stem[:-1])
        if len(word) >= 7 and word.endswith("ly"):
            candidates.append(word[:-2])
        return [c for c in candidates if len(c) >= 3 and c not in _STOP and c in self.vocabulary]

    def __call__(self, word: str) -> str:
        if word not in self._base:
            base, seen = word, {word}
            while (options := self._candidates(base)) and options[0] not in seen:
                base = options[0]
                seen.add(base)
            self._base[word] = base
        return self._base[word]


@dataclass(frozen=True)
class StandaloneGraphSettings:
    restart_probability: float = 0.35
    structural_mass: float = 0.15
    title_weight: float = 2.0
    max_term_df_fraction: float = 0.75
    tolerance: float = 1e-9
    max_iterations: int = 100
    # Graph construction; LEGACY_SETTINGS reproduces the original graph.
    heading_words: bool = True  # passages inherit the words of the headings they sit under
    structure: str = "headings"  # "headings" section/heading tree; "consecutive" adjacent passages
    query_links: str = "volume"  # "volume" IDF x concept edge volume; "idf" IDF only
    word_forms: str = "plural"  # "corpus" also merges -ing/-ed/-ly forms into corpus words

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
        if self.word_forms not in {"corpus", "plural"}:
            raise ValueError("word_forms must be corpus or plural")
        if self.structure not in {"headings", "consecutive", "none"}:
            raise ValueError("structure must be headings, consecutive, or none")
        if self.query_links not in {"volume", "idf"}:
            raise ValueError("query_links must be volume or idf")


LEGACY_SETTINGS = StandaloneGraphSettings(
    heading_words=False, structure="consecutive", query_links="idf"
)


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
    path_labels: tuple[str, ...]
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


@dataclass
class _Section:
    """A section, markdown heading, or reference document grouping passages."""

    node_id: str
    label: str
    parent: int | None
    passages: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)


def _clean_heading(text: str) -> str:
    return " ".join(re.sub(r"[*_`]+", "", text).split())


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
        self.word_forms = None
        if self.settings.word_forms == "corpus":
            vocabulary: set[str] = set()
            for document in self.documents:
                vocabulary.update(_words(document.page_content))
                vocabulary.update(_words(str(document.metadata.get("title") or "")))
            self.word_forms = CorpusWordForms(vocabulary)
        has_tree = self.settings.structure == "headings"
        self.sections: list[_Section] = []
        attached: list[list[int]] = [[] for _ in self.documents]
        if has_tree or self.settings.heading_words:
            self.sections, attached = self._section_tree()
        self.section_passages = self._passages_under()
        # Each passage sits under one or more chains: section > heading > subheading.
        self.breadcrumbs = [[self._chain(index) for index in sections] for sections in attached]
        counts = []
        inherited: set[tuple[int, str]] = set()
        df: Counter[str] = Counter()
        for passage_index, document in enumerate(self.documents):
            content_counts = self.concepts(document.page_content)
            body = set(content_counts)
            # Title and heading words count title_weight times, before log scaling.
            titles = [str(document.metadata.get("title") or "")]
            if self.settings.heading_words:
                titles += [self.sections[index].label for index in dict.fromkeys(
                    index for chain in self.breadcrumbs[passage_index] for index in chain
                )]
            for title in titles:
                for key, value in self.concepts(title).items():
                    content_counts[key] += self.settings.title_weight * value
            inherited.update((passage_index, key) for key in content_counts.keys() - body)
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
        # Passage-concept edges whose word appears only in the passage's headings.
        self.heading_only = {
            (passage, self.concept_index[key]) for passage, key in inherited
            if key in self.concept_index
        }
        # Total edge weight per concept; it sets the query-link volume correction.
        self.concept_volume = np.asarray(self.incidence.sum(axis=0)).ravel()
        # Concept -> passage normalizes each concept's outgoing evidence mass.
        self.concept_to_passage = self._row_normalize(self.incidence.T.tocsr())
        # Passage -> concept prefers locally distinctive concepts.
        passage_to_concept = self._row_normalize(self.incidence.multiply(self.idf).tocsr())
        has_concepts = np.asarray(passage_to_concept.sum(axis=1)).ravel() > 0
        # Section and heading nodes join the graph only with the heading tree.
        tree = self.sections if has_tree else []
        if tree:
            structural = self._passage_to_section()
        elif self.settings.structure == "consecutive":
            structural = self._consecutive_edges()
        else:
            structural = sparse.csr_matrix((self.passage_count, 0))
        has_structure = np.asarray(structural.sum(axis=1)).ravel() > 0
        structural_share = np.where(
            has_structure, np.where(has_concepts, self.settings.structural_mass, 1.0), 0.0
        )
        passage_to_concept = sparse.diags(1 - structural_share) @ passage_to_concept
        structure_transition = sparse.diags(structural_share) @ self._row_normalize(structural)
        if tree:
            section_to_passage, section_to_section = self._section_transitions()
            self.transition = sparse.bmat([
                [None, passage_to_concept, structure_transition],
                [self.concept_to_passage, None, None],
                [section_to_passage, None, section_to_section],
            ], format="csr")
        else:
            self.transition = sparse.bmat([
                [structure_transition if self.settings.structure == "consecutive" else None,
                 passage_to_concept],
                [self.concept_to_passage, None],
            ], format="csr")
        self.transition.sort_indices()
        self.incoming = self.transition.T.tocsr()
        self.dangling = np.asarray(self.transition.sum(axis=1)).ravel() == 0
        self.node_ids = ([f"passage:{value}" for value in self.passage_ids] + self.concept_ids
                         + [section.node_id for section in tree])
        # Passages are labeled by their breadcrumb, headings by their wording.
        self.node_labels = (
            [self._breadcrumb_label(passage) for passage in range(self.passage_count)]
            + [""] * len(self.concept_ids) + [section.label for section in tree]
        )
        self.stats = {
            "passage_nodes": self.passage_count,
            "concept_nodes": len(self.concept_ids),
            "section_nodes": sum(section.parent is None for section in tree),
            "heading_nodes": sum(section.parent is not None for section in tree),
            "total_nodes": len(self.node_ids),
            "incidence_edges": self.incidence.nnz,
            "heading_only_incidence_edges": len(self.heading_only),
            "structural_edges": (structural.nnz + sum(section.parent is not None for section in tree)
                                 if tree else structural.nnz // 2),
            "directed_transition_edges": self.transition.nnz,
        }

    @staticmethod
    def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        total = np.asarray(matrix.sum(axis=1)).ravel()
        inverse = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
        return (sparse.diags(inverse) @ matrix).tocsr()

    def concepts(self, text: str) -> Counter[str]:
        """Concepts are normalized keywords, linked by exact match."""
        words = _words(text)
        normalize = self.word_forms or _plural
        return Counter(f"term:{normalize(word)}" for word in words)

    def _document_groups(self) -> dict[tuple[str, ...], list[tuple[int, int, int]]]:
        """Passages of one legal section or FinDER reference, in reading order."""
        groups: defaultdict[tuple[str, ...], list[tuple[int, int, int]]] = defaultdict(list)
        for index, document in enumerate(self.documents):
            metadata = document.metadata
            match = _PASSAGE_ID.fullmatch(str(metadata.get("passage_id", "")))
            if match:
                groups[("legal", match[1])].append((int(match[2]), int(match[3]), index))
            elif "reference_chunk_number" in metadata and "row_id" in metadata:
                key = ("finder", str(metadata["row_id"]), str(metadata.get("reference_number", 1)))
                groups[key].append((0, int(metadata["reference_chunk_number"]), index))
        return {key: sorted(members) for key, members in groups.items()}

    def _consecutive_edges(self) -> sparse.csr_matrix:
        pairs: set[tuple[int, int]] = set()
        for ordered in self._document_groups().values():
            for left, right in zip(ordered, ordered[1:]):
                pairs.add((left[2], right[2]))
                pairs.add((right[2], left[2]))
        ordered_pairs = sorted(pairs)
        return sparse.csr_matrix(
            ([1.0] * len(ordered_pairs),
             ([pair[0] for pair in ordered_pairs], [pair[1] for pair in ordered_pairs])),
            shape=(self.passage_count, self.passage_count),
        )

    def _section_tree(self) -> tuple[list[_Section], list[list[int]]]:
        """Section/document roots with their markdown heading trees, and the
        sections each passage is attached to.

        Reading each section in order, a passage sits under the heading in
        effect where it starts and under every heading that begins inside it.
        A level-1 heading names the section itself.
        """
        sections: list[_Section] = []
        attached_by_passage: list[list[int]] = [[] for _ in self.documents]
        for key, ordered in sorted(self._document_groups().items()):
            name = key[1] if key[0] == "legal" else f"{key[1]}/ref{key[2]}"
            root = len(sections)
            sections.append(_Section(
                f"section:{name}" if key[0] == "legal" else f"document:{name}", "", None
            ))
            stack: list[tuple[int, int]] = []  # (heading level, section index)
            for _, _, index in ordered:
                text = self.documents[index].page_content
                headings = list(_HEADING.finditer(text))
                attached = []
                if not headings or text[:headings[0].start()].strip():
                    attached.append(stack[-1][1] if stack else root)
                for heading in headings:
                    level, label = len(heading[1]), _clean_heading(heading[2])
                    if level == 1:
                        stack.clear()
                        sections[root].label = sections[root].label or label
                        attached.append(root)
                        continue
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    parent = stack[-1][1] if stack else root
                    sections.append(_Section(
                        f"heading:{name}#{len(sections) - root}", label, parent
                    ))
                    sections[parent].children.append(len(sections) - 1)
                    stack.append((level, len(sections) - 1))
                    attached.append(len(sections) - 1)
                attached_by_passage[index] = list(dict.fromkeys(attached))
                for section_index in attached_by_passage[index]:
                    sections[section_index].passages.append(index)
        return sections, attached_by_passage

    def _chain(self, index: int) -> list[int]:
        """Section indices from the root down to ``index``."""
        chain = [index]
        while (parent := self.sections[chain[-1]].parent) is not None:
            chain.append(parent)
        return chain[::-1]

    def _breadcrumb_label(self, passage: int) -> str:
        """'Section > Heading', with '|' before a sibling heading that begins
        inside the passage."""
        text, previous = "", []
        for chain in self.breadcrumbs[passage]:
            shared = 0
            while shared < min(len(chain), len(previous)) and chain[shared] == previous[shared]:
                shared += 1
            rest = [self.sections[index].label for index in chain[shared:]
                    if self.sections[index].label]
            if rest:
                joiner = " > " if shared == len(previous) else " | "
                text += (joiner if text else "") + " > ".join(rest)
            previous = chain
        return text

    def _passages_under(self) -> list[set[int]]:
        """Passages in each section's subtree (children follow their parents)."""
        under = [set(section.passages) for section in self.sections]
        for index in reversed(range(len(self.sections))):
            if (parent := self.sections[index].parent) is not None:
                under[parent] |= under[index]
        return under

    def _passage_to_section(self) -> sparse.csr_matrix:
        pairs = sorted({(passage, index) for index, section in enumerate(self.sections)
                        for passage in section.passages})
        return sparse.csr_matrix(
            ([1.0] * len(pairs), ([pair[0] for pair in pairs], [pair[1] for pair in pairs])),
            shape=(self.passage_count, len(self.sections)),
        )

    def _section_transitions(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """A section passes structural_mass to its parent and splits the rest
        over its own passages and subheadings by the passages each covers, so
        every passage under a heading receives the same share."""
        to_passage: list[tuple[int, int, float]] = []
        to_section: list[tuple[int, int, float]] = []
        for index, section in enumerate(self.sections):
            sizes = [len(self.section_passages[child]) for child in section.children]
            targets = len(section.passages) + sum(sizes)
            up = self.settings.structural_mass if section.parent is not None and targets else 0.0
            if section.parent is not None:
                to_section.append((index, section.parent, 1.0 if not targets else up))
            for passage in section.passages:
                to_passage.append((index, passage, (1 - up) / targets))
            for child, size in zip(section.children, sizes):
                to_section.append((index, child, (1 - up) * size / targets))
        shape = len(self.sections)

        def matrix(entries: list[tuple[int, int, float]], columns: int) -> sparse.csr_matrix:
            return sparse.csr_matrix(
                ([e[2] for e in entries], ([e[0] for e in entries], [e[1] for e in entries])),
                shape=(shape, columns),
            )

        return matrix(to_passage, self.passage_count), matrix(to_section, shape)

    def link_query(self, query: str) -> dict[str, float]:
        """Normalized link weights over exact normalized corpus-concept matches.

        With ``query_links="volume"`` a concept's weight is IDF times its total
        edge weight, so the first walk step gives each passage IDF x its
        log-scaled mention weight for every matched concept. With ``"idf"``,
        one-off words such as names receive most of the query mass.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        matched = sorted(set(self.concepts(query)) & self.concept_index.keys())
        volume = self.settings.query_links == "volume"
        weights = {
            key: float(self.idf[i] * (self.concept_volume[i] if volume else 1.0))
            for key in matched for i in [self.concept_index[key]]
        }
        total = sum(weights.values())
        return {key: weight / total for key, weight in weights.items()} if total else {}

    def _kind(self, index: int) -> str:
        if index < self.passage_count:
            return "passage"
        return "concept" if index < self.passage_count + len(self.concept_ids) else "section"

    def _relation(self, left: int, right: int) -> str:
        kinds = {self._kind(left), self._kind(right)}
        if kinds == {"passage"}:
            return "consecutive_passage"
        if kinds == {"passage", "concept"}:
            passage, concept = sorted((left, right))
            if (passage, concept - self.passage_count) in self.heading_only:
                return "heading_mentions_concept"
            return "mentions_concept"
        return "under_heading"

    def node_label(self, index: int) -> str:
        label = self.node_labels[index]
        return f'{self.node_ids[index]} "{label}"' if label else self.node_ids[index]

    def _representative_path(
        self, target: int, seeds: set[int], scores: np.ndarray
    ) -> tuple[int, ...]:
        """Shortest positive-transition path, preferring stronger incoming
        edges; illustrative, not total attribution."""
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
            sources = self.incoming.indices[start:end]
            strength = self.incoming.data[start:end] * scores[sources]
            for order in np.argsort(-strength, kind="stable"):
                previous = int(sources[order])
                if self.incoming.data[start + order] > 0 and previous not in toward_target:
                    toward_target[previous] = node
                    queue.append(previous)
        return ()

    def retrieve(
        self, query: str, *, propagate: bool = True, top_k: int | None = None
    ) -> StandaloneResult:
        return self.retrieve_from_links(
            self.link_query(query), propagate=propagate, top_k=top_k
        )

    def retrieve_from_links(
        self, links: dict[str, float], *, propagate: bool = True, top_k: int | None = None
    ) -> StandaloneResult:
        """Rank passages from an explicit concept personalization (weights sum to 1)."""
        top_k = self.top_k if top_k is None else top_k
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if unknown := set(links) - self.concept_index.keys():
            raise ValueError(f"Unknown concepts in personalization: {sorted(unknown)[:5]}")
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
        ranked = sorted(candidates, key=lambda i: (-scores[i], self.passage_ids[i]))[:top_k]
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
                if self._kind(source) != "concept":
                    structure += float(contribution)
                elif source in seed_nodes:
                    matched += float(contribution)
                else:
                    other += float(contribution)
                contributions.append({
                    "source": self.node_ids[source], "label": self.node_labels[source],
                    "relation": self._relation(int(source), int(target)),
                    "contribution": float(contribution),
                })
            contributions.sort(key=lambda row: (-row["contribution"], row["source"]))
            path = self._representative_path(int(target), seed_nodes, previous)
            relations = tuple(self._relation(left, right) for left, right in zip(path, path[1:]))
            direct_matches = tuple(
                self.node_ids[source] for source in sources if source in seed_nodes
            )
            explanations.append(StandaloneExplanation(
                rank, self.passage_ids[target], float(scores[target]), matched, other,
                structure, tuple(self.node_ids[index] for index in path), relations,
                tuple(self.node_label(index) for index in path),
                direct_matches, tuple(contributions[:5]),
                float(sum(row["contribution"] for row in contributions[5:])),
            ))
        return StandaloneResult(
            [self.documents[index] for index in ranked], explanations,
            links, iterations, residual, converged,
        )

    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query).documents
