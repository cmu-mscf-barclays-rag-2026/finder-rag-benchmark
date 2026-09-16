"""Reliable Task A evaluation for dense FinDER retrieval.

FinDER references are deduplicated into a source-passage corpus. Passages are
split for embedding, but chunk scores are max-pooled back to source passages
before ranking. Metrics therefore use the same exact passage relevance labels
for every chunking configuration.

Chunking parameters are compared only on the deterministic development split.
The selected configuration is then evaluated once on held-out test questions.
Answers and reasoning are never indexed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from datasets import Dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from rag import DATASET_ID, RAGSettings, load_finder_records


@dataclass(frozen=True)
class ChunkingConfig:
    """One chunking configuration in the development-set sweep."""

    label: str
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class RetrievalMetrics:
    """Macro passage-level retrieval metrics for one split and top-k."""

    split: str
    label: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    questions: int
    source_passages: int
    chunks: int
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    f1_at_k: float
    index_seconds: float
    latency_ms_per_query: float


DEFAULT_CONFIGS = (
    ChunkingConfig("size=500, overlap=75", 500, 75),
    ChunkingConfig("size=1000, overlap=150 (baseline)", 1_000, 150),
    ChunkingConfig("size=1500, overlap=225", 1_500, 225),
    ChunkingConfig("size=1000, overlap=0", 1_000, 0),
    ChunkingConfig("size=1000, overlap=300", 1_000, 300),
)
DEFAULT_TOP_K = (1, 3, 5, 10)
PRIMARY_K = 5


def normalize_text(text: str) -> str:
    """Collapse whitespace without changing the evidence content."""

    return " ".join(str(text).split())


def build_source_corpus(
    records: Sequence[dict[str, Any]],
) -> tuple[list[str], dict[str, set[int]]]:
    """Build a deduplicated reference corpus and exact passage-level qrels."""

    corpus: list[str] = []
    passage_to_id: dict[str, int] = {}
    qrels: dict[str, set[int]] = {}
    for row in records:
        relevant: set[int] = set()
        for value in row.get("references", []):
            passage = normalize_text(value)
            if not passage:
                continue
            if passage not in passage_to_id:
                passage_to_id[passage] = len(corpus)
                corpus.append(passage)
            relevant.add(passage_to_id[passage])
        row_id = str(row.get("_id", ""))
        if row_id and str(row.get("text", "")).strip() and relevant:
            qrels[row_id] = relevant
    return corpus, qrels


def deterministic_split(row_ids: Sequence[str]) -> np.ndarray:
    """Return the frozen 20% development / 80% held-out test split."""

    return np.asarray(
        [
            "dev"
            if int(hashlib.sha1(row_id.encode()).hexdigest()[:8], 16) % 5 == 0
            else "test"
            for row_id in row_ids
        ]
    )


def split_source_corpus(
    corpus: Sequence[str], config: ChunkingConfig
) -> tuple[list[str], np.ndarray]:
    """Return chunk texts and the source-passage ID of every chunk."""

    documents = [
        Document(page_content=text, metadata={"source_id": source_id})
        for source_id, text in enumerate(corpus)
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return (
        [chunk.page_content for chunk in chunks],
        np.asarray([int(chunk.metadata["source_id"]) for chunk in chunks]),
    )


def query_metrics(
    ranking: Sequence[int], relevant: set[int], k: int
) -> tuple[float, float, float]:
    """Return exact passage-level Precision@k, Recall@k, and Hit@k."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        raise ValueError("each query must have at least one relevant passage")
    hits = sum(int(source_id in relevant) for source_id in ranking[:k])
    return hits / k, hits / len(relevant), float(hits > 0)


def _harmonic_mean(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_rankings(
    rankings: np.ndarray,
    qrels: Sequence[set[int]],
    config: ChunkingConfig,
    split: str,
    source_passages: int,
    chunks: int,
    index_seconds: float,
    latency_ms_per_query: float,
    top_k_values: Iterable[int] = DEFAULT_TOP_K,
) -> list[RetrievalMetrics]:
    """Macro-average passage metrics over one query split."""

    if len(rankings) != len(qrels) or not qrels:
        raise ValueError("rankings and non-empty qrels must have the same length")
    output: list[RetrievalMetrics] = []
    for k in sorted(set(top_k_values)):
        values = [query_metrics(ranking, relevant, k) for ranking, relevant in zip(rankings, qrels)]
        precision = statistics.fmean(value[0] for value in values)
        recall = statistics.fmean(value[1] for value in values)
        output.append(
            RetrievalMetrics(
                split=split,
                label=config.label,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                top_k=k,
                questions=len(qrels),
                source_passages=source_passages,
                chunks=chunks,
                precision_at_k=precision,
                recall_at_k=recall,
                hit_rate_at_k=statistics.fmean(value[2] for value in values),
                f1_at_k=_harmonic_mean(precision, recall),
                index_seconds=index_seconds,
                latency_ms_per_query=latency_ms_per_query,
            )
        )
    return output


def rank_sources(
    query_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_source_ids: np.ndarray,
    source_count: int,
    depth: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, float]:
    """Rank sources by the maximum cosine score of any of their chunks."""

    if depth < 1 or depth > source_count:
        raise ValueError("depth must be between 1 and source_count")
    rankings: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, len(query_embeddings), batch_size):
        scores = query_embeddings[offset : offset + batch_size] @ chunk_embeddings.T
        for chunk_scores in scores:
            source_scores = np.full(source_count, -np.inf, dtype=np.float32)
            np.maximum.at(source_scores, chunk_source_ids, chunk_scores)
            candidate_ids = np.argpartition(source_scores, -depth)[-depth:]
            order = np.argsort(source_scores[candidate_ids])[::-1]
            rankings.append(candidate_ids[order])
    elapsed_ms = 1_000 * (time.perf_counter() - started) / len(query_embeddings)
    return np.vstack(rankings), elapsed_ms


def _encode(
    model: SentenceTransformer, texts: Sequence[str], batch_size: int
) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)


def run_reliable_evaluation(
    records: Sequence[dict[str, Any]],
    embedding_model: str = RAGSettings().embedding_model,
    embedding_device: str = RAGSettings().embedding_device,
    configs: Sequence[ChunkingConfig] = DEFAULT_CONFIGS,
    top_k_values: Sequence[int] = DEFAULT_TOP_K,
    primary_k: int = PRIMARY_K,
    batch_size: int = 128,
) -> tuple[list[RetrievalMetrics], list[RetrievalMetrics], dict[str, Any]]:
    """Tune chunking on development queries and evaluate once on held-out test."""

    corpus, qrels_by_id = build_source_corpus(records)
    usable = [row for row in records if str(row.get("_id")) in qrels_by_id]
    row_ids = [str(row["_id"]) for row in usable]
    questions = [str(row["text"]) for row in usable]
    splits = deterministic_split(row_ids)
    dev_indices = np.flatnonzero(splits == "dev")
    test_indices = np.flatnonzero(splits == "test")
    if not len(dev_indices) or not len(test_indices):
        raise ValueError("both development and test queries are required")

    model = SentenceTransformer(embedding_model, device=embedding_device)
    query_started = time.perf_counter()
    all_query_embeddings = _encode(model, questions, batch_size)
    query_embedding_ms = 1_000 * (time.perf_counter() - query_started) / len(questions)
    dev_embeddings = all_query_embeddings[dev_indices]
    test_embeddings = all_query_embeddings[test_indices]
    dev_qrels = [qrels_by_id[row_ids[index]] for index in dev_indices]
    test_qrels = [qrels_by_id[row_ids[index]] for index in test_indices]
    depth = max(top_k_values)

    dev_results: list[RetrievalMetrics] = []
    best_key: tuple[float, float, float] | None = None
    selected: ChunkingConfig | None = None
    selected_chunk_embeddings: np.ndarray | None = None
    selected_chunk_source_ids: np.ndarray | None = None
    selected_index_seconds = 0.0

    for config in configs:
        print(f"Development sweep: {config.label}", flush=True)
        chunk_texts, chunk_source_ids = split_source_corpus(corpus, config)
        index_started = time.perf_counter()
        chunk_embeddings = _encode(model, chunk_texts, batch_size)
        index_seconds = time.perf_counter() - index_started
        dev_rankings, search_ms = rank_sources(
            dev_embeddings,
            chunk_embeddings,
            chunk_source_ids,
            len(corpus),
            depth,
            batch_size,
        )
        config_results = evaluate_rankings(
            dev_rankings,
            dev_qrels,
            config,
            "dev",
            len(corpus),
            len(chunk_texts),
            index_seconds,
            query_embedding_ms + search_ms,
            top_k_values,
        )
        dev_results.extend(config_results)
        primary = next(result for result in config_results if result.top_k == primary_k)
        # Predeclared objective: maximize dev F1@5, then Recall@5, then fewer chunks.
        key = (primary.f1_at_k, primary.recall_at_k, -float(primary.chunks))
        if best_key is None or key > best_key:
            best_key = key
            selected = config
            selected_chunk_embeddings = chunk_embeddings
            selected_chunk_source_ids = chunk_source_ids
            selected_index_seconds = index_seconds

    assert selected is not None
    assert selected_chunk_embeddings is not None
    assert selected_chunk_source_ids is not None
    print(f"Held-out test: {selected.label}", flush=True)
    test_rankings, test_search_ms = rank_sources(
        test_embeddings,
        selected_chunk_embeddings,
        selected_chunk_source_ids,
        len(corpus),
        depth,
        batch_size,
    )
    test_results = evaluate_rankings(
        test_rankings,
        test_qrels,
        selected,
        "test",
        len(corpus),
        len(selected_chunk_embeddings),
        selected_index_seconds,
        query_embedding_ms + test_search_ms,
        top_k_values,
    )
    summary = {
        "dataset_id": f"FinDER-{len(records)}",
        "dataset_source": DATASET_ID,
        "dataset_content_sha256": dataset_fingerprint(records),
        "corpus_definition": "whitespace-normalized deduplicated union of FinDER references",
        "source_passages": len(corpus),
        "split_id": "sha1-mod5-dev-v1",
        "split_definition": "dev iff int(sha1(_id)[:8], 16) modulo 5 equals 0; test otherwise",
        "dev_queries": len(dev_indices),
        "test_queries": len(test_indices),
        "embedding_model": embedding_model,
        "embedding_device": embedding_device,
        "similarity": "cosine (L2-normalized embeddings)",
        "chunk_score_aggregation": "maximum chunk score per source passage",
        "selection_rule": f"maximum development F1@{primary_k}; tie-break Recall@{primary_k}, then fewer chunks",
        "selected_config": asdict(selected),
        "primary_k": primary_k,
        "top_k_values": list(top_k_values),
        "query_embedding_ms_per_query": query_embedding_ms,
    }
    return dev_results, test_results, summary


def dataset_fingerprint(records: Sequence[dict[str, Any]]) -> str:
    """Hash IDs, questions, and normalized references for provenance."""

    digest = hashlib.sha256()
    for row in sorted(records, key=lambda item: str(item.get("_id", ""))):
        payload = {
            "_id": str(row.get("_id", "")),
            "text": str(row.get("text", "")),
            "references": [normalize_text(value) for value in row.get("references", [])],
        }
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_records(dataset_file: Path | None, sample_size: int, seed: int) -> list[dict[str, Any]]:
    """Load FinDER from Hugging Face or an optional local Arrow/Parquet file."""

    if dataset_file is None:
        return load_finder_records(sample_size or 10_000_000, seed)
    if dataset_file.suffix.lower() == ".arrow":
        dataset = Dataset.from_file(str(dataset_file))
    elif dataset_file.suffix.lower() in {".parquet", ".pq"}:
        dataset = Dataset.from_parquet(str(dataset_file))
    else:
        raise ValueError("dataset-file must be an .arrow or .parquet file")
    if sample_size and sample_size < len(dataset):
        indices = np.random.default_rng(seed).permutation(len(dataset))[:sample_size]
        return [dataset[int(index)] for index in indices]
    return dataset.to_list()


def _write_csv(results: Sequence[RetrievalMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="0 uses all 5,703 rows (required for final reported results)",
    )
    parser.add_argument("--seed", type=int, default=42, help="only affects optional subsampling")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    records = load_records(args.dataset_file, args.sample_size, args.seed)
    if len(records) != 5_703:
        print(
            f"WARNING: using {len(records)} rows; results are a smoke test, not the final benchmark.",
            flush=True,
        )
    dev_results, test_results, summary = run_reliable_evaluation(
        records, embedding_device=args.device, batch_size=args.batch_size
    )
    _write_csv(dev_results, args.output_dir / "task_a_dense_sweep.csv")
    _write_csv(test_results, args.output_dir / "task_a_test_metrics.csv")
    _write_json([asdict(result) for result in dev_results], args.output_dir / "task_a_dense_sweep.json")
    _write_json(summary, args.output_dir / "task_a_run_summary.json")
    print(f"Wrote Task A results to {args.output_dir}")


if __name__ == "__main__":
    main()
