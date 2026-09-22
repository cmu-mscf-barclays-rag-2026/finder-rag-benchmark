"""Evaluate dense retrieval versus interpretable graph expansion.

This evaluator uses the official Legal RAG Bench test corpus and questions.  It
does not tune on those 100 questions: graph construction and weights are fixed
before labels are read.  Both methods use exactly the same embeddings and dense
candidate search, isolating the contribution of graph expansion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from graph_rag import InterpretableDocumentGraph, RetrievalExplanation
from legal_data import (
    LEGAL_DATASET_ID,
    legal_records_to_documents,
    legal_records_to_questions,
    load_legal_corpus,
    load_legal_questions,
)
from rag import RAGSettings


DEFAULT_TOP_K = (1, 3, 5, 10)
PRIMARY_K = 5


@dataclass(frozen=True)
class GraphEvaluationMetrics:
    method: str
    top_k: int
    questions: int
    corpus_passages: int
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    latency_ms_per_query: float
    explanation_coverage: float
    graph_influenced_result_fraction: float
    graph_only_result_fraction: float
    graph_path_gold_rate: float


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


def dense_rankings(
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    depth: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return top dense indices, their cosine scores, and search latency."""

    if not 1 <= depth <= len(passage_embeddings):
        raise ValueError("depth must be between 1 and the corpus size")
    started = time.perf_counter()
    scores = query_embeddings @ passage_embeddings.T
    candidate_indices = np.argpartition(scores, -depth, axis=1)[:, -depth:]
    candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    rankings = np.take_along_axis(candidate_indices, order, axis=1)
    ranked_scores = np.take_along_axis(candidate_scores, order, axis=1)
    latency_ms = 1_000 * (time.perf_counter() - started) / len(query_embeddings)
    return rankings, ranked_scores, latency_ms


def _rank_of_gold(ranking: Sequence[str], gold: str, k: int) -> int | None:
    for rank, passage_id in enumerate(ranking[:k], start=1):
        if passage_id == gold:
            return rank
    return None


def evaluate_method(
    *,
    method: str,
    rankings: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    top_k_values: Sequence[int],
    corpus_size: int,
    latency_ms_per_query: float,
    traces: Sequence[Sequence[RetrievalExplanation]] | None = None,
) -> list[GraphEvaluationMetrics]:
    """Compute exact single-qrel retrieval and interpretability metrics."""

    if len(rankings) != len(gold_ids) or not rankings:
        raise ValueError("rankings and non-empty gold_ids must have equal length")
    output = []
    for k in sorted(set(top_k_values)):
        ranks = [
            _rank_of_gold(ranking, gold, k)
            for ranking, gold in zip(rankings, gold_ids)
        ]
        hits = [float(rank is not None) for rank in ranks]
        reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in ranks]
        discounted_gains = [
            0.0 if rank is None else 1.0 / math.log2(rank + 1) for rank in ranks
        ]

        if traces is None:
            explanation_coverage = 1.0
            graph_influenced_result_fraction = 0.0
            graph_only_result_fraction = 0.0
            graph_path_gold_rate = 0.0
        else:
            selected_traces = [list(items[:k]) for items in traces]
            expected = sum(min(k, len(ranking)) for ranking in rankings)
            explained = sum(
                int(bool(item.relation and item.evidence))
                for items in selected_traces
                for item in items
            )
            graph_influenced_items = sum(
                int(item.hop > 0) for items in selected_traces for item in items
            )
            graph_only_items = sum(
                int(item.hop > 0 and item.dense_score is None)
                for items in selected_traces
                for item in items
            )
            explanation_coverage = explained / expected if expected else 0.0
            graph_influenced_result_fraction = (
                graph_influenced_items / expected if expected else 0.0
            )
            graph_only_result_fraction = graph_only_items / expected if expected else 0.0
            graph_path_gold = 0
            for items, gold in zip(selected_traces, gold_ids):
                graph_path_gold += int(
                    any(item.node_id == gold and item.hop > 0 for item in items)
                )
            graph_path_gold_rate = graph_path_gold / len(gold_ids)

        hit_rate = statistics.fmean(hits)
        output.append(
            GraphEvaluationMetrics(
                method=method,
                top_k=k,
                questions=len(gold_ids),
                corpus_passages=corpus_size,
                precision_at_k=hit_rate / k,
                recall_at_k=hit_rate,
                hit_rate_at_k=hit_rate,
                mrr_at_k=statistics.fmean(reciprocal_ranks),
                ndcg_at_k=statistics.fmean(discounted_gains),
                latency_ms_per_query=latency_ms_per_query,
                explanation_coverage=explanation_coverage,
                graph_influenced_result_fraction=graph_influenced_result_fraction,
                graph_only_result_fraction=graph_only_result_fraction,
                graph_path_gold_rate=graph_path_gold_rate,
            )
        )
    return output


def compare_rankings(
    dense: Sequence[Sequence[str]],
    graph: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    k: int,
) -> dict[str, int]:
    """Count query-level graph wins, losses, and unchanged outcomes."""

    wins = losses = ties = 0
    for dense_ranking, graph_ranking, gold in zip(dense, graph, gold_ids):
        dense_hit = _rank_of_gold(dense_ranking, gold, k) is not None
        graph_hit = _rank_of_gold(graph_ranking, gold, k) is not None
        if graph_hit and not dense_hit:
            wins += 1
        elif dense_hit and not graph_hit:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def run_graph_evaluation(
    corpus_records: Sequence[dict[str, Any]],
    question_records: Sequence[dict[str, Any]],
    *,
    embedding_model: str = RAGSettings().embedding_model,
    embedding_device: str = RAGSettings().embedding_device,
    batch_size: int = 128,
    seed_k: int = 10,
    graph_weight: float = 0.35,
    top_k_values: Sequence[int] = DEFAULT_TOP_K,
) -> tuple[list[GraphEvaluationMetrics], dict[str, Any], list[dict[str, Any]]]:
    """Run a controlled dense-versus-graph comparison."""

    depth = max(max(top_k_values), seed_k)
    documents = legal_records_to_documents(corpus_records)
    questions = legal_records_to_questions(question_records)
    if not documents or not questions:
        raise ValueError("Both corpus passages and questions are required")

    passage_ids = [str(document.metadata["passage_id"]) for document in documents]
    passage_id_set = set(passage_ids)
    missing = sorted(
        {item.relevant_passage_id for item in questions} - passage_id_set
    )
    if missing:
        raise ValueError(f"Gold passages missing from corpus: {missing[:5]}")

    graph_started = time.perf_counter()
    graph = InterpretableDocumentGraph(documents)
    graph_build_seconds = time.perf_counter() - graph_started

    model = SentenceTransformer(embedding_model, device=embedding_device)
    passage_texts = [
        f"{document.metadata.get('title', '')}\n{document.page_content}".strip()
        for document in documents
    ]
    passage_started = time.perf_counter()
    passage_embeddings = _encode(model, passage_texts, batch_size)
    passage_embedding_seconds = time.perf_counter() - passage_started

    query_started = time.perf_counter()
    query_embeddings = _encode(
        model, [item.question for item in questions], batch_size
    )
    query_embedding_ms = 1_000 * (time.perf_counter() - query_started) / len(questions)
    dense_indices, dense_scores, dense_search_ms = dense_rankings(
        query_embeddings, passage_embeddings, depth
    )
    dense_ids = [
        [passage_ids[int(index)] for index in ranking]
        for ranking in dense_indices
    ]

    graph_started = time.perf_counter()
    graph_ids: list[list[str]] = []
    graph_traces: list[list[RetrievalExplanation]] = []
    trace_rows: list[dict[str, Any]] = []
    for question, indices, scores in zip(questions, dense_indices, dense_scores):
        seeds = [
            (passage_ids[int(index)], float(score))
            for index, score in zip(indices[:seed_k], scores[:seed_k])
        ]
        result = graph.expand_seeds(
            seeds, top_k=depth, graph_weight=graph_weight
        )
        ids = [item.node_id for item in result.explanations]
        graph_ids.append(ids)
        graph_traces.append(result.explanations)
        trace_rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "gold_passage_id": question.relevant_passage_id,
                "dense_ranking": dense_ids[len(trace_rows)],
                "graph_ranking": ids,
                "graph_trace": [item.to_dict() for item in result.explanations],
            }
        )
    graph_expansion_ms = 1_000 * (time.perf_counter() - graph_started) / len(questions)

    gold_ids = [item.relevant_passage_id for item in questions]
    dense_latency = query_embedding_ms + dense_search_ms
    graph_latency = dense_latency + graph_expansion_ms
    metrics = evaluate_method(
        method="dense_cosine",
        rankings=dense_ids,
        gold_ids=gold_ids,
        top_k_values=top_k_values,
        corpus_size=len(documents),
        latency_ms_per_query=dense_latency,
    )
    metrics.extend(
        evaluate_method(
            method="dense_plus_interpretable_graph",
            rankings=graph_ids,
            gold_ids=gold_ids,
            top_k_values=top_k_values,
            corpus_size=len(documents),
            latency_ms_per_query=graph_latency,
            traces=graph_traces,
        )
    )

    primary_dense = next(
        item for item in metrics if item.method == "dense_cosine" and item.top_k == PRIMARY_K
    )
    primary_graph = next(
        item
        for item in metrics
        if item.method == "dense_plus_interpretable_graph" and item.top_k == PRIMARY_K
    )
    summary = {
        "dataset_id": LEGAL_DATASET_ID,
        "protocol": "official test corpus; fixed graph configuration; no label-based tuning",
        "questions": len(questions),
        "corpus_passages": len(documents),
        "embedding_model": embedding_model,
        "embedding_device": embedding_device,
        "passage_representation": "title + benchmark passage text + footnotes",
        "seed_k": seed_k,
        "graph_weight": graph_weight,
        "graph": asdict(graph.stats),
        "graph_build_seconds": graph_build_seconds,
        "passage_embedding_seconds": passage_embedding_seconds,
        "query_embedding_ms_per_query": query_embedding_ms,
        "dense_search_ms_per_query": dense_search_ms,
        "graph_expansion_ms_per_query": graph_expansion_ms,
        "primary_k": PRIMARY_K,
        "dense_hit_rate_at_5": primary_dense.hit_rate_at_k,
        "graph_hit_rate_at_5": primary_graph.hit_rate_at_k,
        "absolute_hit_rate_lift_at_5": (
            primary_graph.hit_rate_at_k - primary_dense.hit_rate_at_k
        ),
        "query_level_comparison_at_5": compare_rankings(
            dense_ids, graph_ids, gold_ids, PRIMARY_K
        ),
        "warning": (
            "Legal RAG Bench has one public test split. Do not tune graph parameters "
            "on these scores and report them as an unbiased final test."
        ),
    }
    return metrics, summary, trace_rows


def _write_outputs(
    metrics: Sequence[GraphEvaluationMetrics],
    summary: dict[str, Any],
    traces: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [asdict(item) for item in metrics]
    with (output_dir / "task_b_graph_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / "task_b_graph_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (output_dir / "task_b_graph_traces.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-k", type=int, default=10)
    parser.add_argument("--graph-weight", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    corpus = load_legal_corpus()
    question_rows = load_legal_questions(args.query_sample_size, args.seed)
    if len(question_rows) != 100:
        print(
            f"WARNING: using {len(question_rows)} queries; this is a smoke test, not "
            "the official full benchmark.",
            flush=True,
        )
    metrics, summary, traces = run_graph_evaluation(
        corpus,
        question_rows,
        embedding_device=args.device,
        batch_size=args.batch_size,
        seed_k=args.seed_k,
        graph_weight=args.graph_weight,
    )
    _write_outputs(metrics, summary, traces, args.output_dir)
    print(f"Wrote graph evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
