"""Evaluate independent graph retrieval and its one-step matching ablation.

No embedding model, vector store, or LLM is loaded by this evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from legal_data import (
    LEGAL_DATASET_ID, legal_records_to_documents, legal_records_to_questions,
    load_legal_corpus, load_legal_questions,
)
from standalone_graph import StandaloneGraphRetriever


def ranking_metrics(ranking: list[str], gold_id: str, k: int) -> dict[str, float]:
    if k < 1 or len(ranking) != len(set(ranking)):
        raise ValueError("k must be positive and rankings must have unique passage IDs")
    rank = next((i for i, value in enumerate(ranking[:k], 1) if value == gold_id), None)
    hit = float(rank is not None)
    return {
        "precision_at_k": hit / k,
        "recall_at_k": hit,
        "hit_rate_at_k": hit,
        "mrr_at_k": 1 / rank if rank else 0.0,
        "ndcg_at_k": 1 / math.log2(rank + 1) if rank else 0.0,
    }


def evaluate(corpus: list[dict], qa: list[dict], output_dir: Path) -> dict[str, Any]:
    documents = legal_records_to_documents(corpus)
    questions = legal_records_to_questions(qa)
    if not documents or not questions:
        raise ValueError("Non-empty corpus and questions required")
    passage_ids = {str(doc.metadata["passage_id"]) for doc in documents}
    if any(q.relevant_passage_id not in passage_ids for q in questions):
        raise ValueError("A gold passage is missing from the corpus")
    started = time.perf_counter()
    retriever = StandaloneGraphRetriever(documents, top_k=10)
    build_seconds = time.perf_counter() - started
    metrics, traces = [], []
    rankings: dict[str, list[list[str]]] = {}
    for name, propagate in (("one_step_concept_match", False), ("standalone_graph", True)):
        results, elapsed = [], []
        for question in questions:
            started = time.perf_counter()
            result = retriever.retrieve(question.question, propagate=propagate)
            elapsed.append((time.perf_counter() - started) * 1_000)
            results.append(result)
            if propagate:
                traces.append({
                    "question_id": question.question_id,
                    "question": question.question,
                    "gold_passage_id": question.relevant_passage_id,
                    "query_links": result.query_links,
                    "iterations": result.iterations,
                    "residual": result.residual,
                    "converged": result.converged,
                    "trace": [item.to_dict() for item in result.explanations],
                })
        rankings[name] = [[item.node_id for item in result.explanations] for result in results]
        for k in (1, 3, 5, 10):
            values = [ranking_metrics(ranking, q.relevant_passage_id, k)
                      for ranking, q in zip(rankings[name], questions)]
            explanations = [item for result in results for item in result.explanations[:k]]
            metrics.append({
                "method": name, "top_k": k, "questions": len(questions),
                **{key: statistics.fmean(value[key] for value in values) for key in values[0]},
                "latency_ms_per_query_including_explanations": statistics.fmean(elapsed),
                "explanation_coverage": (
                    sum(bool(item.path) for item in explanations) / len(explanations)
                    if explanations else 0.0
                ),
            })
    wins, losses = [], []
    for index, question in enumerate(questions):
        direct = rankings["one_step_concept_match"][index]
        graph = rankings["standalone_graph"][index]
        direct_hit = question.relevant_passage_id in direct[:5]
        graph_hit = question.relevant_passage_id in graph[:5]
        if graph_hit and not direct_hit:
            wins.append(question.question_id)
        if direct_hit and not graph_hit:
            losses.append(question.question_id)
        traces[index]["one_step_ranking"] = direct
    corpus_payload = sorted([
        (str(doc.metadata["passage_id"]), str(doc.metadata["title"]), doc.page_content)
        for doc in documents
    ])
    summary = {
        "dataset_id": LEGAL_DATASET_ID,
        "corpus_passages": len(documents), "questions": len(questions),
        "corpus_sha256": hashlib.sha256(json.dumps(corpus_payload, ensure_ascii=False).encode()).hexdigest(),
        "query_sha256": hashlib.sha256(json.dumps([
            (q.question_id, q.question, q.relevant_passage_id) for q in questions
        ], ensure_ascii=False).encode()).hexdigest(),
        "retriever": "standalone concept/passage Personalized PageRank",
        "embedding_model": None, "dense_retrieval_used": False,
        "settings": asdict(retriever.settings), "graph": retriever.stats,
        "build_seconds": build_seconds,
        "converged_queries": sum(row["converged"] for row in traces),
        "unmatched_queries": sum(not row["query_links"] for row in traces),
        "graph_vs_one_step_at_5": {"win_ids": wins, "loss_ids": losses,
                                   "ties": len(questions) - len(wins) - len(losses)},
        "protocol": "Exploratory public test evaluation. No parameter sweep or fitting to qrels. These questions were examined in prior experiments; this is not an unseen holdout.",
        "timing": "Sequential CPU queries; includes concept matching, ranking, and explanation construction. Index building excluded.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "standalone_graph_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output_dir / "standalone_graph_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (output_dir / "standalone_graph_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in metrics:
        if row["top_k"] == 5:
            print(json.dumps(row), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-sample-size", type=int, default=0, help="0 evaluates all 100 questions")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    summary = evaluate(load_legal_corpus(), load_legal_questions(args.query_sample_size), args.output_dir)
    print(f"Wrote {summary['questions']}-query results to {args.output_dir}")


if __name__ == "__main__":
    main()
