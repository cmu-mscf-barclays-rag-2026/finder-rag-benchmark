"""Shared evaluate_rag entry point for BM25, dense, hybrid and MMR runs."""

from __future__ import annotations

import math
import platform
import statistics
import time
from datetime import datetime, timezone
from typing import Callable

from .data import digest, select_queries
from .metrics import (answer_exact_match, answer_token_f1, ndcg_at_k, precision_at_k,
                      recall_at_k, reciprocal_rank, validate_ranking)

EVALUATOR_VERSION = "finder_shared_v1"
AnswerScorer = Callable[[str, str], float]


def collect_run(bundle: dict, retriever, *, method: str = "bm25", top_k: int = 5,
                partition: str = "test", limit: int | None = None, config: dict | None = None,
                build_latency_ms: float | None = None) -> dict:
    """Retriever contract: search(query: str, top_k: int) -> [{doc_id, score}]."""
    validate_ranking([], {}, top_k)
    queries = select_queries(bundle, partition, limit)
    results = []
    for query in queries:
        start = time.perf_counter()
        hits = retriever.search(query["text"], top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        results.append({"query_id": query["query_id"], "doc_ids": [hit["doc_id"] for hit in hits],
                        "scores": [float(hit["score"]) for hit in hits], "latency_ms": latency_ms})
    return {
        "schema_version": 1, "method": method, "bundle_fingerprint": bundle["manifest"]["fingerprint"],
        "partition": partition, "query_ids": [q["query_id"] for q in queries], "top_k": top_k,
        "config": config or {}, "build_latency_ms": build_latency_ms,
        "latency_scope": "retriever.search_only", "timing_protocol": "sequential_one_measurement_per_query",
        "environment": {"python": platform.python_version(), "system": platform.platform(),
                        "machine": platform.machine()},
        "created_at": datetime.now(timezone.utc).isoformat(), "results": results,
    }


def _coverage(doc_ids: list[str], query: dict, documents: dict[str, dict]) -> float:
    """Fraction of normalized gold-reference characters covered, merging overlaps."""
    lengths: dict[str, int] = {}
    intervals: dict[str, list[tuple[int, int]]] = {ref: [] for ref in query["reference_ids"]}
    for doc in documents.values():
        if doc["reference_id"] in intervals:
            lengths[doc["reference_id"]] = doc["reference_length"]
    for doc_id in doc_ids:
        doc = documents[doc_id]
        if doc["reference_id"] in intervals:
            intervals[doc["reference_id"]].append((doc["start"], doc["end"]))
    covered = 0
    for spans in intervals.values():
        last_end = 0
        for start, end in sorted(spans):
            covered += max(0, end - max(start, last_end))
            last_end = max(last_end, end)
    return covered / sum(lengths.values()) if lengths else 0.0


def _mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def evaluate_rag(bundle: dict, run: dict, *, answer_scorer: AnswerScorer | None = None,
                 answer_scorer_name: str | None = None) -> dict:
    """All methods receive the same metrics; unavailable generation scores stay null.

    An answer_scorer receives (prediction, gold_answer), returns [0, 1], and must
    have a stable name/version. Person 3 can plug the team's correctness metric here.
    """
    if run.get("schema_version") != 1:
        raise ValueError("Unsupported run schema version")
    if run.get("bundle_fingerprint") != bundle["manifest"]["fingerprint"]:
        raise ValueError("Run and prepared data have different fingerprints")
    if answer_scorer is not None and not answer_scorer_name:
        raise ValueError("Name/version the answer correctness scorer for reproducibility")
    k = run["top_k"]
    validate_ranking([], {}, k)
    queries = {q["query_id"]: q for q in bundle["queries"]}
    documents = {doc["doc_id"]: doc for doc in bundle["corpus"]}
    expected = run["query_ids"]
    if not expected or len(set(expected)) != len(expected) or set(expected) - set(queries):
        raise ValueError("Run must declare a nonempty, unique set of known query_ids")
    allowed = {q["query_id"] for q in select_queries(bundle, run["partition"])}
    if set(expected) - allowed:
        raise ValueError("Run query IDs do not belong to its declared partition")
    actual = [row["query_id"] for row in run["results"]]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("Supply exactly one result per declared query; use doc_ids=[] for no hits")
    per_query = []
    for result in run["results"]:
        query_id, ranking = result["query_id"], result["doc_ids"]
        if not isinstance(ranking, list) or any(not isinstance(doc_id, str) for doc_id in ranking):
            raise ValueError("doc_ids must be a list of strings")
        if len(ranking) > k or set(ranking) - set(documents):
            raise ValueError("Rankings must use shared corpus IDs and return at most top_k documents")
        grades = bundle["qrels"][query_id]
        validate_ranking(ranking, grades, k)
        scores = result.get("scores")
        if scores is not None and (len(scores) != len(ranking) or any(not math.isfinite(s) for s in scores)):
            raise ValueError("scores must be finite and align with doc_ids")
        latency = result.get("latency_ms")
        if latency is not None and (not math.isfinite(latency) or latency < 0):
            raise ValueError("latency_ms must be finite and nonnegative, or null if unmeasured")
        query = queries[query_id]
        row = {
            "query_id": query_id, "query": query["text"], "category": query["category"],
            "reasoning": query["reasoning"], "type": query["type"], "top_k": k,
            "precision": precision_at_k(ranking, grades, k), "recall": recall_at_k(ranking, grades, k),
            "reciprocal_rank": reciprocal_rank(ranking, grades, k), "ndcg": ndcg_at_k(ranking, grades, k),
            "evidence_coverage": _coverage(ranking, query, documents),
            "first_relevant_rank": next((rank for rank, doc in enumerate(ranking, 1) if grades.get(doc, 0) > 0), None),
            "retrieved_count": len(ranking), "relevant_count": sum(value > 0 for value in grades.values()),
            "latency_ms": latency, "answer_accuracy_exact_match": None,
            "answer_token_f1": None, "answer_correctness": None,
        }
        prediction = result.get("answer")
        if prediction is not None and not isinstance(prediction, str):
            raise ValueError("answer must be a string or null")
        if prediction is not None and query["answer"].strip():
            row["answer_accuracy_exact_match"] = answer_exact_match(prediction, query["answer"])
            row["answer_token_f1"] = answer_token_f1(prediction, query["answer"])
            if answer_scorer is not None:
                correctness = float(answer_scorer(prediction, query["answer"]))
                if not math.isfinite(correctness) or not 0 <= correctness <= 1:
                    raise ValueError("Answer correctness scorer must return a finite value in [0, 1]")
                row["answer_correctness"] = correctness
        per_query.append(row)
    metric_keys = ("precision", "recall", "reciprocal_rank", "ndcg", "evidence_coverage", "retrieved_count",
                   "latency_ms", "answer_accuracy_exact_match", "answer_token_f1", "answer_correctness")
    summary = {key: _mean(per_query, key) for key in metric_keys}
    summary["mrr"] = summary.pop("reciprocal_rank")
    latencies = sorted(row["latency_ms"] for row in per_query if row["latency_ms"] is not None)
    answered_ids = sorted(row["query_id"] for row in per_query if row["answer_accuracy_exact_match"] is not None)
    eligible_answers = sum(bool(queries[qid]["answer"].strip()) for qid in expected)
    summary.update({
        "method": run["method"], "top_k": k, "query_count": len(per_query),
        "bundle_fingerprint": bundle["manifest"]["fingerprint"], "query_set_fingerprint": digest(sorted(expected)),
        "corpus_scope": bundle["manifest"]["corpus_scope"], "partition": run["partition"],
        "evaluator_version": EVALUATOR_VERSION, "answer_scorer": answer_scorer_name if answer_scorer else None,
        "answer_scored_count": len(answered_ids), "answer_eligible_count": eligible_answers,
        "answer_query_set_fingerprint": digest(answered_ids),
        "answer_status": "not_scored" if not answered_ids else "complete" if len(answered_ids) == eligible_answers else "partial",
        "latency_measured_count": len(latencies), "latency_scope": run.get("latency_scope", "unspecified"),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        "build_latency_ms": run.get("build_latency_ms"),
    })
    by_category = []
    for category in sorted({row["category"] for row in per_query}):
        rows = [row for row in per_query if row["category"] == category]
        by_category.append({"category": category, "query_count": len(rows),
                            **{key: _mean(rows, key) for key in metric_keys}})
    return {"summary": summary, "per_query": per_query, "by_category": by_category}
