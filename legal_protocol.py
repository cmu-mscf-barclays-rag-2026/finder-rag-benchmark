"""Team Legal RAG Bench protocol (``legal_rag/`` on ``feature/person2-bm25``).

Shared by the graph evaluators so their results are directly comparable with
the team's BM25, dense, and hybrid runs:

- Corpus: all 4,876 passages, passage text only
  (``legal_records_to_documents(..., text_only=True)``).
- Split: questions grouped by SHA-256 of the gold passage's text, groups
  sorted and shuffled with seed 42, dev filled to 20%. This gives 20 dev and
  80 held-out questions and is checked against the published split.
- Metrics: legal_rag names at K = 1, 3, 5, 10, 20, credited at the first gold
  rank. Whole passages are retrieved, so coverage and all-evidence hit equal
  Hit.
- Files: ``per_query.csv`` and ``test_metrics.csv`` use legal_rag's columns.
- Latency: 3 warmups per method, then shuffled, interleaved repeats of all
  held-out queries.
- Uncertainty: paired Hit@5 percentile bootstrap over gold-passage groups
  (10,000 resamples, seed 42).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from legal_data import LegalQuestion


KS = (1, 3, 5, 10, 20)
METRICS = ("recall", "hit_rate", "all_evidence_hit", "evidence_coverage", "mrr", "ndcg")
SPLIT_ID = "group-gold-text-shuffle42-20pct-v1"
# legal_rag/results/initial/split.json on feature/person2-bm25.
PUBLISHED_DEV_IDS = {"2", "4", "9", "10", "13", "24", "30", "34", "36", "51", "52", "53",
                     "55", "64", "76", "80", "85", "86", "91", "96"}


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


def team_metrics(ranking: list[str], gold_id: str, k: int) -> dict[str, float]:
    """legal_rag per-query metrics for whole-passage rankings."""
    values = ranking_metrics(ranking, gold_id, k)
    hit = values["hit_rate_at_k"]
    return {"recall": hit, "hit_rate": hit, "all_evidence_hit": hit, "evidence_coverage": hit,
            "mrr": values["mrr_at_k"], "ndcg": values["ndcg_at_k"]}


def split_queries(qa: Sequence[dict], corpus: Sequence[dict], seed: int = 42) -> tuple[set[str], set[str]]:
    """Same rule as legal_rag/run.py: whole gold-text groups fill dev to 20%."""
    text = {str(p["id"]): p["text"] for p in corpus}
    groups: dict[str, list[str]] = {}
    for q in qa:
        key = hashlib.sha256(text[str(q["relevant_passage_id"])].encode()).hexdigest()
        groups.setdefault(key, []).append(str(q["id"]))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    dev: set[str] = set()
    for key in keys:
        if len(dev) >= math.ceil(len(qa) * 0.2):
            break
        dev.update(groups[key])
    test = {str(q["id"]) for q in qa} - dev
    if len(qa) == 100 and dev != PUBLISHED_DEV_IDS:
        raise ValueError("Split differs from the published legal_rag split")
    return dev, test


def paired_bootstrap(deltas: list[float], groups: list[str], seed: int = 42) -> tuple[float, float]:
    """Percentile 95% interval, resampling gold-passage groups (legal_rag analyze_results.py)."""
    clusters: dict[str, list[float]] = {}
    for delta, group in zip(deltas, groups):
        clusters.setdefault(group, []).append(delta)
    values = list(clusters.values())
    rng = random.Random(seed)
    means = []
    for _ in range(10_000):
        sample = [d for cluster in rng.choices(values, k=len(values)) for d in cluster]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[249], means[9749]


def load_reference_hits(path: Path) -> dict[str, dict[str, float]]:
    """Held-out Hit@5 per method and question from a legal_rag per_query.csv."""
    hits: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig") as handle:  # tolerate a PowerShell BOM
        for row in csv.DictReader(handle):
            if row["k"] == "5":
                hits.setdefault(f"legal_rag_{row['method']}", {})[row["query_id"]] = float(row["hit_rate"])
    return hits


def warm_latency(
    searchers: dict[str, Callable[[str], Any]],
    dev: Sequence[LegalQuestion],
    test: Sequence[LegalQuestion],
    repeats: int = 3,
) -> dict[str, list[float]]:
    """legal_rag timing: 4 BLAS threads, 3 dev warmups per method, then shuffled
    interleaved repeats of all held-out queries."""
    timings: dict[str, list[float]] = {name: [] for name in searchers}
    with threadpool_limits(limits=4):
        for search in searchers.values():
            for question in dev[:3]:
                search(question.question)
        for repeat in range(repeats):
            jobs = [(name, question) for name in searchers for question in test]
            random.Random(100 + repeat).shuffle(jobs)
            for name, question in jobs:
                started = time.perf_counter()
                searchers[name](question.question)
                timings[name].append((time.perf_counter() - started) * 1_000)
    return timings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    output_dir: Path,
    *,
    questions: Sequence[LegalQuestion],
    dev_ids: set[str],
    test_ids: set[str],
    rankings: dict[str, dict[str, list[str]]],
    config_ids: dict[str, str],
    latency: dict[str, list[float]],
    baseline: str | Sequence[str],
    references: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Write split.json, dev_metrics.csv, per_query.csv, test_metrics.csv, and
    paired_hit5_intervals.csv. ``rankings`` maps method -> question ID -> passage IDs;
    every method is compared with each baseline method and each reference."""
    baselines = [baseline] if isinstance(baseline, str) else list(baseline)
    by_id = {q.question_id: q for q in questions}
    dev, test = sorted(dev_ids, key=int), sorted(test_ids, key=int)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(
        json.dumps({"dev": [int(i) for i in dev], "test": [int(i) for i in test]}, indent=2) + "\n")
    dev_rows, per_query, summaries = [], [], []
    for method, ranked in rankings.items():
        values = [team_metrics(ranked[qid], by_id[qid].relevant_passage_id, 5) for qid in dev]
        dev_rows.append({"method": method, "config_id": config_ids[method], "k": 5,
                         "n_queries": len(values),
                         **{m: statistics.fmean(v[m] for v in values) for m in METRICS}})
        for qid in test:
            for k in KS:
                per_query.append({"method": method, "query_id": qid, "k": k,
                                  **team_metrics(ranked[qid], by_id[qid].relevant_passage_id, k)})
        times = latency.get(method, [])
        for k in KS:
            rows = [r for r in per_query if r["method"] == method and r["k"] == k]
            summaries.append({
                "method": method, "config_id": config_ids[method], "k": k, "n_queries": len(rows),
                **{m: statistics.fmean(r[m] for r in rows) for m in METRICS},
                "latency_mean_ms": statistics.fmean(times) if times else "",
                "latency_p95_ms": float(np.percentile(times, 95)) if times else "",
            })
    hit5 = {method: {qid: float(by_id[qid].relevant_passage_id in ranked[qid][:5]) for qid in test}
            for method, ranked in rankings.items()}
    references = references or {}
    for name, hits in references.items():
        if set(hits) != set(test):
            raise ValueError(f"{name} per-query results do not cover the same held-out questions")
    golds = [by_id[qid].relevant_passage_id for qid in test]
    paired = []
    for method in rankings:
        for other, other_hits in {**{name: hit5[name] for name in baselines}, **references}.items():
            if method == other:
                continue
            deltas = [hit5[method][qid] - other_hits[qid] for qid in test]
            low, high = paired_bootstrap(deltas, golds)
            paired.append({"comparison": f"{method} minus {other}", "n_queries": len(deltas),
                           "hit5_difference": statistics.fmean(deltas),
                           "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                           "paired_bootstrap_95pct_low": low, "paired_bootstrap_95pct_high": high})
    write_csv(output_dir / "dev_metrics.csv", dev_rows)
    write_csv(output_dir / "per_query.csv", per_query)
    write_csv(output_dir / "test_metrics.csv", summaries)
    write_csv(output_dir / "paired_hit5_intervals.csv", paired)
    for row in summaries:
        if row["k"] == 5:
            ms = f"{row['latency_mean_ms']:.1f} ms" if row["latency_mean_ms"] != "" else "not timed"
            print(f"{row['method']:24s} recall@5={row['recall']:.4f} mrr@5={row['mrr']:.4f} "
                  f"ndcg@5={row['ndcg']:.4f} {ms}")
    for row in paired:
        print(f"{row['comparison']:50s} {row['hit5_difference']:+.4f} "
              f"[{row['paired_bootstrap_95pct_low']:+.4f}, {row['paired_bootstrap_95pct_high']:+.4f}] "
              f"{row['wins']}W/{row['losses']}L")
    return {"test_metrics": summaries, "paired": paired}
