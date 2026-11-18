"""Independently check saved rankings and calculate paired test comparisons."""

import argparse
from collections import defaultdict
import csv
import json
import gzip
import math
from pathlib import Path
import statistics

import numpy as np

from benchmark_core import save_json, write_csv


ROOT = Path(__file__).resolve().parent


def independently_score(ranking, gold, k):
    retrieved = set(ranking[:k])
    relevant = set(gold)
    found = retrieved & relevant
    first = [i + 1 for i, pid in enumerate(ranking[:k]) if pid in relevant]
    discounts = [1 / math.log2(i + 2) for i in range(k)]
    gains = [gold.get(pid, 0) for pid in ranking[:k]]
    ideal = sorted(gold.values(), reverse=True)[:k]
    return {
        "hit": float(bool(found)), "recall": len(found) / len(relevant),
        "precision": len(found) / k, "mrr": 1 / min(first) if first else 0,
        "ndcg": sum(g * d for g, d in zip(gains, discounts)) /
                sum(g * d for g, d in zip(ideal, discounts)),
        "all_judged_hit": float(relevant <= retrieved),
        "complete_page_hit": float(any(gold.get(pid) == 2 for pid in retrieved))
                             if 2 in gold.values() else None,
    }


def audit():
    results = ROOT / "results"
    ranks = json.loads((results / "rankings.json").read_text(encoding="utf-8"))
    raw_gold = json.loads((results / "qrels.json").read_text(encoding="utf-8"))
    gold = {int(q): {int(p): g for p, g in labels.items()} for q, labels in raw_gold.items()}
    with (results / "splits.csv").open(encoding="utf-8") as source:
        splits = {int(r["query_id"]): r for r in csv.DictReader(source)}
    group_splits = defaultdict(set)
    for row in splits.values():
        group_splits[row["group"]].add(row["split"])
    assert all(len(values) == 1 for values in group_splits.values())
    assert len(ranks) == len(splits) == len(gold) == 1854
    reference_methods = set(next(iter(ranks.values())))
    assert len(reference_methods) == 16
    checked = 0
    test = defaultdict(dict)
    with gzip.open(results / "per_query.csv.gz", "rt", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            qid, k = int(row["query_id"]), int(row["k"])
            ranking = ranks[str(qid)][row["method"]]
            assert len(ranking) == len(set(ranking)) == 20
            assert set(ranks[str(qid)]) == reference_methods
            expected = independently_score(ranking, gold[qid], k)
            for name, value in expected.items():
                if value is None:
                    assert row[name] == ""
                else:
                    assert abs(float(row[name]) - value) < 1e-11, (qid, name)
            assert splits[qid]["split"] == row["split"]
            if row["split"] == "test" and row["language"] == "english":
                test[row["method"], k][qid] = expected
            checked += 1
    assert checked == 1854 * 16 * 4
    english_ids = sorted(test["bm25/original", 10])
    assert len(english_ids) == 247
    summary_checked = 0
    with (results / "summary.csv").open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["scope"] == "split_language" and row["group"] == "test/english":
                expected = list(test[row["method"], int(row["k"])].values())
                assert int(row["queries"]) == len(expected) == 247
                for metric in expected[0]:
                    values = [r[metric] for r in expected if r[metric] is not None]
                    assert abs(float(row[metric]) - statistics.mean(values)) < 1e-11
                summary_checked += 1
    assert summary_checked == 64
    # Resample relevance-signature groups so questions with the same qrels stay together.
    group_ids = sorted({splits[q]["group"] for q in english_ids})
    members = [[q for q in english_ids if splits[q]["group"] == g] for g in group_ids]
    rng = np.random.default_rng(20261117)
    draws = rng.integers(0, len(members), size=(2000, len(members)))
    comparisons = []
    for backend in ("bm25", "ocr", "image", "combined"):
        for family in ("multiquery", "dspy_initial", "dspy_optimized"):
            for k in (5, 10, 20):
                for metric in ("hit", "recall", "ndcg"):
                    left = test[f"{backend}/original", k]
                    right = test[f"{backend}/{family}", k]
                    deltas = {q: right[q][metric] - left[q][metric] for q in english_ids}
                    sums = np.array([sum(deltas[q] for q in group) for group in members])
                    counts = np.array([len(group) for group in members])
                    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
                    low, high = np.quantile(samples, [0.025, 0.975])
                    comparisons.append({
                        "backend": backend, "family": family, "k": k, "metric": metric,
                        "queries": 247, "groups": len(members),
                        "difference": statistics.mean(deltas.values()),
                        "paired_group_bootstrap_lower_95": float(low),
                        "paired_group_bootstrap_upper_95": float(high),
                        "wins": sum(d > 1e-12 for d in deltas.values()),
                        "losses": sum(d < -1e-12 for d in deltas.values()),
                        "ties": sum(abs(d) <= 1e-12 for d in deltas.values()),
                    })
    write_csv(results / "paired_comparisons.csv", comparisons)
    report = {"metric_rows_independently_checked": checked, "queries": len(ranks),
              "headline_summary_rows_checked": summary_checked,
              "test_english_queries": 247, "methods": len(reference_methods),
              "translation_group_split_overlap": 0,
              "bootstrap_replicates": 2000,
              "interval_note": "Descriptive pointwise paired intervals; no multiple-testing correction and no report-level resampling.",
              "status": "passed"}
    save_json(results / "independent_audit.json", report)
    print(report)


if __name__ == "__main__":
    audit()
