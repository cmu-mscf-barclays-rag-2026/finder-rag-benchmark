"""Export query provenance, matched-setting controls and transparent cost summaries."""

import argparse
from collections import defaultdict
import importlib.metadata
from hashlib import sha256
import json
from pathlib import Path
import statistics

from benchmark_core import metric_values, write_csv, save_json, DATA_REVISION, MODEL_REVISION
from experiment import Experiment, ROOT, read, BACKENDS


def supplement(args):
    e = Experiment(args)
    selection = read(e.results / "selection.json")
    baseline = selection["baseline"]
    test = e.partition("test", english=True)
    matched = []
    for backend in BACKENDS:
        params = selection["expansion"][backend]["dspy_optimized"]
        for family, strategy in (("dspy_initial", "dspy_initial"),
                                 ("dspy_optimized", params["strategy"])):
            scores = []
            for q in test:
                alternatives = read(e.cache / "generated" / strategy / f"{q['query_id']}.json")["queries"]
                ranking = e.expanded_rank(q, alternatives, backend, baseline, params)
                scores.append(metric_values(ranking, e.gold[q["query_id"]], 10))
            matched.append({"backend": backend, "family": family, "queries": len(test),
                            "ndcg_at_10": statistics.mean(s["ndcg"] for s in scores),
                            "recall_at_10": statistics.mean(s["recall"] for s in scores),
                            "hit_at_10": statistics.mean(s["hit"] for s in scores),
                            "parameters": json.dumps(params, sort_keys=True)})
    write_csv(e.results / "matched_dspy_control.csv", matched)

    controls = []
    for name, backend, config in (
        ("bm25_previous_default", "bm25", {**baseline, "k1": 1.5, "b": 0.75}),
        ("fusion_previous_50_50", "combined", {**baseline, "ocr_weight": 0.5}),
    ):
        for q in e.queries:
            ranking = e.rank(q["query"], backend, config)
            for k in (1, 5, 10, 20):
                controls.append({"query_id": q["query_id"], "language": q["language"],
                                 "method": name, "k": k,
                                 **metric_values(ranking, e.gold[q["query_id"]], k)})
    write_csv(e.results / "previous_setting_controls.csv", controls)

    strategies = sorted({p["strategy"] for methods in selection["expansion"].values()
                         for p in methods.values()})
    repairs = defaultdict(int)
    query_rows = []
    by_id = e.by_id
    for strategy in strategies:
        for q in e.queries:
            saved = read(e.cache / "generated" / strategy / f"{q['query_id']}.json")
            assert len(saved["queries"]) == 5
            repairs[strategy] += saved.get("format_repaired", False)
            query_rows.append({**saved, "language": q["language"], "question": q["query"]})
    with (e.results / "generated_queries.jsonl").open("w", encoding="utf-8") as output:
        for row in query_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Choose qualitative examples by saved per-question nDCG deltas, without new tuning.
    ranks = read(e.results / "rankings.json")
    cases = []
    for backend in BACKENDS:
        deltas = []
        for q in test:
            qid = q["query_id"]
            a = ranks[str(qid)][f"{backend}/original"]
            b = ranks[str(qid)][f"{backend}/multiquery"]
            delta = metric_values(b, e.gold[qid], 10)["ndcg"] - metric_values(a, e.gold[qid], 10)["ndcg"]
            deltas.append((delta, qid, a, b))
        for kind, item in (("largest_gain", max(deltas)), ("largest_loss", min(deltas))):
            delta, qid, a, b = item
            strategy = selection["expansion"][backend]["multiquery"]["strategy"]
            cases.append({"backend": backend, "example_type": kind, "query_id": qid,
                          "question": by_id[qid]["query"], "ndcg_at_10_change": delta,
                          "original_top10": a[:10], "expanded_top10": b[:10],
                          "judged_pages": e.gold[qid],
                          "generated_queries": read(e.cache / "generated" / strategy / f"{qid}.json")["queries"]})
    save_json(e.results / "qualitative_examples.json", cases)
    usage = e.api_client().usage()
    versions = {}
    for package in ("dspy", "langchain-classic", "langchain-core", "optuna", "transformers",
                    "torch", "numpy", "pyarrow", "requests"):
        versions[package] = importlib.metadata.version(package)
    manifest = read(e.results / "manifest.json")
    embedding_metadata = {}
    for kind in ("ocr", "image", "query"):
        original = read(e.args.vectors / f"{kind}.json")
        lengths = original["sample_lengths"]
        embedding_metadata[kind] = {
            "rows": len(original["ids"]), "input_cap": original["text_limit"],
            "inputs_at_cap": sum(length == original["text_limit"] for length in lengths),
            "tokens_min": min(lengths), "tokens_max": max(lengths),
            "tokens_median": statistics.median(lengths),
            "original_gpu": original["gpu"],
        }
    for seed in (7, 19):
        assert not read(e.results / f"dspy_seed_{seed}.json")["generate"]["demos"]
    manifest.update({"dataset_revision": DATA_REVISION, "model_revision": MODEL_REVISION,
                     "versions": versions, "selected_generation_strategies": strategies,
                     "exported_query_generation_records": len(query_rows),
                     "format_repair_counts": dict(repairs),
                     "reused_embedding_metadata": embedding_metadata,
                     "generated_query_max_utf8_bytes": max(len(query.encode()) for row in query_rows for query in row["queries"]),
                     "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
                     "cost_note": "Recorded unique successful request usage estimate, without cache discounts; excludes interrupted/unrecorded calls and is not a billing statement.",
                     "query_embedding_batches": [read(p) for p in sorted((e.cache / "query_vectors").glob("batch_times*.json"))],
                     "optimizer_note": "Twelve optimization trials plus the default-program evaluation per seed; saved DSPy scores are rounded by the framework.",
                     "api_latency_median_seconds": statistics.median(r["seconds"] for r in usage),
                     "api_latency_p95_seconds": sorted(r["seconds"] for r in usage)[int(0.95 * (len(usage) - 1))]})
    save_json(e.results / "manifest.json", manifest)
    print("Supplementary controls, examples and provenance exported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".cache/data")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/model")
    parser.add_argument("--vectors", type=Path, default=ROOT / ".cache/page_vectors")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/experiment")
    supplement(parser.parse_args())
