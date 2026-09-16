"""Run with python -m finder_bm25 from the Person 2 folder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError

from .bm25 import BM25Retriever, TOKENIZER_VERSION
from .data import (digest, load_bundle, prepare_records, read_jsonl, save_bundle, write_json)
from .download import fetch_records
from .evaluation import collect_run, evaluate_rag
from .reporting import compare_runs, save_evaluation, table, write_csv


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="FinDER Person 2: BM25 + common ranking evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare the fixed shared corpus and qrels")
    prepare.add_argument("--input", type=Path, help="Local FinDER-shaped JSONL (omit to download FinDER)")
    prepare.add_argument("--output", type=Path, default=Path("data/shared"))
    prepare.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dev-fraction", type=float, default=0.2)
    prepare.add_argument("--split-protocol", choices=["seeded-random-v1", "sha1-mod5-dev-v1"],
                         default="seeded-random-v1", help="Use sha1-mod5-dev-v1 for the team comparison")
    prepare.add_argument("--chunk-size", type=int, default=0, help="Characters; 0 keeps whole references")
    prepare.add_argument("--overlap", type=int, default=0, help="Characters")
    run = commands.add_parser("run", help="Evaluate BM25 at every requested k")
    run.add_argument("--partition", choices=["dev", "test", "all"], default="test")
    run.add_argument("--limit", type=int, help="Limit queries only; always index the entire shared corpus")
    run.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10])
    run.add_argument("--output", type=Path, default=Path("results/bm25"))
    run.add_argument("--method", default="bm25")
    search = commands.add_parser("search", help="Inspect BM25 evidence for a custom question")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    sweep = commands.add_parser("sweep", help="Tune k1/b on DEV only, leaving test queries for final evaluation")
    sweep.add_argument("--k1-values", type=float, nargs="+", default=[0.8, 1.2, 1.6])
    sweep.add_argument("--b-values", type=float, nargs="+", default=[0.25, 0.75, 1.0])
    sweep.add_argument("--top-k", type=int, default=5)
    sweep.add_argument("--limit", type=int)
    sweep.add_argument("--output", type=Path, default=Path("results/dev_sweep"))
    evaluate = commands.add_parser("evaluate", help="Apply the common metrics to any method's run")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--answers", type=Path, help="JSONL: query_id and answer; optional Person 3 predictions")
    evaluate.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Re-score and compare compatible BM25/dense/hybrid/MMR runs")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, default=Path("results/comparison"))
    for command in (run, search, sweep, evaluate, compare):
        command.add_argument("--data", type=Path, default=Path("data/shared"))
    for command in (run, search):
        command.add_argument("--k1", type=float, default=1.2)
        command.add_argument("--b", type=float, default=0.75)
    return root


def _bm25(bundle: dict, k1: float, b: float) -> tuple[BM25Retriever, float, dict]:
    start = time.perf_counter()
    retriever = BM25Retriever(bundle["corpus"], k1=k1, b=b)
    elapsed = (time.perf_counter() - start) * 1000
    return retriever, elapsed, {"k1": k1, "b": b, "tokenizer": TOKENIZER_VERSION, "idf": "positive_robertson"}


def execute(args: argparse.Namespace) -> None:
    if args.command == "prepare":
        if args.input:
            records = read_jsonl(args.input)
            source = {"kind": "local_jsonl", "path": str(args.input), "records_fingerprint": digest(records)}
        else:
            print("Loading pinned FinDER snapshot...", flush=True)
            records, source = fetch_records(args.cache_dir)
        bundle = prepare_records(records, seed=args.seed, dev_fraction=args.dev_fraction,
                                 chunk_size=args.chunk_size, overlap=args.overlap, source=source,
                                 split_protocol=args.split_protocol)
        save_bundle(bundle, args.output)
        print(json.dumps(bundle["manifest"], indent=2))
        return
    bundle = load_bundle(args.data)
    if args.command == "compare":
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
        compare_runs(bundle, runs, args.output)
        print(f"Comparison written to {args.output / 'comparison.md'}")
    elif args.command == "evaluate":
        run = json.loads(args.run.read_text(encoding="utf-8"))
        if args.answers:
            predictions = read_jsonl(args.answers)
            answer_ids = [row["query_id"] for row in predictions]
            if len(set(answer_ids)) != len(answer_ids) or set(answer_ids) - set(run["query_ids"]):
                raise ValueError("Predictions contain duplicate or unknown query IDs")
            answers = {row["query_id"]: row["answer"] for row in predictions}
            for row in run["results"]:
                if row["query_id"] in answers:
                    row["answer"] = answers[row["query_id"]]
        evaluation = evaluate_rag(bundle, run)
        save_evaluation(bundle, run, evaluation, args.output)
        print(table([evaluation["summary"]]))
    elif args.command == "sweep":
        summaries = []
        for k1 in sorted(set(args.k1_values)):
            for b in sorted(set(args.b_values)):
                print(f"DEV: k1={k1}, b={b}, k={args.top_k}", flush=True)
                retriever, elapsed, config = _bm25(bundle, k1, b)
                run = collect_run(bundle, retriever, method=f"bm25_k1_{k1}_b_{b}", top_k=args.top_k,
                                  partition="dev", limit=args.limit, config=config, build_latency_ms=elapsed)
                evaluation = evaluate_rag(bundle, run)
                save_evaluation(bundle, run, evaluation, args.output / f"k1_{k1}_b_{b}")
                summaries.append({**evaluation["summary"], "k1": k1, "b": b})
        best = max(summaries, key=lambda row: (row["ndcg"], row["mrr"]))
        write_csv(args.output / "sweep.csv", summaries)
        write_json(args.output / "best_config.json", {"k1": best["k1"], "b": best["b"],
                   "selection_metric": f"dev_ndcg@{args.top_k}", "dev_ndcg": best["ndcg"],
                   "query_set_fingerprint": best["query_set_fingerprint"],
                   "bundle_fingerprint": best["bundle_fingerprint"]})
        print(f"Selected on dev: k1={best['k1']}, b={best['b']}; nDCG={best['ndcg']:.4f}")
    else:
        print("Building BM25 index...", flush=True)
        retriever, elapsed, config = _bm25(bundle, args.k1, args.b)
        if args.command == "search":
            docs = {doc["doc_id"]: doc for doc in bundle["corpus"]}
            hits = retriever.search(args.query, args.top_k)
            if not hits:
                print("No matching query terms in the corpus.")
            for rank, hit in enumerate(hits, 1):
                print(f"\n[{rank}] score={hit['score']:.4f} {hit['doc_id']}\n{docs[hit['doc_id']]['text']}\n")
        else:
            summaries = []
            for k in sorted(set(args.top_k)):
                print(f"Evaluating {args.partition} queries at k={k}...", flush=True)
                run = collect_run(bundle, retriever, method=args.method, top_k=k, partition=args.partition,
                                  limit=args.limit, config=config, build_latency_ms=elapsed)
                evaluation = evaluate_rag(bundle, run)
                save_evaluation(bundle, run, evaluation, args.output / f"k{k}")
                summaries.append(evaluation["summary"])
            write_csv(args.output / "summary.csv", summaries)
            (args.output / "summary.md").write_text("# BM25 results\n\n" + table(summaries) + "\n", encoding="utf-8")
            print(table(summaries))


def main() -> None:
    arguments = parser().parse_args()
    try:
        execute(arguments)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
