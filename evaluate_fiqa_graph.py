"""Evaluate the standalone graph on FiQA-2018 (BEIR version) passage retrieval.

FiQA-2018 task 2 asks financial opinion questions over 57,638 forum answers.
This uses the BEIR release (``BeIR/fiqa`` and ``BeIR/fiqa-qrels``) and its
standard reporting: the 648 test queries, nDCG@10 and Recall@100, plus MRR@10
and Recall@10. By default every method is one fixed configuration and the
validation split is unused (direct transfer).

``--tune`` (exploratory, not yet reported) adds ``tuned_graph``: pivoted
length normalization and the amount of propagation are chosen on the 500
validation queries alone, frozen, and run once on test.

- ``improved_graph`` / ``improved_graph_one_step``: the app's graph, with and
  without propagation. FiQA passages have no headings or sections, so the
  heading tree and heading words are inactive; only the concept graph and
  the volume-weighted query links act.
- ``original_graph``: ``LEGACY_SETTINGS``.
- ``bm25``: the team's BM25 (team_bm25.py) with BEIR's Anserini parameters
  (k1 = 0.9, b = 0.4). BEIR reports nDCG@10 0.236 for its BM25.
- ``dense_minilm``: all-MiniLM-L6-v2 cosine search, the team's dense model.

Paired percentile bootstrap (10,000 query resamples, seed 42) for nDCG@10
differences. Latency is a single CPU pass over the test queries, top-100.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from datasets import load_dataset
from langchain_core.documents import Document
from threadpoolctl import threadpool_limits

from standalone_graph import LEGACY_SETTINGS, StandaloneGraphRetriever, StandaloneGraphSettings
from team_bm25 import BM25Retriever


DEPTH = 100
GRID_LENGTH_NORMALIZATION = (0.0, 0.25, 0.5, 0.75, 1.0)
# (mode, restart probability); None = one step, no propagation.
GRID_PROPAGATION = (("walk_r0.35", 0.35), ("walk_r0.6", 0.6), ("one_step", None))
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE = Path.home() / ".cache" / "finder-rag-baseline" / "fiqa_minilm_corpus.npy"


def load_fiqa(split: str = "test") -> tuple[list[dict], dict[str, str], dict[str, set[str]]]:
    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus").to_list()
    queries = {str(q["_id"]): q["text"] for q in load_dataset("BeIR/fiqa", "queries", split="queries")}
    qrels: dict[str, set[str]] = {}
    for row in load_dataset("BeIR/fiqa-qrels", split=split):
        if row["score"] > 0:
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return corpus, {q: queries[q] for q in qrels}, qrels


def passage_text(row: dict) -> str:
    return f"{row['title']}\n{row['text']}".strip()


def metrics(ranking: list[str], relevant: set[str]) -> dict[str, float]:
    gains = [float(pid in relevant) for pid in ranking]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:10]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    first = next((i for i, g in enumerate(gains[:10], 1) if g), None)
    return {
        "ndcg@10": dcg / ideal,
        "mrr@10": 1 / first if first else 0.0,
        "recall@10": sum(gains[:10]) / len(relevant),
        "recall@100": sum(gains[:100]) / len(relevant),
    }


def bootstrap(deltas: list[float], seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(10_000))
    return means[249], means[9749]


def dense_searcher(corpus: list[dict]) -> Callable[[str], list[str]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL, device="cpu")
    if EMBED_CACHE.exists():
        embeddings = np.load(EMBED_CACHE)
    else:
        embeddings = model.encode([passage_text(r) for r in corpus], batch_size=128,
                                  normalize_embeddings=True, show_progress_bar=True)
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBED_CACHE, embeddings)
    ids = [str(r["_id"]) for r in corpus]

    def search(query: str) -> list[str]:
        scores = embeddings @ model.encode(query, normalize_embeddings=True)
        top = np.argpartition(-scores, DEPTH)[:DEPTH]
        return [ids[i] for i in sorted(top, key=lambda i: (-scores[i], ids[i]))]

    return search


def tune_on_validation(documents: list[Document]) -> tuple[dict, list[dict]]:
    """Grid over length normalization x propagation, on the validation split only.

    Selection: highest validation nDCG@10, then MRR@10, then the simpler
    setting (less normalization, less propagation)."""
    _, queries, qrels = load_fiqa("validation")
    grid = []
    for b in GRID_LENGTH_NORMALIZATION:
        graph = StandaloneGraphRetriever(
            documents, top_k=10, settings=replace(StandaloneGraphSettings(), length_normalization=b))
        for mode, restart in GRID_PROPAGATION:
            if restart is not None:
                graph.settings = replace(graph.settings, restart_probability=restart)
            rows = [metrics([e.node_id for e in graph.retrieve(text, propagate=restart is not None).explanations],
                            qrels[qid]) for qid, text in queries.items()]
            grid.append({"length_normalization": b, "propagation": mode,
                         "restart_probability": restart if restart is not None else "",
                         "n_queries": len(rows),
                         "ndcg@10": statistics.fmean(r["ndcg@10"] for r in rows),
                         "mrr@10": statistics.fmean(r["mrr@10"] for r in rows)})
            print(f"validation b={b} {mode}: nDCG@10={grid[-1]['ndcg@10']:.4f}", flush=True)
    order = {mode: i for i, (mode, _) in enumerate(GRID_PROPAGATION)}
    best = max(grid, key=lambda r: (round(r["ndcg@10"], 6), round(r["mrr@10"], 6),
                                    -r["length_normalization"], order[r["propagation"]]))
    return best, grid


def evaluate(output_dir: Path, with_dense: bool = True, tune: bool = False) -> dict:
    corpus, queries, qrels = load_fiqa("test")
    documents = [Document(page_content=passage_text(r), metadata={"passage_id": str(r["_id"])})
                 for r in corpus]
    configs = [("improved_graph", StandaloneGraphSettings()), ("original_graph", LEGACY_SETTINGS)]
    best, grid, tuned_propagate = None, [], True
    if tune:
        best, grid = tune_on_validation(documents)
        tuned_settings = replace(StandaloneGraphSettings(), length_normalization=best["length_normalization"],
                                 **({"restart_probability": best["restart_probability"]}
                                    if best["restart_probability"] != "" else {}))
        tuned_propagate = best["restart_probability"] != ""
        configs.append(("tuned_graph", tuned_settings))
    build: dict[str, float] = {}
    graphs: dict[StandaloneGraphSettings, StandaloneGraphRetriever] = {}
    for name, settings in configs:
        if settings not in graphs:
            started = time.perf_counter()
            graphs[settings] = StandaloneGraphRetriever(documents, top_k=DEPTH, settings=settings)
            build[name] = time.perf_counter() - started
    improved, original = graphs[StandaloneGraphSettings()], graphs[LEGACY_SETTINGS]
    tuned = graphs[configs[-1][1]] if tune else None
    started = time.perf_counter()
    bm25 = BM25Retriever([{"doc_id": str(r["_id"]), "text": passage_text(r)} for r in corpus], k1=0.9, b=0.4)
    build["bm25"] = time.perf_counter() - started

    searchers: dict[str, Callable[[str], list[str]]] = {
        "improved_graph": lambda q: [e.node_id for e in improved.retrieve(q).explanations],
        "improved_graph_one_step": lambda q: [e.node_id for e in improved.retrieve(q, propagate=False).explanations],
        "original_graph": lambda q: [e.node_id for e in original.retrieve(q).explanations],
        "bm25": lambda q: [hit["doc_id"] for hit in bm25.search(q, top_k=DEPTH)],
    }
    if tune:
        searchers["tuned_graph"] = lambda q: [
            e.node_id for e in tuned.retrieve(q, propagate=tuned_propagate).explanations]
    if with_dense:
        started = time.perf_counter()
        searchers["dense_minilm"] = dense_searcher(corpus)
        build["dense_minilm"] = time.perf_counter() - started

    per_query, latency, traces = [], {}, []
    with threadpool_limits(limits=4):
        for name, search in searchers.items():
            search(next(iter(queries.values())))  # warm-up
            times = []
            for qid, text in queries.items():
                started = time.perf_counter()
                ranking = search(text)
                times.append((time.perf_counter() - started) * 1000)
                per_query.append({"method": name, "query_id": qid, **metrics(ranking, qrels[qid])})
                if name in ("improved_graph", "tuned_graph"):
                    graph = improved if name == "improved_graph" else tuned
                    result = graph.retrieve(text, top_k=3, propagate=name == "improved_graph" or tuned_propagate)
                    traces.append({"method": name, "query_id": qid, "query": text, "relevant": sorted(qrels[qid]),
                                   "query_links": result.query_links,
                                   "top3": [e.to_dict() for e in result.explanations]})
            latency[name] = statistics.fmean(times)
            print(f"{name}: done", flush=True)

    names = list(searchers)
    summary_rows = []
    for name in names:
        rows = [r for r in per_query if r["method"] == name]
        summary_rows.append({"method": name, "n_queries": len(rows),
                             **{m: statistics.fmean(r[m] for r in rows)
                                for m in ("ndcg@10", "mrr@10", "recall@10", "recall@100")},
                             "latency_mean_ms": latency[name], "build_seconds": build.get(name, "")})
    by = {(r["method"], r["query_id"]): r["ndcg@10"] for r in per_query}
    paired = []
    for name in names:
        for other in ("bm25", "original_graph") + (("improved_graph",) if tune else ()):
            if name == other:
                continue
            deltas = [by[(name, q)] - by[(other, q)] for q in queries]
            low, high = bootstrap(deltas)
            paired.append({"comparison": f"{name} minus {other}", "n_queries": len(deltas),
                           "ndcg10_difference": statistics.fmean(deltas),
                           "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                           "bootstrap_95pct_low": low, "bootstrap_95pct_high": high})

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [("per_query.csv", per_query), ("test_metrics.csv", summary_rows),
               ("paired_ndcg10_intervals.csv", paired)] + ([("validation_grid.csv", grid)] if tune else [])
    for filename, rows in outputs:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "FiQA-2018 (BEIR: BeIR/fiqa, BeIR/fiqa-qrels), test split",
        "corpus_passages": len(corpus), "test_queries": len(queries),
        "mean_relevant_per_query": statistics.fmean(len(v) for v in qrels.values()),
        "passage_representation": "title + text (BEIR); titles are empty in FiQA",
        "graph_settings": {"improved_graph": improved.settings.__dict__,
                           "original_graph": original.settings.__dict__},
        "graph": {"improved_graph": improved.stats, "original_graph": original.stats},
        "bm25": "team_bm25.py (Florence, ce54a01), k1=0.9, b=0.4 (BEIR/Anserini defaults)",
        "dense": DENSE_MODEL if with_dense else None,
        "published_reference": "BEIR (Thakur et al., 2021) BM25 nDCG@10 on FiQA: 0.236",
        "protocol": "Fixed configurations; nothing tuned on FiQA; validation split unused.",
        "timing": "one CPU pass over test queries (4 BLAS threads), top-100, after one warm-up query",
    }
    if tune:
        summary["graph_settings"]["tuned_graph"] = {**tuned.settings.__dict__, "propagate": tuned_propagate}
        summary["tuning"] = {
            "split": "validation (500 queries)", "selected": best,
            "rule": "highest validation nDCG@10, then MRR@10, then less normalization and propagation",
            "grid": {"length_normalization": GRID_LENGTH_NORMALIZATION,
                     "propagation": [m for m, _ in GRID_PROPAGATION]}}
        summary["protocol"] = ("tuned_graph was selected on the validation split only and run once on "
                               "test; every other method is a fixed configuration.")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in summary_rows:
        print(f"{row['method']:24s} nDCG@10={row['ndcg@10']:.4f} MRR@10={row['mrr@10']:.4f} "
              f"R@10={row['recall@10']:.4f} R@100={row['recall@100']:.4f} {row['latency_mean_ms']:.0f} ms")
    for row in paired:
        print(f"{row['comparison']:44s} {row['ndcg10_difference']:+.4f} "
              f"[{row['bootstrap_95pct_low']:+.4f}, {row['bootstrap_95pct_high']:+.4f}] "
              f"{row['wins']}W/{row['losses']}L")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fiqa_graph"))
    parser.add_argument("--no-dense", action="store_true", help="skip the MiniLM dense reference")
    parser.add_argument("--tune", action="store_true",
                        help="exploratory: tune length normalization and propagation on validation")
    args = parser.parse_args()
    evaluate(args.output_dir, with_dense=not args.no_dense, tune=args.tune)


if __name__ == "__main__":
    main()
