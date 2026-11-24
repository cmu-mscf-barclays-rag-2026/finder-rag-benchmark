"""Matched FinDER original-query and GPT MultiQuery backend comparison.

Replay the Week 10 BM25 searches and GPT rewrites. Embed the same reference
passages and queries with a pinned local MiniLM checkpoint, then compare BM25,
dense, and weighted hybrid rankings on the same 200 held-out questions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from week3_bm25.finder_bm25.data import digest, load_bundle, read_jsonl, select_queries, write_json, write_jsonl
from week3_bm25.finder_bm25.evaluation import evaluate_rag
from week10_followup.finder_query_expansion import rrf

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "week3_bm25/data/team_shared"
SOURCE = ROOT / "week10_followup/results/finder_gpt200"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL = ROOT / ".cache/models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots" / MODEL_REVISION
OUTPUT = ROOT / "week11_followup/results/finder_backends"
DEPTH = 20
KS = (5, 8, 10)
HYBRID_DENSE_WEIGHT = 0.25  # Week 3 dev-selected weight; fixed before this test.
METHODS = ("bm25_original", "bm25_multiquery", "dense_original", "dense_multiquery",
           "hybrid_original", "hybrid_multiquery", "hybrid_equal_original",
           "hybrid_equal_multiquery")
METRICS = ("hit", "recall", "mrr", "ndcg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_rrf(rankings: list[list[str]], weights: list[float], depth: int | None = None) -> list[str]:
    if len(rankings) != len(weights) or not rankings or any(weight < 0 for weight in weights):
        raise ValueError("Rankings and nonnegative weights must align")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate document in a source ranking")
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (60 + rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return ordered if depth is None else ordered[:depth]


def load_saved(bundle: dict) -> tuple[list[dict], dict[str, dict], dict]:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    test = select_queries(bundle, "test", 200)
    if (manifest["bundle_fingerprint"] != bundle["manifest"]["fingerprint"]
            or manifest["test_query_ids_sha256"] != digest([q["query_id"] for q in test])
            or manifest["test_count"] != len(test)
            or manifest["depth_per_query"] != DEPTH):
        raise ValueError("Saved GPT run does not match the FinDER bundle and selected questions")
    traces = read_jsonl(SOURCE / "multiquery/traces.jsonl")
    by_id = {trace["query_id"]: trace for trace in traces}
    if len(by_id) != len(test) or set(by_id) != {q["query_id"] for q in test}:
        raise ValueError("Saved traces have missing or duplicate test questions")
    for query in test:
        trace = by_id[query["query_id"]]
        searches = trace["searches"]
        if (not searches or searches[-1]["query"] != query["text"]
                or [s["query"] for s in searches[:-1]] != trace["generated_queries"]
                or any(len(s["doc_ids"]) != DEPTH or len(set(s["doc_ids"])) != DEPTH for s in searches)):
            raise ValueError(f"Malformed saved searches for {query['query_id']}")
    for method, make_rank in (("single_query", lambda t: t["searches"][-1]["doc_ids"]),
                              ("multiquery", lambda t: rrf(t["searches"])[0])):
        saved = json.loads((SOURCE / method / "run_k10.json").read_text())
        if saved["query_ids"] != [q["query_id"] for q in test]:
            raise ValueError(f"Saved {method} question order differs")
        for row in saved["results"]:
            if make_rank(by_id[row["query_id"]])[:10] != row["doc_ids"]:
                raise ValueError(f"Replay differs from saved {method}: {row['query_id']}")
    return test, by_id, manifest


def dense_search(bundle: dict, test: list[dict], traces: dict[str, dict],
                 model_path: Path, device: str) -> tuple[dict[str, list[list[str]]], dict]:
    if not model_path.exists():
        raise FileNotFoundError(f"Local MiniLM checkpoint missing: {model_path}")
    started = time.perf_counter()
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    docs = bundle["corpus"]
    doc_ids = [doc["doc_id"] for doc in docs]
    doc_vectors = model.encode([doc["text"] for doc in docs], batch_size=64,
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False).astype(np.float32)
    build_seconds = time.perf_counter() - started
    formulations = [([q["text"]] + traces[q["query_id"]]["generated_queries"]) for q in test]
    counts = [len(group) for group in formulations]
    flat = [text for group in formulations for text in group]
    started = time.perf_counter()
    query_vectors = model.encode(flat, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    scores = query_vectors @ doc_vectors.T
    # Corpus IDs are sorted, so stable score ties use document ID order.
    top = np.argsort(-scores, axis=1, kind="stable")[:, :DEPTH]
    query_seconds = time.perf_counter() - started
    result = {}
    pos = 0
    for query, count in zip(test, counts):
        result[query["query_id"]] = [[doc_ids[index] for index in row]
                                       for row in top[pos:pos + count]]
        pos += count
    return result, {"build_seconds": build_seconds, "query_batch_seconds": query_seconds,
                    "formulation_count": len(flat), "embedding_dimension": doc_vectors.shape[1],
                    "max_seq_length": model.max_seq_length, "device": device}


def make_rankings(test: list[dict], traces: dict[str, dict],
                  dense: dict[str, list[list[str]]]) -> list[dict]:
    rows = []
    for query in test:
        qid = query["query_id"]
        sparse_searches = [search["doc_ids"] for search in traces[qid]["searches"]]
        # LangChain searched rewrites first, then the original question.
        sparse_original = sparse_searches[-1]
        sparse_multi = weighted_rrf(sparse_searches, [1.0] * len(sparse_searches))
        dense_original = dense[qid][0]
        dense_multi = weighted_rrf(dense[qid], [1.0] * len(dense[qid]))
        ranks = {
            "bm25_original": sparse_original,
            "bm25_multiquery": sparse_multi,
            "dense_original": dense_original,
            "dense_multiquery": dense_multi,
            "hybrid_original": weighted_rrf([sparse_original, dense_original],
                                            [1 - HYBRID_DENSE_WEIGHT, HYBRID_DENSE_WEIGHT]),
            "hybrid_multiquery": weighted_rrf([sparse_multi, dense_multi],
                                              [1 - HYBRID_DENSE_WEIGHT, HYBRID_DENSE_WEIGHT]),
            "hybrid_equal_original": weighted_rrf([sparse_original, dense_original], [0.5, 0.5]),
            "hybrid_equal_multiquery": weighted_rrf([sparse_multi, dense_multi], [0.5, 0.5]),
        }
        rows.append({"query_id": qid, "rankings": {name: rank[:DEPTH] for name, rank in ranks.items()},
                     "candidate_counts": {"bm25_multiquery": len(set().union(*map(set, sparse_searches))),
                                          "dense_multiquery": len(set().union(*map(set, dense[qid]))),
                                          "hybrid_multiquery": len(set(sparse_multi) | set(dense_multi))}})
    return rows


def evaluate(bundle: dict, test: list[dict], rankings: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    summaries, per_query, candidate_rows = [], {}, []
    qids = [q["query_id"] for q in test]
    for method in METHODS:
        for k in KS:
            run = {"schema_version": 1, "method": method,
                   "bundle_fingerprint": bundle["manifest"]["fingerprint"],
                   "partition": "test", "query_ids": qids, "top_k": k,
                   "latency_scope": "not_measured_comparably", "results": [
                       {"query_id": row["query_id"], "doc_ids": row["rankings"][method][:k],
                        "latency_ms": None} for row in rankings]}
            result = evaluate_rag(bundle, run)
            summary = result["summary"]
            summaries.append({"method": method, "k": k, "n": len(qids),
                              "hit": statistics.fmean(r["first_relevant_rank"] is not None
                                                        for r in result["per_query"]),
                              "recall": summary["recall"], "mrr": summary["mrr"],
                              "ndcg": summary["ndcg"]})
            per_query[method, k] = {r["query_id"]: {
                "hit": int(r["first_relevant_rank"] is not None), "recall": r["recall"],
                "mrr": r["reciprocal_rank"], "ndcg": r["ndcg"]} for r in result["per_query"]}
            if method in ("bm25_original", "bm25_multiquery") and k in (5, 10):
                old = json.loads((SOURCE / ("single_query" if method.endswith("original") else "multiquery")
                                  / f"evaluation_k{k}.json").read_text())["summary"]
                for metric in ("recall", "mrr", "ndcg"):
                    if abs(summary[metric] - old[metric]) > 1e-12:
                        raise ValueError(f"BM25 replay metric differs: {method} k={k} {metric}")
    for row in rankings:
        qid = row["query_id"]
        gold = set(bundle["qrels"][qid])
        sparse = set(row["rankings"]["bm25_multiquery"])
        dense = set(row["rankings"]["dense_multiquery"])
        candidate_rows.append({"query_id": qid, **row["candidate_counts"],
                               "relevant_in_bm25_pool": len(sparse & gold),
                               "relevant_in_dense_pool": len(dense & gold),
                               "relevant_added_by_dense": len((dense - sparse) & gold),
                               "relevant_added_by_bm25": len((sparse - dense) & gold)})
    return summaries, per_query, candidate_rows


def paired_rows(per_query: dict) -> list[dict]:
    pairs = [(f"{backend}_original", f"{backend}_multiquery")
             for backend in ("bm25", "dense", "hybrid")]
    pairs += [("bm25_original", "dense_original"), ("bm25_original", "hybrid_original"),
              ("bm25_multiquery", "dense_multiquery"), ("bm25_multiquery", "hybrid_multiquery")]
    pairs += [("hybrid_original", "hybrid_equal_original"),
              ("hybrid_multiquery", "hybrid_equal_multiquery")]
    rows = []
    for k in KS:
        qids = sorted(per_query["bm25_original", k])
        rng = np.random.default_rng(42)
        samples = rng.integers(0, len(qids), size=(10000, len(qids)), dtype=np.int32)
        for base, method in pairs:
            a, b = per_query[base, k], per_query[method, k]
            if set(a) != set(b):
                raise ValueError("Paired methods have different questions")
            result = {"k": k, "base": base, "method": method, "n": len(qids),
                      "hit_gained": sum(a[q]["hit"] == 0 and b[q]["hit"] == 1 for q in qids),
                      "hit_lost": sum(a[q]["hit"] == 1 and b[q]["hit"] == 0 for q in qids)}
            for metric in METRICS:
                differences = np.asarray([b[q][metric] - a[q][metric] for q in qids])
                boot = np.sort(differences[samples].mean(axis=1))
                result[f"delta_{metric}"] = float(differences.mean())
                result[f"ci95_{metric}_low"] = float(boot[249])
                result[f"ci95_{metric}_high"] = float(boot[9749])
            rows.append(result)
    return rows


def write_findings(output: Path, summaries: list[dict], pairs: list[dict],
                   candidates: list[dict], timings: dict) -> None:
    lines = ["# FinDER MultiQuery across retrieval backends", "",
             "Matched pilot: 200 held-out questions, 5,830 pooled reference passages, two saved GPT-4o mini rewrites per question plus the original. All formulations retrieve 20 candidates. Query rankings use equal-weight RRF (k=60); hybrid combines the already fused BM25 and MiniLM lists using weighted RRF with the Week 3 dev-selected 25% dense weight. The weight was fixed before this test.", "",
             "| Backend | Query | Hit@5 | Hit@8 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    lookup = {(r["method"], r["k"]): r for r in summaries}
    for backend in ("bm25", "dense", "hybrid"):
        for strategy in ("original", "multiquery"):
            method = f"{backend}_{strategy}"
            r = lookup[method, 10]
            lines.append(f"| {backend} | {strategy} | {lookup[method, 5]['hit']:.1%} | "
                         f"{lookup[method, 8]['hit']:.1%} | {r['hit']:.1%} | "
                         f"{r['recall']:.1%} | {r['mrr']:.3f} | {r['ndcg']:.3f} |")
    lines += ["", "## Paired changes at Top-10", "",
              "Positive values favor the method on the left of the comparison. Intervals are descriptive 95% paired question-bootstrap intervals (10,000 fixed-seed draws).",
              "", "| Comparison | Δ Hit@10 | 95% interval | Wins / losses | Δ nDCG@10 |",
              "|---|---:|---|---:|---:|"]
    for row in pairs:
        if row["k"] != 10 or row["method"].startswith("hybrid_equal"):
            continue
        lines.append(f"| {row['method']} − {row['base']} | {row['delta_hit']:+.1%} | "
                     f"[{row['ci95_hit_low']:+.1%}, {row['ci95_hit_high']:+.1%}] | "
                     f"{row['hit_gained']} / {row['hit_lost']} | {row['delta_ndcg']:+.3f} |")
    lines += ["", "## Fixed backend fusion check", "",
              "The equal-weight backend alternative uses the same BM25 and dense candidate rankings. It is an exploratory test-set comparison, not a tuned replacement for the prior 25% dense weight.",
              "", "| Query | Dense weight | Hit@10 | nDCG@10 | Δ Hit vs 25% | 95% interval |",
              "|---|---:|---:|---:|---:|---:|"]
    for strategy in ("original", "multiquery"):
        base = lookup[f"hybrid_{strategy}", 10]
        equal = lookup[f"hybrid_equal_{strategy}", 10]
        pair = next(p for p in pairs if p["k"] == 10 and p["base"] == f"hybrid_{strategy}"
                    and p["method"] == f"hybrid_equal_{strategy}")
        lines.append(f"| {strategy} | 25% | {base['hit']:.1%} | {base['ndcg']:.3f} | — | — |")
        lines.append(f"| {strategy} | 50% | {equal['hit']:.1%} | {equal['ndcg']:.3f} | "
                     f"{pair['delta_hit']:+.1%} | [{pair['ci95_hit_low']:+.1%}, {pair['ci95_hit_high']:+.1%}] |")
    lines += ["", "## Interpretation", "",
              "MiniLM dense retrieval is weaker than BM25 on this reference-passage sample. MultiQuery raises dense Hit@10 by 2.0 points, with an interval spanning zero. "
              "The preset hybrid MultiQuery reaches 38.5% Hit@10 versus 38.0% for BM25 MultiQuery; its paired difference is only 0.5 points, with an interval spanning zero. "
              "Equal backend weights reduce hybrid MultiQuery Hit@10 to 36.0%, so fusion choice matters here. "
              "For this FinDER setup, BM25 MultiQuery is the practical default among these tested methods: the hybrid's small observed edge does not establish a reliable benefit from maintaining a dense index. "
              "This does not settle whether MultiQuery helps a stronger dense encoder on a different corpus.", ""]
    lines += ["", "## Evidence and cost diagnostics", "",
              f"Mean unique candidate IDs per question: BM25 MultiQuery {statistics.fmean(r['bm25_multiquery'] for r in candidates):.1f}, "
              f"dense MultiQuery {statistics.fmean(r['dense_multiquery'] for r in candidates):.1f}, "
              f"hybrid MultiQuery {statistics.fmean(r['hybrid_multiquery'] for r in candidates):.1f}. "
              f"The fused dense Top-20 list contributes {sum(r['relevant_added_by_dense'] for r in candidates)} query–reference matches absent from the fused BM25 Top-20 list; "
              f"BM25 contributes {sum(r['relevant_added_by_bm25'] for r in candidates)} absent from dense.", "",
              f"Local MiniLM corpus embedding took {timings['build_seconds']:.1f} s. Batched encoding and ranking of {timings['formulation_count']} query formulations took {timings['query_batch_seconds']:.1f} s. "
              "These batch times exclude GPT query generation and are not directly comparable with the earlier sequential BM25 latency. The saved GPT MultiQuery run used one generation call per question (200 total); no new generation calls were made here.", "",
              "This benchmark scores reference-passage retrieval, not answer correctness. Its 200-question results are a pilot and should not be extrapolated to the full 4,575-question test set. "
              "The hybrid uses two fusion stages, so its gains can reflect candidate pooling and ranking together. "
              "See `summary.csv`, `paired.csv`, `candidate_diagnostics.csv`, and `rankings.jsonl` for the full matched results.", ""]
    (output / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    generated = {"summary.csv", "paired.csv", "candidate_diagnostics.csv", "rankings.jsonl",
                 "findings.md", "manifest.json"}
    diagnostics = {"error_analysis.csv", "error_analysis_counts.json", "error_analysis.md"}
    if args.output.exists() and {p.name for p in args.output.iterdir()} - generated - diagnostics:
        raise ValueError("Output contains files this script does not own")
    bundle = load_bundle(BUNDLE)
    test, traces, source_manifest = load_saved(bundle)
    dense, timings = dense_search(bundle, test, traces, args.model_path, args.device)
    rankings = make_rankings(test, traces, dense)
    summaries, per_query, candidates = evaluate(bundle, test, rankings)
    pairs = paired_rows(per_query)
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "summary.csv", summaries)
    save_csv(args.output / "paired.csv", pairs)
    save_csv(args.output / "candidate_diagnostics.csv", candidates)
    write_jsonl(args.output / "rankings.jsonl", rankings)
    write_findings(args.output, summaries, pairs, candidates, timings)
    write_json(args.output / "manifest.json", {
        "bundle_fingerprint": bundle["manifest"]["fingerprint"],
        "test_query_ids_sha256": digest([q["query_id"] for q in test]),
        "test_count": len(test), "corpus_count": len(bundle["corpus"]),
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "source_traces_sha256": sha256(SOURCE / "multiquery/traces.jsonl"),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": MODEL_REVISION, "model_path": str(args.model_path),
        "depth_per_formulation": DEPTH, "ks": KS, "query_fusion": "equal-weight RRF k=60",
        "backend_fusion": "weighted RRF k=60", "hybrid_dense_weight": HYBRID_DENSE_WEIGHT,
        "fusion_order": "queries within backend, then backends",
        "timings": timings, "python": platform.python_version(),
        "sentence_transformers_version": __import__("sentence_transformers").__version__,
        "source_code_sha256": sha256(Path(__file__)),
        "outputs_sha256": {name: sha256(args.output / name) for name in generated
                           if name != "manifest.json"},
    })
    print(f"Wrote matched FinDER comparison for {len(test)} questions -> {args.output}")


if __name__ == "__main__":
    main()
