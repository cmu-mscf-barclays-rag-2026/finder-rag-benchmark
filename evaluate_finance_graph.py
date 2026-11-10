"""Graph retrieval on MTEB's financial retrieval sets (FinQA, FinanceBench).

Split and metrics are in finance_data.py (grouped 20/80 development/test,
seed 42; one gold passage per question). Methods, all fixed configurations:

- ``improved_graph`` / ``improved_graph_one_step``: the app's graph, with and
  without propagation. These passages have no headings or titles, so the
  heading tree and heading words are inactive.
- ``original_graph``: ``LEGACY_SETTINGS``.
- ``bm25``: the team's BM25 (team_bm25.py), BEIR/Anserini k1 = 0.9, b = 0.4.
- ``dense_minilm``: all-MiniLM-L6-v2 cosine search.

Error analysis per question (improved graph and BM25): gold rank (near miss
= rank 2-10, far miss = not in the top 100), whether the question's years
appear in the gold and in the top-ranked passage, the share of the question's
concepts each contains, and passage lengths. Paired bootstrap intervals
resample gold-passage groups (legal_protocol.paired_bootstrap).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
from langchain_core.documents import Document
from threadpoolctl import threadpool_limits

from finance_data import DATASETS, FinanceQuestion, load_finance, metrics, split_questions
from legal_protocol import paired_bootstrap
from standalone_graph import LEGACY_SETTINGS, StandaloneGraphRetriever, StandaloneGraphSettings
from team_bm25 import BM25Retriever


DEPTH = 100
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRICS = ("ndcg@10", "mrr@10", "recall@1", "recall@5", "recall@10", "recall@100")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def dense_searcher(corpus: list[dict]) -> Callable[[str], list[str]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL, device="cpu")
    embeddings = model.encode([r["text"] for r in corpus], batch_size=32, normalize_embeddings=True)
    ids = [r["id"] for r in corpus]

    def search(query: str) -> list[str]:
        scores = embeddings @ model.encode(query, normalize_embeddings=True)
        order = np.argsort(-scores, kind="stable")[:DEPTH]
        return [ids[i] for i in order]

    return search


def diagnostics(graph: StandaloneGraphRetriever, text_by_id: dict[str, str], question: FinanceQuestion,
                ranking: list[str]) -> dict:
    """Why a ranking misses: years and concept overlap of gold vs. top passage."""
    gold = next(iter(question.relevant))
    concepts = set(graph.concepts(question.text))
    years = set(_YEAR.findall(question.text))

    def coverage(pid: str) -> float:
        return len(concepts & set(graph.concepts(text_by_id[pid]))) / len(concepts) if concepts else 0.0

    top = ranking[0] if ranking else None
    return {
        "question_years": len(years),
        "gold_has_years": float(bool(years) and years <= set(_YEAR.findall(text_by_id[gold]))) if years else "",
        "top_has_years": (float(bool(years) and years <= set(_YEAR.findall(text_by_id[top])))
                          if years and top else ""),
        "gold_concept_coverage": coverage(gold),
        "top_concept_coverage": coverage(top) if top else "",
        "gold_words": len(text_by_id[gold].split()),
        "top10_median_words": statistics.median(len(text_by_id[p].split()) for p in ranking[:10]) if ranking else "",
    }


def evaluate_dataset(name: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    corpus, questions = load_finance(name)
    dev, test = split_questions(questions)
    text_by_id = {r["id"]: r["text"] for r in corpus}
    documents = [Document(page_content=r["text"], metadata={"passage_id": r["id"]}) for r in corpus]
    improved = StandaloneGraphRetriever(documents, top_k=DEPTH)
    original = StandaloneGraphRetriever(documents, top_k=DEPTH, settings=LEGACY_SETTINGS)
    bm25 = BM25Retriever([{"doc_id": r["id"], "text": r["text"]} for r in corpus], k1=0.9, b=0.4)
    searchers: dict[str, Callable[[str], list[str]]] = {
        "improved_graph": lambda q: [e.node_id for e in improved.retrieve(q).explanations],
        "improved_graph_one_step": lambda q: [e.node_id for e in improved.retrieve(q, propagate=False).explanations],
        "original_graph": lambda q: [e.node_id for e in original.retrieve(q).explanations],
        "bm25": lambda q: [hit["doc_id"] for hit in bm25.search(q, top_k=DEPTH)],
        "dense_minilm": dense_searcher(corpus),
    }
    ordered = sorted(questions, key=lambda q: (len(q), q))
    per_query, latency = [], {}
    with threadpool_limits(limits=4):
        for method, search in searchers.items():
            for qid in ordered[:3]:
                search(questions[qid].text)
            times = []
            for qid in ordered:
                started = time.perf_counter()
                ranking = search(questions[qid].text)
                times.append((time.perf_counter() - started) * 1000)
                row = {"dataset": name, "method": method, "query_id": qid,
                       "split": "dev" if qid in dev else "test", **metrics(ranking, questions[qid])}
                if method in ("improved_graph", "bm25"):
                    row.update(diagnostics(improved, text_by_id, questions[qid], ranking))
                per_query.append(row)
            latency[method] = statistics.fmean(times)
    summary_rows = []
    for method in searchers:
        for split in ("dev", "test"):
            rows = [r for r in per_query if r["method"] == method and r["split"] == split]
            summary_rows.append({"dataset": name, "method": method, "split": split, "n_queries": len(rows),
                                 **{m: statistics.fmean(r[m] for r in rows) for m in METRICS},
                                 "latency_mean_ms": latency[method]})
    value = {(r["method"], r["query_id"]): r for r in per_query}
    test_ids = sorted(test, key=lambda q: (len(q), q))
    intervals = []
    for method in searchers:
        for other in ("bm25", "improved_graph"):
            if method == other:
                continue
            deltas = [value[(method, q)]["ndcg@10"] - value[(other, q)]["ndcg@10"] for q in test_ids]
            low, high = paired_bootstrap(deltas, [questions[q].group for q in test_ids])
            intervals.append({"dataset": name, "comparison": f"{method} minus {other}", "metric": "ndcg@10",
                              "difference": statistics.fmean(deltas), "wins": sum(d > 0 for d in deltas),
                              "losses": sum(d < 0 for d in deltas),
                              "bootstrap_95pct_low": low, "bootstrap_95pct_high": high})
    lengths = [len(r["text"].split()) for r in corpus]
    info = {"dataset": DATASETS[name], "corpus_passages": len(corpus), "questions": len(questions),
            "dev": len(dev), "test": len(test),
            "questions_per_gold_passage": len(questions) / len({q.group for q in questions.values()}),
            "passage_words_median": statistics.median(lengths),
            "passage_words_cv": statistics.pstdev(lengths) / statistics.fmean(lengths),
            "graph": improved.stats}
    return per_query, summary_rows, intervals, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/finance_graph"))
    args = parser.parse_args()
    all_rows, all_summary, all_intervals, infos = [], [], [], {}
    for name in DATASETS:
        per_query, summary_rows, intervals, info = evaluate_dataset(name)
        all_rows += per_query
        all_summary += summary_rows
        all_intervals += intervals
        infos[name] = info
        for row in summary_rows:
            if row["split"] == "test":
                print(f"{name:12s} {row['method']:24s} nDCG@10={row['ndcg@10']:.4f} R@1={row['recall@1']:.4f} "
                      f"R@10={row['recall@10']:.4f} R@100={row['recall@100']:.4f} {row['latency_mean_ms']:.1f} ms",
                      flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("per_query.csv", all_rows), ("metrics.csv", all_summary),
                           ("paired_intervals.csv", all_intervals)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)
    summary = {"datasets": infos, "split": "questions grouped by gold passage; seed 42; 20% development",
               "bm25": "team_bm25.py (Florence, ce54a01), k1=0.9, b=0.4", "dense": DENSE_MODEL,
               "graph_settings": StandaloneGraphSettings().__dict__,
               "timing": "one CPU pass over all questions (4 BLAS threads), top-100, after 3 warm-ups"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in all_intervals:
        print(f"{row['dataset']:12s} {row['comparison']:42s} {row['difference']:+.4f} "
              f"[{row['bootstrap_95pct_low']:+.4f}, {row['bootstrap_95pct_high']:+.4f}] {row['wins']}W/{row['losses']}L")


if __name__ == "__main__":
    main()
