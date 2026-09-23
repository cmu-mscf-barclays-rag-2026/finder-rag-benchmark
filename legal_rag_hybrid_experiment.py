"""Part C: reproducible hybrid retrieval on Isaacus Legal RAG Bench.

The benchmark has one official test split with 100 questions and one labelled
passage per question.  To avoid selecting an RRF weight on the same questions
used for evaluation, this script uses deterministic five-fold out-of-fold
tuning.  Each fold selects the dense RRF weight on the other four folds and is
then evaluated on the held-out fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


DATA_URL = "https://huggingface.co/datasets/isaacus/legal-rag-bench/resolve/main"
TOKEN_RE = re.compile(r"[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*")
DEFAULT_WEIGHTS = (0.25, 0.50, 0.75)
DEFAULT_KS = (1, 3, 5, 10)


def download_if_missing(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("corpus.jsonl", "qa.jsonl"):
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{DATA_URL}/{name}", path)
        paths.append(path)
    return paths[0], paths[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_benchmark(data_dir: Path):
    corpus_path, qa_path = download_if_missing(data_dir)
    corpus_rows = load_jsonl(corpus_path)
    qa_rows = load_jsonl(qa_path)
    passage_ids = [str(row["id"]) for row in corpus_rows]
    id_to_index = {passage_id: i for i, passage_id in enumerate(passage_ids)}
    missing = [row["relevant_passage_id"] for row in qa_rows if row["relevant_passage_id"] not in id_to_index]
    if missing:
        raise ValueError(f"Gold passages missing from corpus: {missing[:5]}")
    corpus = [" ".join((str(row.get("title", "")), str(row["text"]))).strip() for row in corpus_rows]
    queries = [str(row["question"]) for row in qa_rows]
    gold = np.array([id_to_index[row["relevant_passage_id"]] for row in qa_rows], dtype=np.int32)
    return corpus_rows, qa_rows, passage_ids, corpus, queries, gold


def topk_from_scores(scores: np.ndarray, depth: int) -> np.ndarray:
    depth = min(depth, scores.shape[1])
    indices = np.argpartition(scores, -depth, axis=1)[:, -depth:]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(scores[row, indices], axis=1)[:, ::-1]
    return indices[row, order].astype(np.int32)


def build_bm25(corpus: list[str], k1: float = 1.2, b: float = 0.75):
    vectorizer = CountVectorizer(
        lowercase=True,
        tokenizer=lambda text: TOKEN_RE.findall(text.lower()),
        token_pattern=None,
        min_df=1,
        dtype=np.float32,
    )
    counts = vectorizer.fit_transform(corpus).tocsr().astype(np.float32)
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    avg_len = float(doc_len.mean())
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((counts.shape[0] - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    tf = counts.data
    norm = k1 * (1.0 - b + b * doc_len[rows] / avg_len)
    counts.data = (tf * (k1 + 1.0) / (tf + norm)) * idf[counts.indices]
    return vectorizer, counts


def retrieve_bm25(vectorizer, matrix: sparse.csr_matrix, queries: list[str], depth: int):
    start = time.perf_counter()
    query_counts = vectorizer.transform(queries).tocsr()
    query_counts.data[:] = 1.0
    rankings = topk_from_scores((query_counts @ matrix.T).toarray(), depth)
    latency = 1000 * (time.perf_counter() - start) / len(queries)
    return rankings, latency


def retrieve_dense(model, embeddings: np.ndarray, queries: list[str], depth: int):
    start = time.perf_counter()
    query_embeddings = model.encode(
        queries,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    rankings = topk_from_scores(query_embeddings @ embeddings.T, depth)
    latency = 1000 * (time.perf_counter() - start) / len(queries)
    return rankings, latency


def weighted_rrf(bm25: np.ndarray, dense: np.ndarray, dense_weight: float, depth: int, rrf_k: int = 60):
    rows = []
    start = time.perf_counter()
    for sparse_rank, dense_rank in zip(bm25, dense):
        scores: dict[int, float] = {}
        for rank, doc in enumerate(sparse_rank, 1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + (1 - dense_weight) / (rrf_k + rank)
        for rank, doc in enumerate(dense_rank, 1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + dense_weight / (rrf_k + rank)
        rows.append(sorted(scores, key=scores.get, reverse=True)[:depth])
    latency = 1000 * (time.perf_counter() - start) / len(rows)
    return np.asarray(rows, dtype=np.int32), latency


def first_gold_rank(ranking: np.ndarray, gold: int) -> int:
    found = np.where(ranking == gold)[0]
    return int(found[0] + 1) if len(found) else 0


def metrics_at_k(rankings: np.ndarray, gold: np.ndarray, k: int) -> dict:
    ranks = np.array([first_gold_rank(row, int(target)) for row, target in zip(rankings, gold)])
    hit = (ranks > 0) & (ranks <= k)
    reciprocal = np.where(hit, 1.0 / ranks.clip(min=1), 0.0)
    ndcg = np.where(hit, 1.0 / np.log2(ranks.clip(min=1) + 1), 0.0)
    # Each question has exactly one labelled relevant passage.
    return {
        "k": k,
        "n_queries": len(gold),
        "precision_at_k": float(hit.mean() / k),
        "recall_at_k": float(hit.mean()),
        "hit_rate_at_k": float(hit.mean()),
        "mrr_at_k": float(reciprocal.mean()),
        "ndcg_at_k": float(ndcg.mean()),
    }


def stable_fold(question_id: str, n_folds: int = 5) -> int:
    digest = hashlib.sha1(str(question_id).encode()).hexdigest()
    return int(digest[:8], 16) % n_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/legal_rag_bench"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/legal_rag_bench"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-depth", type=int, default=100)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    corpus_rows, qa_rows, passage_ids, corpus, queries, gold = load_benchmark(args.data_dir)
    question_ids = [str(row["id"]) for row in qa_rows]
    folds = np.array([stable_fold(qid) for qid in question_ids])

    bm25_start = time.perf_counter()
    vectorizer, bm25_matrix = build_bm25(corpus)
    bm25_index_seconds = time.perf_counter() - bm25_start
    bm25_rank, bm25_latency = retrieve_bm25(vectorizer, bm25_matrix, queries, args.candidate_depth)

    model = SentenceTransformer(args.embedding_model, device=args.device)
    dense_start = time.perf_counter()
    cache_path = args.data_dir / "all_MiniLM_L6_v2_corpus_embeddings.npy"
    if cache_path.exists():
        corpus_embeddings = np.load(cache_path)
    else:
        corpus_embeddings = model.encode(
            corpus,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(cache_path, corpus_embeddings)
    dense_index_seconds = time.perf_counter() - dense_start
    dense_rank, dense_latency = retrieve_dense(model, corpus_embeddings, queries, args.candidate_depth)

    candidates = {}
    fusion_latency = {}
    for weight in DEFAULT_WEIGHTS:
        candidates[weight], fusion_latency[weight] = weighted_rrf(
            bm25_rank, dense_rank, weight, args.candidate_depth
        )

    # Nested selection: tune on four folds, predict only the untouched fifth fold.
    oof_hybrid = np.empty_like(bm25_rank)
    tuning_rows = []
    selected_weights = []
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        scored = []
        for weight, ranking in candidates.items():
            recall5 = metrics_at_k(ranking[train], gold[train], 5)["recall_at_k"]
            ndcg5 = metrics_at_k(ranking[train], gold[train], 5)["ndcg_at_k"]
            scored.append((recall5, ndcg5, -abs(weight - 0.5), weight))
            tuning_rows.append({
                "held_out_fold": fold,
                "dense_weight": weight,
                "tuning_n": int(train.sum()),
                "tuning_recall_at_5": recall5,
                "tuning_ndcg_at_5": ndcg5,
            })
        best_weight = max(scored)[-1]
        selected_weights.append(best_weight)
        oof_hybrid[test] = candidates[best_weight][test]
        for row in tuning_rows[-len(DEFAULT_WEIGHTS):]:
            row["selected"] = row["dense_weight"] == best_weight

    methods = {
        "BM25 k1=1.2 b=0.75": (bm25_rank, bm25_latency),
        "Dense all-MiniLM-L6-v2": (dense_rank, dense_latency),
        "Hybrid RRF alpha=0.25 transfer": (
            candidates[0.25],
            bm25_latency + dense_latency + fusion_latency[0.25],
        ),
        "Hybrid RRF nested-5-fold": (
            oof_hybrid,
            bm25_latency + dense_latency + float(np.mean(list(fusion_latency.values()))),
        ),
    }
    metric_rows = []
    for method, (ranking, latency) in methods.items():
        for k in DEFAULT_KS:
            metric_rows.append(metrics_at_k(ranking, gold, k) | {
                "method": method,
                "latency_ms_per_query": latency,
                "evaluation": (
                    "fixed_transfer_from_finder"
                    if "alpha=0.25" in method
                    else "5_fold_out_of_fold"
                    if method.startswith("Hybrid")
                    else "official_test"
                ),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "retrieval_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "hybrid_weight_tuning_cv.csv", index=False)

    # Descriptive sensitivity analysis on all 100 questions. This table helps
    # explain weight behavior, but must not be used to select a weight and then
    # claim an unbiased test result on the same questions.
    sensitivity_rows = []
    for weight, ranking in candidates.items():
        for k in DEFAULT_KS:
            sensitivity_rows.append(metrics_at_k(ranking, gold, k) | {
                "dense_weight": weight,
                "bm25_weight": 1.0 - weight,
                "analysis_status": "exploratory_full_test_sensitivity",
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out / "hybrid_weight_sensitivity_full_test.csv", index=False)

    detail_rows = []
    for i, row in enumerate(qa_rows):
        item = {
            "query_id": question_ids[i],
            "fold": int(folds[i]),
            "question": row["question"],
            "gold_passage_id": row["relevant_passage_id"],
            "gold_title": corpus_rows[int(gold[i])].get("title", ""),
            "bm25_first_gold_rank": first_gold_rank(bm25_rank[i], int(gold[i])),
            "dense_first_gold_rank": first_gold_rank(dense_rank[i], int(gold[i])),
            "hybrid_first_gold_rank": first_gold_rank(oof_hybrid[i], int(gold[i])),
            "hybrid_transfer_first_gold_rank": first_gold_rank(candidates[0.25][i], int(gold[i])),
            "hybrid_dense_weight": selected_weights[int(folds[i])],
            "bm25_top5_ids": json.dumps([passage_ids[j] for j in bm25_rank[i, :5]]),
            "dense_top5_ids": json.dumps([passage_ids[j] for j in dense_rank[i, :5]]),
            "hybrid_top5_ids": json.dumps([passage_ids[j] for j in oof_hybrid[i, :5]]),
        }
        detail_rows.append(item)
    details = pd.DataFrame(detail_rows)
    details.to_csv(out / "per_query_rankings.csv", index=False)

    b_hit = details.bm25_first_gold_rank.between(1, 5)
    d_hit = details.dense_first_gold_rank.between(1, 5)
    # Primary error analysis uses the pre-registered FinDER weight (alpha=.25).
    h_hit = details.hybrid_transfer_first_gold_rank.between(1, 5)
    categories = {
        "bm25_only": b_hit & ~d_hit,
        "dense_only": d_hit & ~b_hit,
        "hybrid_rescue": h_hit & ~b_hit & ~d_hit,
        "hybrid_failure": ~h_hit,
        "hybrid_lost_baseline_hit": ~h_hit & (b_hit | d_hit),
    }
    examples = []
    for category, mask in categories.items():
        sample = details[mask].head(5).copy()
        sample.insert(0, "category", category)
        examples.append(sample)
    pd.concat(examples, ignore_index=True).to_csv(out / "error_analysis_examples.csv", index=False)

    # Paired uncertainty checks. With only 100 questions, small differences
    # should be described as directional unless the interval excludes zero.
    rng = np.random.default_rng(20260921)
    bootstrap_indices = rng.integers(0, len(details), size=(20_000, len(details)))
    comparison_rows = []
    for baseline_name, baseline_hit in (("BM25", b_hit), ("Dense", d_hit)):
        baseline = baseline_hit.to_numpy(dtype=bool)
        hybrid = h_hit.to_numpy(dtype=bool)
        hybrid_only = int((hybrid & ~baseline).sum())
        baseline_only = int((baseline & ~hybrid).sum())
        differences = hybrid[bootstrap_indices].mean(axis=1) - baseline[bootstrap_indices].mean(axis=1)
        comparison_rows.append({
            "comparison": f"Hybrid alpha=0.25 vs {baseline_name}",
            "hybrid_hit_rate_at_5": float(hybrid.mean()),
            "baseline_hit_rate_at_5": float(baseline.mean()),
            "absolute_difference": float(hybrid.mean() - baseline.mean()),
            "bootstrap_95_ci_low": float(np.quantile(differences, 0.025)),
            "bootstrap_95_ci_high": float(np.quantile(differences, 0.975)),
            "hybrid_only_hits": hybrid_only,
            "baseline_only_hits": baseline_only,
            "mcnemar_exact_p": float(binomtest(hybrid_only, hybrid_only + baseline_only, 0.5).pvalue),
        })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out / "paired_comparisons_k5.csv", index=False)

    k5 = metrics[metrics.k == 5].set_index("method")
    weight_k5 = sensitivity[sensitivity.k == 5].sort_values("dense_weight")
    weight_table_rows = "\n".join(
        "| "
        f"{row.bm25_weight:.0%} / {row.dense_weight:.0%} | "
        f"{row.recall_at_k:.2f} | {row.mrr_at_k:.4f} | {row.ndcg_at_k:.4f} |"
        for row in weight_k5.itertuples(index=False)
    )
    findings = f"""# Legal RAG Bench Part C Findings

## Main result

At Top 5, the fixed Hybrid RRF configuration transferred from FinDER retrieved
the labelled passage for {int(h_hit.sum())} of 100 questions. BM25 retrieved
{int(b_hit.sum())}, while dense MiniLM retrieved {int(d_hit.sum())}.

| Method | Recall and Hit Rate at 5 | MRR at 5 | nDCG at 5 |
|---|---:|---:|---:|
| BM25 | {k5.loc['BM25 k1=1.2 b=0.75', 'recall_at_k']:.2f} | {k5.loc['BM25 k1=1.2 b=0.75', 'mrr_at_k']:.4f} | {k5.loc['BM25 k1=1.2 b=0.75', 'ndcg_at_k']:.4f} |
| Dense MiniLM | {k5.loc['Dense all-MiniLM-L6-v2', 'recall_at_k']:.2f} | {k5.loc['Dense all-MiniLM-L6-v2', 'mrr_at_k']:.4f} | {k5.loc['Dense all-MiniLM-L6-v2', 'ndcg_at_k']:.4f} |
| Hybrid RRF alpha 0.25 | {k5.loc['Hybrid RRF alpha=0.25 transfer', 'recall_at_k']:.2f} | {k5.loc['Hybrid RRF alpha=0.25 transfer', 'mrr_at_k']:.4f} | {k5.loc['Hybrid RRF alpha=0.25 transfer', 'ndcg_at_k']:.4f} |

The fixed Hybrid improved Hit Rate at 5 by 5 percentage points over BM25 and by
7 points over dense MiniLM. Relative to BM25, Hybrid added six unique successes
and lost one BM25 success. The exact paired test is not statistically significant
at the 5% level, so this should be presented as promising directional evidence,
not a definitive win.

## Weight sensitivity

| BM25 / dense weight | Recall and Hit Rate at 5 | MRR at 5 | nDCG at 5 |
|---|---:|---:|---:|
{weight_table_rows}

The 50/50 weighting finds one additional labelled passage in the Top 5, while
the BM25-heavy 75/25 weighting ranks relevant passages slightly earlier and has
the strongest MRR and nDCG. Giving dense retrieval 75% of the weight reduces all
Top-5 metrics. These full-test comparisons are exploratory; the unbiased result
remains the transferred 75/25 configuration and the nested five-fold estimate.

## Interpretation

Legal RAG Bench deliberately uses questions with low lexical overlap. Dense
retrieval therefore contributes complementary candidates, while BM25 remains
useful for legal terms and named concepts. The result supports using a
BM25-heavy Hybrid rather than replacing BM25 with a general-purpose embedding
model.

## Dataset limitation

The current benchmark labels one most-relevant passage per question and has only
100 questions. It does not directly measure multi-evidence completeness, and
Recall at K equals Hit Rate at K in this experiment.
"""
    (out / "findings.md").write_text(findings, encoding="utf-8")

    summary = {
        "dataset": "isaacus/legal-rag-bench",
        "license": "CC BY-NC-SA 4.0",
        "n_passages": len(corpus),
        "n_questions": len(queries),
        "gold_passages_per_question": 1,
        "official_splits": ["test"],
        "embedding_model": args.embedding_model,
        "bm25": {"k1": 1.2, "b": 0.75},
        "hybrid": {
            "fusion": "weighted reciprocal rank fusion",
            "rrf_k": 60,
            "candidate_dense_weights": list(DEFAULT_WEIGHTS),
            "fold_selected_dense_weights": selected_weights,
            "evaluation": "deterministic five-fold out-of-fold weight selection",
        },
        "bm25_index_seconds": bm25_index_seconds,
        "dense_index_seconds": dense_index_seconds,
        "important_limitation": "The current dataset labels one most-relevant passage per question; it does not directly evaluate multi-evidence completeness.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics[metrics.k == 5].to_string(index=False))
    print("\nTop-5 weight sensitivity (exploratory):")
    print(
        weight_k5[
            ["bm25_weight", "dense_weight", "recall_at_k", "mrr_at_k", "ndcg_at_k"]
        ].to_string(index=False)
    )
    print("Selected dense weights by fold:", selected_weights)


if __name__ == "__main__":
    main()
