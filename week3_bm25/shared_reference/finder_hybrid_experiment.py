"""FinDER hybrid retrieval experiment for Barclays MSCF capstone.

This experiment uses the deduplicated union of FinDER's expert-annotated
``references`` as the retrieval corpus.  Each query's own reference passage(s)
are its exact binary relevance labels.  This makes evaluation reproducible and
avoids fuzzy matching between annotations and newly parsed 10-K chunks.

The resulting benchmark is appropriate for comparing team retrieval methods
that use the same corpus.  It is not directly comparable to the FinDER paper's
full-10-K RAGAS Context Recall numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


TOKEN_RE = re.compile(r"[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*")


def normalize_text(text: str) -> str:
    """Collapse whitespace without changing the underlying evidence."""
    return " ".join(str(text).split())


def build_reference_corpus(df: pd.DataFrame):
    """Return unique evidence passages, qrels, and a stable passage ID map."""
    corpus: list[str] = []
    passage_to_id: dict[str, int] = {}
    qrels: list[set[int]] = []

    for refs in df["references"]:
        relevant: set[int] = set()
        for ref in refs:
            passage = normalize_text(ref)
            if passage not in passage_to_id:
                passage_to_id[passage] = len(corpus)
                corpus.append(passage)
            relevant.add(passage_to_id[passage])
        qrels.append(relevant)
    return corpus, qrels


def deterministic_split(ids: list[str]) -> np.ndarray:
    """Stable 20% development split; remaining 80% is the held-out test set."""
    return np.array(
        ["dev" if int(hashlib.sha1(x.encode()).hexdigest()[:8], 16) % 5 == 0 else "test" for x in ids]
    )


def topk_from_scores(scores: np.ndarray, depth: int) -> np.ndarray:
    """Return descending top-depth document indices for every score row."""
    depth = min(depth, scores.shape[1])
    idx = np.argpartition(scores, -depth, axis=1)[:, -depth:]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(scores[row, idx], axis=1)[:, ::-1]
    return idx[row, order]


def build_bm25(corpus: list[str], k1: float = 1.2, b: float = 0.75):
    vectorizer = CountVectorizer(
        lowercase=True,
        tokenizer=lambda s: TOKEN_RE.findall(s.lower()),
        token_pattern=None,
        min_df=1,
        dtype=np.float32,
    )
    counts = vectorizer.fit_transform(corpus).tocsr().astype(np.float32)
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    avg_len = float(doc_len.mean())
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    n_docs = counts.shape[0]
    idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)

    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    tf = counts.data
    norm = k1 * (1.0 - b + b * doc_len[rows] / avg_len)
    counts.data = (tf * (k1 + 1.0) / (tf + norm)) * idf[counts.indices]
    return vectorizer, counts


def retrieve_bm25(
    vectorizer: CountVectorizer,
    bm25_matrix: sparse.csr_matrix,
    queries: list[str],
    depth: int,
    batch_size: int = 256,
):
    rankings = []
    start = time.perf_counter()
    for i in range(0, len(queries), batch_size):
        q = vectorizer.transform(queries[i : i + batch_size]).tocsr()
        q.data[:] = 1.0
        scores = (q @ bm25_matrix.T).toarray()
        rankings.append(topk_from_scores(scores, depth))
    elapsed = time.perf_counter() - start
    return np.vstack(rankings), 1000.0 * elapsed / len(queries)


def retrieve_dense(
    model: SentenceTransformer,
    corpus_embeddings: np.ndarray,
    queries: list[str],
    depth: int,
    batch_size: int = 128,
):
    rankings = []
    start = time.perf_counter()
    for i in range(0, len(queries), batch_size):
        q_emb = model.encode(
            queries[i : i + batch_size],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = q_emb @ corpus_embeddings.T
        rankings.append(topk_from_scores(scores, depth))
    elapsed = time.perf_counter() - start
    return np.vstack(rankings), 1000.0 * elapsed / len(queries)


def weighted_rrf(
    bm25_rankings: np.ndarray,
    dense_rankings: np.ndarray,
    dense_weight: float,
    output_depth: int,
    rrf_k: int = 60,
):
    """Weighted reciprocal rank fusion; dense_weight=1 means dense only."""
    all_rows = []
    start = time.perf_counter()
    for b_rank, d_rank in zip(bm25_rankings, dense_rankings):
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(b_rank, start=1):
            scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + (1.0 - dense_weight) / (rrf_k + rank)
        for rank, doc_id in enumerate(d_rank, start=1):
            scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + dense_weight / (rrf_k + rank)
        ordered = sorted(scores, key=scores.get, reverse=True)[:output_depth]
        all_rows.append(ordered)
    elapsed = time.perf_counter() - start
    return np.asarray(all_rows, dtype=np.int32), 1000.0 * elapsed / len(all_rows)


def query_metrics(ranking: np.ndarray, relevant: set[int], k: int):
    top = list(map(int, ranking[:k]))
    hits = [1 if x in relevant else 0 for x in top]
    hit_count = sum(hits)
    precision = hit_count / k
    recall = hit_count / len(relevant)
    hit_rate = float(hit_count > 0)
    reciprocal_rank = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, hit_rate, reciprocal_rank, ndcg


def evaluate_rankings(
    name: str,
    rankings: np.ndarray,
    qrels: list[set[int]],
    mask: np.ndarray,
    latency_ms: float,
    ks=(1, 3, 5, 10),
):
    rows = []
    indices = np.where(mask)[0]
    for k in ks:
        values = np.array([query_metrics(rankings[i], qrels[i], k) for i in indices])
        rows.append(
            {
                "method": name,
                "k": k,
                "n_queries": len(indices),
                "precision_at_k": values[:, 0].mean(),
                "recall_at_k": values[:, 1].mean(),
                "hit_rate_at_k": values[:, 2].mean(),
                "mrr_at_k": values[:, 3].mean(),
                "ndcg_at_k": values[:, 4].mean(),
                "latency_ms_per_query": latency_ms,
            }
        )
    return rows


def normalize_answer(s: str) -> list[str]:
    return re.findall(r"[a-z0-9.%$-]+", str(s).lower())


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def generate_answers(
    method_name: str,
    rankings: np.ndarray,
    sample_indices: np.ndarray,
    queries: list[str],
    answers: list[str],
    corpus: list[str],
    embedding_model: SentenceTransformer,
    generator_name: str,
    top_k_context: int,
    max_input_tokens: int,
):
    """Run a small, reproducible local generation baseline."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    generator = AutoModelForSeq2SeqLM.from_pretrained(generator_name)
    generator.eval()

    predictions = []
    start = time.perf_counter()
    for idx in sample_indices:
        contexts = [corpus[int(x)] for x in rankings[idx, :top_k_context]]
        prompt = (
            "Answer the financial question using only the context. Give a concise answer and calculate if needed.\n"
            f"Question: {queries[idx]}\nContext: " + "\n---\n".join(contexts) + "\nAnswer:"
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        with torch.inference_mode():
            output = generator.generate(**encoded, max_new_tokens=96, do_sample=False)
        predictions.append(tokenizer.decode(output[0], skip_special_tokens=True).strip())
    generation_latency = 1000.0 * (time.perf_counter() - start) / len(sample_indices)

    gold = [answers[i] for i in sample_indices]
    pred_emb = embedding_model.encode(predictions, normalize_embeddings=True, show_progress_bar=False)
    gold_emb = embedding_model.encode(gold, normalize_embeddings=True, show_progress_bar=False)
    semantic = np.sum(pred_emb * gold_emb, axis=1)
    f1 = np.array([token_f1(p, g) for p, g in zip(predictions, gold)])
    exact = np.array([normalize_answer(p) == normalize_answer(g) for p, g in zip(predictions, gold)])

    detail = pd.DataFrame(
        {
            "query_index": sample_indices,
            "method": method_name,
            "question": [queries[i] for i in sample_indices],
            "gold_answer": gold,
            "generated_answer": predictions,
            "answer_token_f1": f1,
            "answer_semantic_similarity": semantic,
            "exact_match": exact.astype(int),
        }
    )
    summary = {
        "method": method_name,
        "n_answer_queries": len(sample_indices),
        "generator": generator_name,
        "answer_token_f1": float(f1.mean()),
        "answer_semantic_similarity": float(semantic.mean()),
        "answer_exact_match": float(exact.mean()),
        "generation_latency_ms_per_query": generation_latency,
    }
    return detail, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu", help="Use cpu for the most reproducible team comparison")
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--answer-sample-size", type=int, default=100)
    parser.add_argument("--generator-model", default="google/flan-t5-small")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    queries = df["text"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()
    corpus, qrels = build_reference_corpus(df)
    split = deterministic_split(df["_id"].astype(str).tolist())
    dev_mask, test_mask, all_mask = split == "dev", split == "test", np.ones(len(df), dtype=bool)

    dataset_summary = {
        "n_queries": len(df),
        "n_unique_reference_passages": len(corpus),
        "n_dev": int(dev_mask.sum()),
        "n_test": int(test_mask.sum()),
        "mean_references_per_query": float(np.mean([len(x) for x in qrels])),
        "corpus_mode": "deduplicated FinDER references",
    }
    (out / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))

    # Sparse retrieval.
    bm25_start = time.perf_counter()
    vectorizer, bm25_matrix = build_bm25(corpus)
    bm25_index_seconds = time.perf_counter() - bm25_start
    bm25_rank, bm25_latency = retrieve_bm25(
        vectorizer, bm25_matrix, queries, args.candidate_depth
    )

    # Dense retrieval.
    model = SentenceTransformer(args.embedding_model, device=args.device)
    dense_start = time.perf_counter()
    corpus_embeddings = model.encode(
        corpus,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    dense_index_seconds = time.perf_counter() - dense_start
    dense_rank, dense_latency = retrieve_dense(
        model, corpus_embeddings, queries, args.candidate_depth
    )

    # Tune only the dense share of weighted RRF on the development split.
    tuning_rows = []
    hybrid_candidates = {}
    for alpha in (0.25, 0.50, 0.75):
        rank, fusion_latency = weighted_rrf(
            bm25_rank, dense_rank, alpha, args.candidate_depth
        )
        hybrid_candidates[alpha] = (rank, fusion_latency)
        metrics = evaluate_rankings(
            f"Hybrid RRF alpha={alpha:.2f}", rank, qrels, dev_mask, fusion_latency, ks=(5,)
        )[0]
        tuning_rows.append(metrics | {"dense_weight": alpha})
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["recall_at_k", "ndcg_at_k"], ascending=False
    )
    best_alpha = float(tuning.iloc[0]["dense_weight"])
    hybrid_rank, fusion_latency = hybrid_candidates[best_alpha]
    tuning.to_csv(out / "hybrid_weight_tuning_dev.csv", index=False)

    latency = {
        "BM25": bm25_latency,
        "Dense MiniLM": dense_latency,
        "Hybrid RRF": bm25_latency + dense_latency + fusion_latency,
    }
    rankings = {
        "BM25": bm25_rank,
        "Dense MiniLM": dense_rank,
        f"Hybrid RRF alpha={best_alpha:.2f}": hybrid_rank,
    }

    metric_rows = []
    for split_name, mask in (("dev", dev_mask), ("test", test_mask), ("all", all_mask)):
        for method, rank in rankings.items():
            key = "Hybrid RRF" if method.startswith("Hybrid") else method
            rows = evaluate_rankings(method, rank, qrels, mask, latency[key])
            for row in rows:
                row["split"] = split_name
            metric_rows.extend(rows)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "retrieval_metrics.csv", index=False)

    # Per-query top-10 IDs and exact qrels for reproducibility and error analysis.
    detail_rows = []
    for i, row in df.iterrows():
        item = {
            "query_index": i,
            "query_id": row["_id"],
            "split": split[i],
            "question": row["text"],
            "category": row["category"],
            "reasoning": bool(row["reasoning"]),
            "gold_passage_ids": json.dumps(sorted(qrels[i])),
        }
        for method, rank in rankings.items():
            slug = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
            top10 = list(map(int, rank[i, :10]))
            item[f"{slug}_top10"] = json.dumps(top10)
            item[f"{slug}_first_relevant_rank"] = next(
                (j + 1 for j, doc_id in enumerate(rank[i]) if int(doc_id) in qrels[i]), 0
            )
        detail_rows.append(item)
    details = pd.DataFrame(detail_rows)
    details.to_csv(out / "per_query_rankings.csv", index=False)

    # Human-readable failure examples for the selected hybrid model.
    hybrid_slug = re.sub(r"[^a-z0-9]+", "_", list(rankings)[-1].lower()).strip("_")
    failures = details[
        (details["split"] == "test") & (details[f"{hybrid_slug}_first_relevant_rank"] == 0)
    ].head(25).copy()
    failures.to_csv(out / "hybrid_top100_failures.csv", index=False)

    answer_summary = []
    if not args.skip_generation and args.answer_sample_size > 0:
        test_indices = np.where(test_mask)[0]
        rng = np.random.default_rng(20260915)
        sample = np.sort(rng.choice(test_indices, size=min(args.answer_sample_size, len(test_indices)), replace=False))
        for method, rank in rankings.items():
            detail, summary = generate_answers(
                method,
                rank,
                sample,
                queries,
                answers,
                corpus,
                model,
                args.generator_model,
                top_k_context=3,
                max_input_tokens=512,
            )
            detail.to_csv(
                out / (re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_") + "_answers.csv"),
                index=False,
            )
            answer_summary.append(summary)
        pd.DataFrame(answer_summary).to_csv(out / "answer_correctness.csv", index=False)

    run_summary = {
        **dataset_summary,
        "embedding_model": args.embedding_model,
        "device": args.device,
        "bm25": {"k1": 1.2, "b": 0.75},
        "hybrid": {"fusion": "weighted reciprocal rank fusion", "rrf_k": 60, "best_dense_weight": best_alpha},
        "candidate_depth": args.candidate_depth,
        "bm25_index_seconds": bm25_index_seconds,
        "dense_index_seconds": dense_index_seconds,
        "answer_generation": answer_summary,
    }
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2))

    packages = [
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
        "sentence-transformers", "transformers", "torch",
    ]
    environment = {"python": __import__("sys").version, "device": args.device, "packages": {}}
    for package in packages:
        try:
            environment["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][package] = None
    (out / "runtime_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))
    print("\nHeld-out test metrics at k=5")
    print(metrics_df[(metrics_df.split == "test") & (metrics_df.k == 5)].to_string(index=False))


if __name__ == "__main__":
    main()
