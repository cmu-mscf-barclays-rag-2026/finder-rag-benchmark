"""Validate saved Person 3 results without rerunning the models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_METHODS = {
    "Question BM25",
    "Question Dense MiniLM",
    "Question BM25 + Question Dense RRF",
    "Reference Answer Dense (oracle)",
    "Question-only HyDE Dense",
    "Question BM25 + HyDE Dense RRF",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "person3_hyde",
    )
    args = parser.parse_args()

    metrics = pd.read_csv(args.results_dir / "retrieval_metrics.csv")
    details = pd.read_csv(args.results_dir / "per_query_results.csv")
    generated = pd.read_csv(args.results_dir / "generated_hyde_answers.csv")

    if len(details) != 100 or len(generated) != 100:
        raise AssertionError("Expected exactly 100 Legal RAG Bench questions")
    if set(metrics.method) != EXPECTED_METHODS:
        raise AssertionError("Saved method set is incomplete or unexpected")
    if generated.hyde_text.isna().any() or (generated.hyde_text.str.strip() == "").any():
        raise AssertionError("One or more generated HyDE texts are empty")
    if "generator_revision" not in generated:
        raise AssertionError("Generated cache does not pin a model revision")
    if not generated.prompt_style.eq("concise_legal").all():
        raise AssertionError("Primary saved results must use concise_legal")

    rank_columns = {
        "Question BM25": "question_bm25_gold_rank",
        "Question Dense MiniLM": "question_dense_gold_rank",
        "Question BM25 + Question Dense RRF": "standard_hybrid_gold_rank",
        "Reference Answer Dense (oracle)": "oracle_answer_dense_gold_rank",
        "Question-only HyDE Dense": "hyde_dense_gold_rank",
        "Question BM25 + HyDE Dense RRF": "hyde_hybrid_gold_rank",
    }
    for method, rank_column in rank_columns.items():
        ranks = details[rank_column].to_numpy()
        for k in (1, 3, 5, 10):
            hit = (ranks > 0) & (ranks <= k)
            mrr = np.where(hit, 1 / np.maximum(ranks, 1), 0).mean()
            ndcg = np.where(hit, 1 / np.log2(np.maximum(ranks, 1) + 1), 0).mean()
            saved = metrics[(metrics.method == method) & (metrics.k == k)].iloc[0]
            np.testing.assert_allclose(saved.recall_at_k, hit.mean(), atol=1e-12)
            np.testing.assert_allclose(saved.mrr_at_k, mrr, atol=1e-12)
            np.testing.assert_allclose(saved.ndcg_at_k, ndcg, atol=1e-12)

    oracle_status = metrics.loc[
        metrics.method == "Reference Answer Dense (oracle)", "status"
    ]
    if not oracle_status.eq("oracle_only").all():
        raise AssertionError("Oracle result must never be marked deployable")
    print("PASS: saved Person 3 results are internally consistent (100 questions).")


if __name__ == "__main__":
    main()
