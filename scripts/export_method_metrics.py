"""Convert one method's raw experiment output to the team CSV contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "schema_version", "owner", "method_id", "method", "dataset_id",
    "corpus_id", "split_id", "split", "k", "n_queries",
    "precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr_at_k",
    "ndcg_at_k", "latency_ms_per_query", "answer_n",
    "answer_exact_match", "answer_token_f1", "answer_semantic_similarity",
    "generator_model", "answer_status", "answer_protocol_id",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-csv", type=Path, required=True)
    parser.add_argument("--answer-csv", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/benchmark_config.json"))
    parser.add_argument("--method", required=True, help="Exact method label in the raw CSV")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--answer-status", choices=["not_run", "pilot", "final"], default="not_run"
    )
    parser.add_argument(
        "--answer-protocol-id",
        default="",
        help="Shared prompt/sample/generator protocol; required for final answer scores",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw = pd.read_csv(args.retrieval_csv)
    rows = raw[(raw["method"] == args.method) & (raw["split"] == "test")].copy()
    if rows.empty:
        available = sorted(raw["method"].dropna().unique())
        raise SystemExit(f"Method {args.method!r} not found. Available methods: {available}")

    rows.insert(0, "schema_version", config["schema_version"])
    rows.insert(1, "owner", args.owner)
    rows.insert(2, "method_id", args.method_id)
    rows.insert(4, "dataset_id", config["dataset_id"])
    rows.insert(5, "corpus_id", config["corpus_id"])
    rows.insert(6, "split_id", config["split_id"])

    answer = None
    if args.answer_csv and args.answer_csv.exists():
        answer_raw = pd.read_csv(args.answer_csv)
        matched = answer_raw[answer_raw["method"] == args.method]
        if not matched.empty:
            answer = matched.iloc[0]
    if answer is not None and args.answer_status == "not_run":
        raise SystemExit("Set --answer-status pilot or final when --answer-csv supplies scores")
    if args.answer_status == "final" and not args.answer_protocol_id:
        raise SystemExit("--answer-protocol-id is required for final answer scores")

    rows["answer_n"] = answer["n_answer_queries"] if answer is not None else pd.NA
    rows["answer_exact_match"] = answer["answer_exact_match"] if answer is not None else pd.NA
    rows["answer_token_f1"] = answer["answer_token_f1"] if answer is not None else pd.NA
    rows["answer_semantic_similarity"] = (
        answer["answer_semantic_similarity"] if answer is not None else pd.NA
    )
    rows["generator_model"] = answer["generator"] if answer is not None else pd.NA
    rows["answer_status"] = args.answer_status if answer is not None else "not_run"
    rows["answer_protocol_id"] = args.answer_protocol_id if answer is not None else pd.NA

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows[METRIC_COLUMNS].sort_values("k").to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
