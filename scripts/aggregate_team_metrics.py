"""Validate team metric files and build presentation-ready comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from export_method_metrics import METRIC_COLUMNS


BENCHMARK_KEYS = ["schema_version", "dataset_id", "corpus_id", "split_id"]
RETRIEVAL_COLUMNS = [
    "precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr_at_k",
    "ndcg_at_k", "latency_ms_per_query",
]
ANSWER_COLUMNS = [
    "answer_exact_match", "answer_token_f1", "answer_semantic_similarity",
]


def validate(df: pd.DataFrame) -> None:
    missing = [column for column in METRIC_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if set(df["split"].dropna()) != {"test"}:
        raise ValueError("Final team submissions must contain only split=test rows")
    for key in BENCHMARK_KEYS:
        values = df[key].dropna().astype(str).unique()
        if len(values) != 1:
            raise ValueError(f"Mixed benchmark values in {key}: {values.tolist()}")
    if df.duplicated(["method_id", "k"]).any():
        duplicates = df.loc[df.duplicated(["method_id", "k"], keep=False), ["method_id", "k"]]
        raise ValueError(f"Duplicate method/K rows:\n{duplicates.to_string(index=False)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("team_metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("presentation"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    files = sorted(path for path in args.input_dir.glob("*.csv") if not path.name.startswith("_"))
    if not files:
        raise SystemExit(f"No CSV submissions found in {args.input_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    validate(combined)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["method_id", "k"]).to_csv(
        args.output_dir / "combined_metrics_long.csv", index=False
    )

    selected = combined[combined["k"] == args.k].copy()
    if selected.empty:
        raise ValueError(f"No rows found for K={args.k}")
    selected = selected.sort_values(["recall_at_k", "ndcg_at_k"], ascending=False)
    display_columns = ["owner", "method", "n_queries", *RETRIEVAL_COLUMNS, "answer_status"]
    selected[display_columns].to_csv(
        args.output_dir / f"comparison_k{args.k}.csv", index=False
    )

    best = selected.iloc[0]
    final_answers = selected[selected["answer_status"] == "final"].copy()
    if len(final_answers) == len(selected):
        protocols = final_answers["answer_protocol_id"].dropna().unique()
        if len(protocols) != 1:
            raise ValueError(f"Final answer rows use different protocols: {protocols.tolist()}")
        answer_columns = [
            "owner", "method", "answer_n", *ANSWER_COLUMNS,
            "generator_model", "answer_protocol_id",
        ]
        final_answers[answer_columns].to_csv(
            args.output_dir / "answer_comparison.csv", index=False
        )
        answer_note = "Final answer scores passed the shared answer-protocol check."
    else:
        pd.DataFrame(columns=[
            "owner", "method", "answer_n", *ANSWER_COLUMNS,
            "generator_model", "answer_protocol_id",
        ]).to_csv(args.output_dir / "answer_comparison.csv", index=False)
        answer_note = (
            "Answer scores are not yet comparable. The answer table stays empty until every "
            "method is marked final under one shared protocol."
        )
    table = selected[display_columns].copy()
    for column in table.select_dtypes(include="number").columns:
        table[column] = table[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = list(table.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    markdown_table = "\n".join(markdown_rows)

    summary = f"""# FinDER retrieval comparison

This table covers the compatible CSV submissions currently in `team_metrics/`.
See [the all-four team overview](team_overview.md) for every contribution,
including results awaiting a compatible export. Source commits are recorded in
[provenance.json](../team_metrics/provenance.json).

All included rows passed the shared benchmark identifier checks for dataset,
corpus, and split. This is a comparison of the included submissions, not a claim
that every team member has a compatible result. Blank metrics are unreported, not zero.
Latency values use different hardware/protocols and must not be ranked as speed.

## Primary comparison at K={args.k}

{markdown_table}

Among the included submissions, the highest Recall@{args.k} is **{best['method']}** at **{best['recall_at_k']:.4f}**.
The same row has nDCG@{args.k} of **{best['ndcg_at_k']:.4f}** and mean retrieval
latency of **{best['latency_ms_per_query']:.2f} ms/query** on the contributor's hardware.

{answer_note}
"""
    (args.output_dir / "presentation_summary.md").write_text(summary, encoding="utf-8")
    print(f"Validated {len(files)} submission file(s) and {len(combined)} metric rows")
    print(f"Presentation table: {args.output_dir / f'comparison_k{args.k}.csv'}")


if __name__ == "__main__":
    main()
