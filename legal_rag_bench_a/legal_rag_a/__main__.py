"""Command line entry points for inspection and shared retrieval evaluation."""

import argparse
import csv
import json
from pathlib import Path

from .data import DEFAULT_DATA, ROOT, REVISION, download, load_benchmark, read_jsonl, sha256
from .evaluation import evaluate
from .profile import profile, team_split


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_profile(data_dir, out, config):
    corpus, queries = load_benchmark(data_dir)
    summary, passages, sections, query_rows, split = profile(corpus, queries, data_dir, config)
    write_json(out / "dataset_profile.json", summary)
    write_csv(out / "passage_lengths.csv", passages)
    write_csv(out / "section_lengths.csv", sections)
    write_csv(out / "query_lengths.csv", query_rows)
    write_csv(out / "query_splits.csv", split)
    by_id = {p["id"]: p for p in corpus}
    # IDs and field types give a compact schema example without copying source text.
    write_json(out / "schema_examples.json", {
        "corpus": {key: type(value).__name__ for key, value in corpus[0].items()},
        "query": {key: type(value).__name__ for key, value in queries[0].items()},
        "link_example": {"query_id": queries[0]["id"], "gold_ids": queries[0]["gold_ids"],
                         "resolved_passage_exists": queries[0]["gold_ids"][0] in by_id},
    })
    controls = []
    for name, predictions in (
        ("oracle_label_check_NOT_A_RETRIEVER", {q["id"]: q["gold_ids"] for q in queries}),
        ("empty_output_check_NOT_A_RETRIEVER", {q["id"]: [] for q in queries}),
    ):
        _, rows = evaluate(queries, predictions, by_id, config["evaluation_ks"])
        controls.extend({"control": name, **row} for row in rows)
    write_csv(out / "evaluator_controls.csv", controls)
    write_json(out / "profile_manifest.json", {
        "status": "Full dataset inspection and evaluator controls executed; no retrieval model benchmark.",
        "revision": REVISION, "config": config,
        "input_sha256": summary["source_sha256"],
        "code_sha256": {p.name: sha256(p) for p in (ROOT / "legal_rag_a").glob("*.py")},
    })
    print(json.dumps({key: summary[key] for key in (
        "n_passages", "n_queries", "n_id_derived_sections", "single_gold_queries",
        "multi_gold_queries", "team_split_counts")}, indent=2))


def run_evaluation(args, config):
    corpus, queries = load_benchmark(args.data_dir)
    split = {row["query_id"]: row["split"] for row in team_split(queries, config)}
    if args.split != "official_test":
        queries = [q for q in queries if split[q["id"]] == args.split]
    rows = read_jsonl(args.predictions)
    predictions = {}
    for row in rows:
        qid = str(row["query_id"])
        if qid in predictions:
            raise ValueError(f"Duplicate prediction query ID: {qid}")
        predictions[qid] = row["passage_ids"]
    details, aggregate = evaluate(queries, predictions, {p["id"] for p in corpus},
                                  config["evaluation_ks"])
    metadata = {"method": args.method, "dataset": config["dataset"], "revision": REVISION,
                "split": args.split, "split_id": "official-test-v1" if args.split == "official_test"
                else config["team_split_id"], "text_policy": args.text_policy,
                "corpus_unit": "original_passage", "corpus_sha256": sha256(args.data_dir / "corpus.jsonl")}
    write_csv(args.out / "per_query_metrics.csv", details)
    write_csv(args.out / "metrics.csv", [{**metadata, **row} for row in aggregate])
    write_json(args.out / "evaluation_manifest.json", {
        **metadata, "predictions_sha256": sha256(args.predictions), "config": config,
        "evaluation_code_sha256": sha256(ROOT / "legal_rag_a/evaluation.py"),
        "latency": "Not measured by an evaluator operating on saved rankings.",
    })
    print(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "profile", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--split", choices=("dev", "test", "official_test"), default="test")
    parser.add_argument("--text-policy", choices=("text_only", "title_text_footnotes"), default="text_only")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.command == "download":
        print(json.dumps(download(args.data_dir), indent=2))
    elif args.command == "profile":
        run_profile(args.data_dir, args.out, config)
    else:
        if args.predictions is None or not args.method:
            parser.error("evaluate requires --predictions and --method")
        run_evaluation(args, config)


if __name__ == "__main__":
    main()
