"""Export reviewable metrics, rankings, and sparse-vs-dense error examples."""

from __future__ import annotations

import csv
from pathlib import Path

from .bm25 import tokenize
from .data import write_json
from .evaluation import evaluate_rag


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value) -> str:
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def table(summaries: list[dict]) -> str:
    columns = [("method", "Method"), ("top_k", "k"), ("query_count", "Queries"),
               ("precision", "P@k"), ("recall", "R@k"), ("mrr", "MRR@k"), ("ndcg", "nDCG@k"),
               ("evidence_coverage", "Evidence coverage@k"), ("answer_accuracy_exact_match", "Answer EM"),
               ("answer_correctness", "Answer correctness"), ("latency_ms", "Mean search ms")]
    return "\n".join([
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(_format(summary.get(key)) for key, _ in columns) + " |" for summary in summaries),
    ])


def save_evaluation(bundle: dict, run: dict, evaluation: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", run)
    write_json(output / "summary.json", evaluation["summary"])
    write_csv(output / "summary.csv", [evaluation["summary"]])
    write_csv(output / "per_query.csv", evaluation["per_query"])
    write_csv(output / "by_category.csv", evaluation["by_category"])
    documents = {doc["doc_id"]: doc for doc in bundle["corpus"]}
    queries = {query["query_id"]: query for query in bundle["queries"]}
    results = {row["query_id"]: row for row in run["results"]}
    summary = evaluation["summary"]
    lines = [f"# {run['method']} — FinDER Person 2 report", "", table([summary]), "",
             f"Corpus: **{summary['corpus_scope']}**; documents: {len(documents)}; partition: {run['partition']}.", "",
             f"Data fingerprint: `{summary['bundle_fingerprint']}`.", "",
             "MRR is truncated at k. Unreturned ranks count as irrelevant for Precision@k. "
             "nDCG uses the complete qrels to build its ideal ranking. Scores are macro-averaged over queries.", "",
             "Answer EM is conservative normalized exact match; it is not semantic correctness. "
             f"Answer scoring: {summary['answer_status']} "
             f"({summary['answer_scored_count']}/{summary['answer_eligible_count']} eligible queries). "
             "N/A means unmeasured, never zero.", "",
             "Latency measures retriever.search only, including query tokenization and ranking. "
             "Index construction, downloads and answer generation are excluded. "
             "Each query is timed once; use repeated, controlled measurements for firm latency conclusions.", "",
             "## Interpretation limits", "",
             "This corpus pools the annotated references from every FinDER row. It omits the full "
             "10-K distractor corpus, so these are reference-pool results, not a reproduction of the paper. "
             "Unjudged documents count as nonrelevant; additional valid evidence can therefore be penalized.", "",
             "Whole-reference mode uses exact evidence IDs. In chunk mode, every child of a gold reference "
             "inherits relevance. That is a source-membership proxy, not proof that each child answers the query. "
             "Evidence coverage separately measures the union of covered normalized reference characters.", "",
             "## Queries to inspect", "",
             "Examples below are ranked by increasing nDCG, then query ID. Token overlap is diagnostic; "
             "it is not used as a relevance label.", ""]
    for row in sorted(evaluation["per_query"], key=lambda r: (r["ndcg"], r["query_id"]))[:8]:
        query = queries[row["query_id"]]
        ranking = results[row["query_id"]]["doc_ids"]
        terms = set(tokenize(query["text"]))
        gold_ids = sorted(bundle["qrels"][query["query_id"]])
        gold_terms = set(term for doc_id in gold_ids for term in tokenize(documents[doc_id]["text"]))
        missing = sorted(terms - gold_terms)
        top_text = documents[ranking[0]]["text"][:450] if ranking else "No lexical matches."
        gold_text = documents[gold_ids[0]]["text"][:450]
        lines.extend([f"### {query['query_id']} — {query['category']}", "", query["text"], "",
                      f"First relevant rank: {_format(row['first_relevant_rank'])}; nDCG@{run['top_k']}: {row['ndcg']:.4f}.", "",
                      f"Query tokens absent from all gold passages: {', '.join(missing) or '(none)' }.", "",
                      f"**Top retrieved excerpt:** {top_text}", "", f"**Gold excerpt:** {gold_text}", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def compare_runs(bundle: dict, runs: list[dict], output: Path) -> None:
    if len(runs) < 2:
        raise ValueError("Comparison requires at least two runs")
    evaluations = [evaluate_rag(bundle, run) for run in runs]
    summaries = [evaluation["summary"] for evaluation in evaluations]
    for key in ("bundle_fingerprint", "query_set_fingerprint", "top_k", "partition", "evaluator_version"):
        if len({summary[key] for summary in summaries}) != 1:
            raise ValueError(f"Cannot compare runs with different {key}")
    if len({run["method"] for run in runs}) != len(runs):
        raise ValueError("Give each compared run a distinct method name")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", summaries)
    lines = ["# Shared-method comparison", "", table(summaries), "",
             "All retrieval scores above were recomputed with the same evaluate_rag function, "
             "prepared corpus, qrels, query set and cutoff.", ""]
    if len({summary["answer_query_set_fingerprint"] for summary in summaries}) != 1:
        lines.extend(["Answer coverage differs: answer metrics are not directly comparable until "
                      "the same eligible queries have predictions for every method.", ""])
    if any(summary["answer_status"] != "complete" for summary in summaries):
        lines.extend(["The final answer-quality comparison remains incomplete; supply generated answers "
                      "and the same correctness scorer for every method.", ""])
    if any(run.get("environment") != runs[0].get("environment") or
           run.get("latency_scope") != runs[0].get("latency_scope") or
           run.get("timing_protocol") != runs[0].get("timing_protocol") for run in runs[1:]):
        lines.extend(["Latency environments/scopes differ. Rerun with the same hardware and timing protocol "
                      "before interpreting latency differences.", ""])
    first = {row["query_id"]: row for row in evaluations[0]["per_query"]}
    for index, evaluation in enumerate(evaluations[1:], 1):
        deltas = []
        for row in evaluation["per_query"]:
            base = first[row["query_id"]]
            deltas.append({"query_id": row["query_id"], "query": row["query"], "category": row["category"],
                           "base_method": runs[0]["method"], "other_method": runs[index]["method"],
                           "base_rr": base["reciprocal_rank"], "other_rr": row["reciprocal_rank"],
                           "delta_rr": row["reciprocal_rank"] - base["reciprocal_rank"],
                           "base_ndcg": base["ndcg"], "other_ndcg": row["ndcg"],
                           "delta_ndcg": row["ndcg"] - base["ndcg"]})
        deltas.sort(key=lambda row: (-abs(row["delta_ndcg"]), row["query_id"]))
        write_csv(output / f"query_deltas_{index}.csv", deltas)
        wins = sum(row["delta_ndcg"] > 1e-12 for row in deltas)
        losses = sum(row["delta_ndcg"] < -1e-12 for row in deltas)
        lines.extend([f"## {runs[index]['method']} versus {runs[0]['method']}", "",
                      f"Per-query nDCG: {wins} wins, {losses} losses, {len(deltas)-wins-losses} ties.", "",
                      f"Inspect `query_deltas_{index}.csv` alongside each method's `run.json` to check "
                      "ticker/acronym mismatches, exact numbers, paraphrases and generic disclosure distractors.", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
