"""Evaluate selective graph routing and one-retry agentic search.

Team Legal RAG protocol (legal_protocol.py), improved graph, cached LLM
outputs from results/legal_llm_graph (no Ollama needed).

Routing headroom: a perfect per-question router is computed for three
choices: graph propagation on/off, the single pass vs. the LLM-reformulated
retry, and (held-out only) the graph vs. legal_rag BM25.

One-retry search: the first graph search is kept when its confidence clears
a threshold; otherwise the question is reformulated by the LLM keyword
expansion and searched once more (question keeps 50% of the seed mass).
Fixed before the held-out run:

- Gate signals: ``peakedness`` (top PageRank score over the top-20 mean; the
  prototype's gate, chosen on development), ``coverage`` (question link mass
  the top passage mentions), and ``llm_judge`` (CRAG-style: highest LLM
  P(Yes) among the top 3; costs 3 LLM calls per query).
- Threshold: the median signal value over the 20 development questions
  (label-free), aiming at about half the queries retried.
- ``oracle_gate`` retries exactly when that helps; it uses labels and is an
  upper bound only.

``budget_curves.csv`` sweeps the retried share (least confident first) with
random gating as the chance baseline. Costs combine the measured live
latencies in results/legal_llm_graph/test_metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from legal_data import (
    LEGAL_DATASET_ID, legal_records_to_documents, legal_records_to_questions,
    load_legal_corpus, load_legal_questions,
)
from legal_protocol import KS, SPLIT_ID, load_reference_hits, split_queries, write_csv, write_results
from agentic_graph import OneRetryGraphRetriever, coverage, peakedness
from evaluate_llm_graph import LLMCache
from llm_graph import (
    EXPANSION_PROMPT_VERSION, RERANK_PROMPT_VERSION, LLMExpandedGraphRetriever, rerank_messages,
)
from standalone_graph import StandaloneGraphRetriever


QUERY_WEIGHT = 0.5
DEPTH = max(KS)  # 20: ranking depth and the window peakedness is read from
JUDGE_TOP = 3
GATES = ("peakedness", "coverage", "llm_judge")
BUDGETS = [step / 10 for step in range(11)]


def metrics_at_5(ranking: list[str], gold: str) -> tuple[float, float, float]:
    rank = next((i for i, value in enumerate(ranking[:5], 1) if value == gold), None)
    return (float(rank is not None), 1 / rank if rank else 0.0,
            1 / math.log2(rank + 1) if rank else 0.0)


def measured_costs(path: Path) -> dict[str, float]:
    """Per-step live latencies (ms) from the LLM-graph evaluation."""
    ms = {row["method"]: float(row["latency_mean_ms"])
          for row in csv.DictReader(path.open()) if row["k"] == "5" and row["latency_mean_ms"]}
    walk = ms["standalone_graph"]
    return {"graph_walk_ms": walk,
            "expansion_ms": ms["llm_keywords_mixed"] - walk,
            "judge_call_ms": (ms["llm_rerank_graph"] - walk) / DEPTH}


def evaluate(corpus: list[dict], qa: list[dict], output_dir: Path, llm_dir: Path, model: str,
             reference_per_query: Path | None = None) -> dict[str, Any]:
    dev_ids, test_ids = split_queries(qa, corpus)
    documents = legal_records_to_documents(corpus, text_only=True)
    by_id = {str(d.metadata["passage_id"]): d for d in documents}
    questions = legal_records_to_questions(qa)
    graph = StandaloneGraphRetriever(documents, top_k=DEPTH)
    cache = LLMCache(llm_dir / "llm_cache.jsonl", model)

    def uncached() -> Any:
        raise RuntimeError("LLM output missing from cache; rerun evaluate_llm_graph.py first")

    def expand(question: str) -> str:
        return cache.get(f"expand:{EXPANSION_PROMPT_VERSION}", question, uncached)

    def judge(question: str, passage_id: str) -> float:
        text = json.dumps(rerank_messages(question, by_id[passage_id]), ensure_ascii=False)
        return cache.get(f"rerank:{RERANK_PROMPT_VERSION}", text, uncached)

    expanded = LLMExpandedGraphRetriever(graph, expand, query_weight=QUERY_WEIGHT)
    rows: dict[str, dict[str, Any]] = {}
    for question in questions:
        first = graph.retrieve(question.question)
        one_step = graph.retrieve(question.question, propagate=False)
        retry = expanded.retrieve(question.question, top_k=DEPTH)
        ids = [item.node_id for item in first.explanations]
        rows[question.question_id] = {
            "split": "dev" if question.question_id in dev_ids else "test",
            "gold": question.relevant_passage_id,
            "first": ids, "retry": [item.node_id for item in retry.explanations],
            "one_step": [item.node_id for item in one_step.explanations],
            "peakedness": peakedness(first), "coverage": coverage(graph, first),
            "llm_judge": max(judge(question.question, pid) for pid in ids[:JUDGE_TOP]),
            "reformulation": expand(question.question),
        }
    thresholds = {gate: statistics.median(rows[q][gate] for q in dev_ids) for gate in GATES}

    def gated(gate: str) -> dict[str, list[str]]:
        return {q: row["retry"] if row[gate] < thresholds[gate] else row["first"]
                for q, row in rows.items()}

    def helps(row: dict[str, Any]) -> bool:
        return metrics_at_5(row["retry"], row["gold"]) > metrics_at_5(row["first"], row["gold"])

    rankings = {
        "single_pass": {q: row["first"] for q, row in rows.items()},
        "always_retry": {q: row["retry"] for q, row in rows.items()},
        **{f"gated_{gate}": gated(gate) for gate in GATES},
        "oracle_gate": {q: row["retry"] if helps(row) else row["first"] for q, row in rows.items()},
    }
    # The prototype class must reproduce the peakedness-gated ranking exactly.
    agent = OneRetryGraphRetriever(graph, expand, threshold=thresholds["peakedness"],
                                   query_weight=QUERY_WEIGHT, depth=DEPTH)
    for question in questions:
        result = agent.retrieve(question.question, top_k=DEPTH)
        if [item.node_id for item in result.explanations] != rankings["gated_peakedness"][question.question_id]:
            raise AssertionError(f"OneRetryGraphRetriever differs on question {question.question_id}")

    costs = measured_costs(llm_dir / "test_metrics.csv")

    def cost(gate: str | None, retry_rate: float) -> tuple[float, float]:
        """Expected ms and LLM calls per query."""
        judge_calls = JUDGE_TOP if gate == "llm_judge" else 0
        ms = (costs["graph_walk_ms"] + judge_calls * costs["judge_call_ms"]
              + retry_rate * (costs["expansion_ms"] + costs["graph_walk_ms"]))
        return ms, judge_calls + retry_rate

    curves = []
    for split, ids in (("dev", sorted(dev_ids, key=int)), ("test", sorted(test_ids, key=int))):
        n = len(ids)
        for signal in (*GATES, "random", "oracle"):
            for budget in BUDGETS:
                k = round(budget * n)
                if signal == "random":
                    draws = [set(random.Random(seed).sample(ids, k)) for seed in range(1000)]
                elif signal == "oracle":
                    order = sorted(ids, key=lambda q: (not helps(rows[q]), int(q)))
                    draws = [set(order[:k])]
                else:
                    draws = [set(sorted(ids, key=lambda q: (rows[q][signal], int(q)))[:k])]
                values = [[metrics_at_5(rows[q]["retry"] if q in chosen else rows[q]["first"],
                                        rows[q]["gold"]) for q in ids] for chosen in draws]
                ms, calls = cost(signal if signal in GATES else None, k / n)
                curves.append({
                    "split": split, "gate": signal, "retry_share": k / n,
                    **{name: statistics.fmean(statistics.fmean(v[i] for v in run) for run in values)
                       for i, name in enumerate(("hit5", "mrr5", "ndcg5"))},
                    "expected_ms": ms, "llm_calls_per_query": calls,
                })

    def hit5(ranking: list[str], gold: str) -> bool:
        return gold in ranking[:5]

    references = load_reference_hits(reference_per_query) if reference_per_query else {}
    headroom = {}
    for split, ids in (("dev", dev_ids), ("test", test_ids)):
        mean = lambda values: statistics.fmean(values)  # noqa: E731
        headroom[split] = {
            "propagation_on_off": {
                "one_step": mean(hit5(rows[q]["one_step"], rows[q]["gold"]) for q in ids),
                "graph_walk": mean(hit5(rows[q]["first"], rows[q]["gold"]) for q in ids),
                "oracle": mean(hit5(rows[q]["one_step"], rows[q]["gold"]) or hit5(rows[q]["first"], rows[q]["gold"]) for q in ids)},
            "single_pass_vs_retry": {
                "single_pass": mean(hit5(rows[q]["first"], rows[q]["gold"]) for q in ids),
                "always_retry": mean(hit5(rows[q]["retry"], rows[q]["gold"]) for q in ids),
                "oracle": mean(hit5(rows[q]["first"], rows[q]["gold"]) or hit5(rows[q]["retry"], rows[q]["gold"]) for q in ids)},
        }
        if split == "test" and "legal_rag_bm25" in references:
            bm25 = references["legal_rag_bm25"]
            headroom[split]["graph_vs_bm25"] = {
                "graph_walk": headroom[split]["propagation_on_off"]["graph_walk"],
                "bm25": mean(bm25[q] for q in ids),
                "oracle": mean(hit5(rows[q]["first"], rows[q]["gold"]) or bm25[q] > 0 for q in ids)}

    results = write_results(
        output_dir, questions=questions, dev_ids=dev_ids, test_ids=test_ids, rankings=rankings,
        config_ids={"single_pass": "graph_improved", "always_retry": "graph_llmkw_q0.5",
                    **{f"gated_{g}": f"retry_if_{g}_below_dev_median" for g in GATES},
                    "oracle_gate": "oracle_upper_bound"},
        latency={}, baseline=("single_pass", "always_retry"), references=references,
    )
    write_csv(output_dir / "budget_curves.csv", curves)
    gate_rows = []
    for gate in (*GATES, "oracle"):
        for split, ids in (("dev", dev_ids), ("test", test_ids)):
            method = f"gated_{gate}" if gate in GATES else "oracle_gate"
            rate = statistics.fmean(rankings[method][q] is rows[q]["retry"] for q in ids)
            ms, calls = cost(gate if gate in GATES else None, rate)
            gate_rows.append({"method": method, "split": split, "threshold": thresholds.get(gate, ""),
                              "retry_share": rate, "expected_ms": ms, "llm_calls_per_query": calls})
    for name, rate in (("single_pass", 0.0), ("always_retry", 1.0)):
        ms, calls = cost(None, rate)
        gate_rows += [{"method": name, "split": split, "threshold": "", "retry_share": rate,
                       "expected_ms": ms, "llm_calls_per_query": calls} for split in ("dev", "test")]
    write_csv(output_dir / "gate_costs.csv", gate_rows)
    summary = {
        "dataset_id": LEGAL_DATASET_ID, "corpus_passages": len(documents),
        "questions": len(questions), "dev_count": len(dev_ids), "test_count": len(test_ids),
        "split": f"{SPLIT_ID} (legal_rag, feature/person2-bm25)",
        "graph_settings": graph.settings.__dict__, "query_weight": QUERY_WEIGHT,
        "thresholds_dev_median": thresholds, "costs_ms": costs,
        "cost_note": ("Expected latency = graph walk + judge calls (llm_judge gate) + retry share x "
                      "(LLM expansion + second walk), from live latencies measured in "
                      "results/legal_llm_graph; not re-timed end to end."),
        "routing_headroom_hit5": headroom,
        "protocol": ("Team legal_rag protocol. Gate signals, threshold rule, and combination were fixed "
                     "before the held-out run; peakedness was chosen on development by how well it "
                     "predicts first-pass misses (AUC 0.65 vs 0.51 for coverage). End to end, no gate "
                     "clearly beats the single pass on development, where a retry changes Hit@5 for only "
                     "3 of 20 questions (the top-5 rank for 7), so development cannot validate a gate. The LLM expansion was designed on "
                     "an earlier split overlapping held-out, so retry numbers are somewhat optimistic."),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for q, row in sorted(rows.items(), key=lambda item: int(item[0])):
            handle.write(json.dumps({
                "question_id": q, "split": row["split"], "gold_passage_id": row["gold"],
                **{gate: row[gate] for gate in GATES}, "reformulation": row["reformulation"],
                "first_top5": row["first"][:5], "retry_top5": row["retry"][:5],
                "retry_helps": helps(row),
            }, ensure_ascii=False) + "\n")
    print("thresholds:", {k: round(v, 4) for k, v in thresholds.items()})
    print("headroom:", json.dumps(headroom))
    for row in gate_rows:
        if row["split"] == "test":
            print(f"{row['method']:20s} retried {row['retry_share']:.2f}  ~{row['expected_ms']:.0f} ms  "
                  f"{row['llm_calls_per_query']:.2f} LLM calls/query")
    return {**summary, **results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/legal_agentic_graph"))
    parser.add_argument("--llm-dir", type=Path, default=Path("results/legal_llm_graph"),
                        help="LLM-graph results: llm_cache.jsonl and live latencies")
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--reference-per-query", type=Path,
                        help="legal_rag results/initial/per_query.csv for paired BM25/dense/hybrid comparisons")
    args = parser.parse_args()
    evaluate(load_legal_corpus(), load_legal_questions(), args.output_dir, args.llm_dir,
             args.model, args.reference_per_query)


if __name__ == "__main__":
    main()
