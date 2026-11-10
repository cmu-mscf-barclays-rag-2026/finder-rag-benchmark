"""Selective one-retry agentic search on MTEB's financial sets, with gpt-4o-mini.

Pipeline (agentic_graph.OneRetryGraphRetriever): search the graph once (the
one-step graph, which beat the walk on development for both sets); if the
search is not confident (peakedness of its top-20 scores below the dataset's
development median), ask the LLM for one reformulated query, search again with
50% question + 50% reformulation seeds, and combine.

Reformulators, fixed before any run:

- ``keywords``: the existing keyword-expansion prompt (question only).
- ``follow_up``: evidence-aware. The LLM sees the question and the first
  search's top 3 passages (first 50 words each) and writes one query for the
  passage that answers it.

Combinations: ``mixed`` (the second search's ranking) or ``rrf``
(reciprocal-rank fusion of both searches).

Protocol, to keep API use small:

- ``--split dev``: all 2 x 2 designs on the development questions of both
  sets (229 FinQA + 30 FinanceBench). The design with the best pooled
  always-retry nDCG@10 is selected and written to dev_selection.json, with
  each set's gate threshold (median development peakedness, label-free).
- ``--split test``: only the selected design, run as deployed: the LLM is
  called only for test questions below the gate threshold, and the gated
  pipeline is compared with the single pass. Always-retry, oracle, and budget
  curves are reported on development only, to halve test-time API use.

LLM responses are cached by exact prompt, each run is capped at
``--max-new-calls`` new calls, and token usage comes from the API's ``usage``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agentic_graph import OneRetryGraphRetriever, peakedness, rrf_fuse
from finance_data import DATASETS, load_finance, metrics, split_questions
from legal_protocol import paired_bootstrap
from llm_graph import EXPANSION_PROMPT, LLMExpandedGraphRetriever
from openai_llm import CachedOpenAIChat, cost_usd
from standalone_graph import StandaloneGraphRetriever


MODEL = "gpt-4o-mini"
DEPTH = 100
GATE_DEPTH = 20
QUERY_WEIGHT = 0.5
PROPAGATE = False  # the one-step graph: better than the walk on development for both sets
EVIDENCE_PASSAGES, EVIDENCE_WORDS = 3, 50
FOLLOW_UP_SYSTEM = (
    "You help a search engine find the passage of a financial report that answers a question. "
    "Given the question and the passages found so far, which may cover the wrong company, year, "
    "or table, write one short search query for the passage that answers it, using the wording a "
    "financial report would use. Output only the query."
)
REFORMULATORS = ("keywords", "follow_up")
COMBINES = ("mixed", "rrf")
MAIN = ("ndcg@10", "mrr@10", "recall@1", "recall@5", "recall@10", "recall@100")
SELECTION_FILE = "dev_selection.json"


def keyword_messages(question: str) -> list[dict[str, str]]:
    roles = {"system": "system", "human": "user"}
    return [{"role": roles[m.type], "content": m.content}
            for m in EXPANSION_PROMPT.format_messages(question=question)]


def follow_up_messages(question: str, passages: list[str]) -> list[dict[str, str]]:
    found = "\n".join(f"[{i}] {' '.join(text.split()[:EVIDENCE_WORDS])}" for i, text in enumerate(passages, 1))
    return [{"role": "system", "content": FOLLOW_UP_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nPassages found so far:\n{found}\n\nSearch query:"}]


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {m: statistics.fmean(r[m] for r in rows) for m in MAIN}


def run_dataset(name: str, split: str, designs: list[tuple[str, str]], llm: CachedOpenAIChat,
                threshold: float | None) -> dict[str, Any]:
    corpus, questions = load_finance(name)
    dev, test = split_questions(questions)
    text = {r["id"]: r["text"] for r in corpus}
    graph = StandaloneGraphRetriever([Document(page_content=r["text"], metadata={"passage_id": r["id"]})
                                      for r in corpus], top_k=DEPTH)
    expanded = LLMExpandedGraphRetriever(graph, lambda q: "", query_weight=QUERY_WEIGHT)
    ids = sorted(dev if split == "dev" else test, key=lambda q: (len(q), q))
    first_ms, second_ms = [], []
    firsts = {}
    for qid in sorted(dev | set(ids), key=lambda q: (len(q), q)):
        started = time.perf_counter()
        firsts[qid] = graph.retrieve(questions[qid].text, propagate=PROPAGATE)
        first_ms.append((time.perf_counter() - started) * 1000)
    if threshold is None:  # label-free: median development peakedness
        threshold = statistics.median(peakedness(firsts[q], GATE_DEPTH) for q in dev)
    rankings: dict[str, dict[str, list[str]]] = {"single_pass": {}}
    info: dict[str, dict[str, Any]] = {}
    for qid in ids:
        first = firsts[qid]
        rankings["single_pass"][qid] = [e.node_id for e in first.explanations]
        info[qid] = {"peakedness": peakedness(first, GATE_DEPTH)}
        if split == "test" and info[qid]["peakedness"] >= threshold:
            continue  # the deployed gate: confident queries make no LLM call
        for reformulator in sorted({r for r, _ in designs}):
            messages = (keyword_messages(questions[qid].text) if reformulator == "keywords"
                        else follow_up_messages(questions[qid].text,
                                                [text[e.node_id] for e in first.explanations[:EVIDENCE_PASSAGES]]))
            output = llm(messages, label=json.dumps([name, qid, reformulator]))
            started = time.perf_counter()
            second = graph.retrieve_from_links(expanded.seed_links(questions[qid].text, output["text"]),
                                               propagate=PROPAGATE, top_k=DEPTH)
            second_ms.append((time.perf_counter() - started) * 1000)
            info[qid][reformulator] = {k: output[k] for k in ("text", "prompt_tokens", "completion_tokens", "seconds")}
            for combine in sorted({c for r, c in designs if r == reformulator}):
                rankings.setdefault(f"always_{reformulator}_{combine}", {})[qid] = (
                    [e.node_id for _, e in rrf_fuse(first, second)][:DEPTH] if combine == "rrf"
                    else [e.node_id for e in second.explanations])
    return {"name": name, "questions": questions, "ids": ids, "rankings": rankings, "info": info,
            "threshold": threshold, "graph": graph, "text": text,
            "first_ms": statistics.fmean(first_ms), "second_ms": statistics.fmean(second_ms) if second_ms else 0.0}


def gated_test_rows(run: dict[str, Any], reformulator: str, combine: str) -> tuple[list[dict], dict]:
    """Test: only the deployed gated pipeline (the LLM is called below the threshold only)."""
    questions, ids, rankings, info = (run[k] for k in ("questions", "ids", "rankings", "info"))
    retried = [q for q in ids if reformulator in info[q]]
    always = rankings.get(f"always_{reformulator}_{combine}", {})
    rankings[f"gated_{reformulator}_{combine}"] = {q: always.get(q, rankings["single_pass"][q]) for q in ids}
    scored = {m: {q: metrics(r[q], questions[q]) for q in ids} for m, r in rankings.items() if not m.startswith("always")}
    tokens = sum(info[q][reformulator]["prompt_tokens"] + info[q][reformulator]["completion_tokens"] for q in retried)
    prompt = sum(info[q][reformulator]["prompt_tokens"] for q in retried)
    completion = sum(info[q][reformulator]["completion_tokens"] for q in retried)
    llm_ms = 1000 * statistics.fmean(info[q][reformulator]["seconds"] for q in retried) if retried else 0.0
    share = len(retried) / len(ids)
    rows = [{"dataset": run["name"], "method": "single_pass", "design": "", "split": "test", "n_queries": len(ids),
             **mean_metrics(list(scored["single_pass"].values())), "retry_share": 0.0, "llm_calls_per_query": 0.0,
             "tokens_per_query": 0.0, "usd_per_1000_queries": 0.0, "expected_ms": run["first_ms"]},
            {"dataset": run["name"], "method": f"gated_{reformulator}_{combine}", "design": f"{reformulator}_{combine}",
             "split": "test", "n_queries": len(ids),
             **mean_metrics(list(scored[f"gated_{reformulator}_{combine}"].values())), "retry_share": share,
             "llm_calls_per_query": share, "tokens_per_query": tokens / len(ids),
             "usd_per_1000_queries": 1000 * (cost_usd(MODEL, prompt, completion) or 0.0) / len(ids),
             "expected_ms": run["first_ms"] + share * (llm_ms + run["second_ms"])}]
    return rows, scored


def method_rows(run: dict[str, Any], designs: list[tuple[str, str]], split: str) -> tuple[list[dict], list[dict], dict]:
    questions, ids, rankings, info, threshold = (run[k] for k in ("questions", "ids", "rankings", "info", "threshold"))
    if split == "test":
        rows, scored = gated_test_rows(run, *designs[0])
        return rows, [], scored
    scored = {m: {q: metrics(r[q], questions[q]) for q in ids} for m, r in rankings.items()}
    rows = [{"dataset": run["name"], "method": "single_pass", "design": "", "split": split, "n_queries": len(ids),
             **mean_metrics(list(scored["single_pass"].values())), "retry_share": 0.0, "llm_calls_per_query": 0.0,
             "tokens_per_query": 0.0, "usd_per_1000_queries": 0.0, "expected_ms": run["first_ms"]}]
    curves = []
    for reformulator, combine in designs:
        always = f"always_{reformulator}_{combine}"
        helps = {q: scored[always][q]["ndcg@10"] > scored["single_pass"][q]["ndcg@10"] for q in ids}
        retried = {"gated": {q for q in ids if info[q]["peakedness"] < threshold}, "oracle": {q for q in ids if helps[q]}}
        for kind, chosen in retried.items():
            name = f"{kind}_{reformulator}_{combine}"
            rankings[name] = {q: rankings[always][q] if q in chosen else rankings["single_pass"][q] for q in ids}
            scored[name] = {q: metrics(rankings[name][q], questions[q]) for q in ids}
        prompt = statistics.fmean(info[q][reformulator]["prompt_tokens"] for q in ids)
        completion = statistics.fmean(info[q][reformulator]["completion_tokens"] for q in ids)
        llm_ms = 1000 * statistics.fmean(info[q][reformulator]["seconds"] for q in ids)
        for method, chosen in ((always, set(ids)), (f"gated_{reformulator}_{combine}", retried["gated"]),
                               (f"oracle_{reformulator}_{combine}", retried["oracle"])):
            share = len(chosen) / len(ids)
            rows.append({"dataset": run["name"], "method": method, "design": f"{reformulator}_{combine}",
                         "split": split, "n_queries": len(ids), **mean_metrics(list(scored[method].values())),
                         "retry_share": share, "llm_calls_per_query": share,
                         "tokens_per_query": share * (prompt + completion),
                         "usd_per_1000_queries": 1000 * share * (cost_usd(MODEL, prompt, completion) or 0.0),
                         "expected_ms": run["first_ms"] + share * (llm_ms + run["second_ms"])})
        orders = {"peakedness": sorted(ids, key=lambda q: (info[q]["peakedness"], q)),
                  "oracle": sorted(ids, key=lambda q: (not helps[q], q))}
        for step in range(11):
            k = round(step / 10 * len(ids))
            for gate, order in orders.items():
                chosen = set(order[:k])
                curves.append({"dataset": run["name"], "design": f"{reformulator}_{combine}", "gate": gate,
                               "retry_share": k / len(ids), **mean_metrics(
                                   [scored[always][q] if q in chosen else scored["single_pass"][q] for q in ids])})
            draws = [set(random.Random(seed).sample(ids, k)) for seed in range(500)]
            values = [mean_metrics([scored[always][q] if q in chosen else scored["single_pass"][q] for q in ids])
                      for chosen in draws]
            curves.append({"dataset": run["name"], "design": f"{reformulator}_{combine}", "gate": "random",
                           "retry_share": k / len(ids), **{m: statistics.fmean(v[m] for v in values) for m in MAIN}})
    return rows, curves, scored


def evaluate(split: str, output_dir: Path, max_new_calls: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = CachedOpenAIChat(MODEL, output_dir / "llm_cache.jsonl", max_new_calls=max_new_calls)
    selection = json.loads((output_dir / SELECTION_FILE).read_text()) if split == "test" else None
    designs = ([(r, c) for r in REFORMULATORS for c in COMBINES] if split == "dev"
               else [(selection["reformulator"], selection["combine"])])
    all_rows, all_curves, intervals, traces, pooled = [], [], [], [], {}
    for name in DATASETS:
        run = run_dataset(name, split, designs, llm, selection["thresholds"][name] if selection else None)
        rows, curves, scored = method_rows(run, designs, split)
        all_rows += rows
        all_curves += curves
        pooled[name] = (run, scored)
        for qid in run["ids"]:
            traces.append({"dataset": name, "query_id": qid, "question": run["questions"][qid].text,
                           "relevant": sorted(run["questions"][qid].relevant), **run["info"][qid],
                           "single_top5": run["rankings"]["single_pass"][qid][:5],
                           **{f"{m}_top5": r[qid][:5] for m, r in run["rankings"].items()
                              if m.startswith("always") and qid in r}})
        if split == "test":
            reformulator, combine = designs[0]
            questions, ids = run["questions"], run["ids"]
            method = f"gated_{reformulator}_{combine}"
            for metric in ("ndcg@10", "recall@1"):
                deltas = [scored[method][q][metric] - scored["single_pass"][q][metric] for q in ids]
                low, high = paired_bootstrap(deltas, [questions[q].group for q in ids])
                intervals.append({"dataset": name, "comparison": f"{method} minus single_pass", "metric": metric,
                                  "difference": statistics.fmean(deltas), "wins": sum(d > 0 for d in deltas),
                                  "losses": sum(d < 0 for d in deltas),
                                  "bootstrap_95pct_low": low, "bootstrap_95pct_high": high})
            # The prototype class must reproduce the gated ranking.
            text = run["text"]
            reformulate = ((lambda q, first: llm(follow_up_messages(
                q, [text[e.node_id] for e in first.explanations[:EVIDENCE_PASSAGES]]), label="check")["text"])
                if reformulator == "follow_up" else (lambda q: llm(keyword_messages(q), label="check")["text"]))
            agent = OneRetryGraphRetriever(run["graph"], reformulate, threshold=run["threshold"],
                                           query_weight=QUERY_WEIGHT, depth=GATE_DEPTH, combine=combine,
                                           with_evidence=reformulator == "follow_up", propagate=PROPAGATE)
            for qid in ids[:50]:
                got = [e.node_id for e in agent.retrieve(questions[qid].text, top_k=DEPTH).explanations]
                if got != run["rankings"][f"gated_{reformulator}_{combine}"][qid]:
                    raise AssertionError(f"OneRetryGraphRetriever differs on {name} {qid}")
    for filename, data in ((f"{split}_methods.csv", all_rows), (f"{split}_budget_curves.csv", all_curves),
                           (f"{split}_paired_intervals.csv", intervals)):
        if not data:
            continue
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (output_dir / f"{split}_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"split": split, "model": MODEL, "designs": [f"{r}_{c}" for r, c in designs],
               "first_search": "one-step graph" if not PROPAGATE else "graph walk",
               "thresholds": {name: run["threshold"] for name, (run, _) in pooled.items()},
               "new_llm_calls_this_run": llm.new_calls, "new_usage_this_run": llm.usage,
               "new_cost_usd_this_run": cost_usd(MODEL, llm.usage["prompt_tokens"], llm.usage["completion_tokens"])}
    if split == "dev":
        # Pooled over both sets' development questions, weighted by question.
        def pooled_ndcg(method: str) -> float:
            values = [s[method][q]["ndcg@10"] for run, s in pooled.values() for q in run["ids"]]
            return statistics.fmean(values)
        scores = {(r, c): pooled_ndcg(f"always_{r}_{c}") for r, c in designs}
        best = max(scores, key=lambda d: (round(scores[d], 6), d))
        selection = {"reformulator": best[0], "combine": best[1],
                     "pooled_dev_always_ndcg@10": {f"{r}_{c}": v for (r, c), v in scores.items()},
                     "pooled_dev_single_pass_ndcg@10": pooled_ndcg("single_pass"),
                     "thresholds": summary["thresholds"],
                     "rule": "highest pooled development always-retry nDCG@10; thresholds = median dev peakedness"}
        (output_dir / SELECTION_FILE).write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        summary["selected_design"] = f"{best[0]}_{best[1]}"
    (output_dir / f"{split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in all_rows:
        print(f"{row['dataset']:12s} {row['method']:28s} nDCG@10={row['ndcg@10']:.4f} R@1={row['recall@1']:.4f} "
              f"retried={row['retry_share']:.2f} tokens/q={row['tokens_per_query']:.0f} ~{row['expected_ms']:.0f} ms")
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("new_", "selected", "thresholds"))}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--max-new-calls", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("results/finance_agentic"))
    args = parser.parse_args()
    evaluate(args.split, args.output_dir, args.max_new_calls)


if __name__ == "__main__":
    main()
