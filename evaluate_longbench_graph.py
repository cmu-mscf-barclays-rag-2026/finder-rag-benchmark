"""Run the standalone graph RAG on LongBench v2 (end to end).

LongBench v2 (``zai-org/LongBench-v2``, 503 multiple-choice questions over
8k-1.4M-word contexts) has no relevance labels, so retrieval is judged by
answer accuracy. For each question the context is cut into 300-word chunks
(original text kept, so markdown headings reach the graph's heading tree),
the chunks of one context form one graph document, and a fixed word budget
of evidence goes to the local LLM with the official 0-shot prompt.

Context methods, all with the same budget (8 chunks = 2,400 words):

- ``graph``: the improved graph's top 8 chunks for question + choices,
  shown in document order.
- ``bm25``: the team's BM25 (team_bm25.py, k1 = 0.9, b = 0.4), same query.
- ``truncate_middle``: the first and last 1,200 words (the official
  LongBench v2 way of fitting a context into a window).
- ``no_context``: question and choices only.

Answers use the official extraction ("The correct answer is (X)"); an
unparseable answer counts as wrong. Accuracy is reported overall and by
difficulty, length, and domain, with paired bootstrap intervals (10,000
resamples, seed 42). LLM outputs are cached by exact prompt, so reruns and
interrupted runs need Ollama only for new prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from datasets import load_dataset
from langchain_core.documents import Document

from evaluate_llm_graph import LLMCache
from standalone_graph import StandaloneGraphRetriever
from team_bm25 import BM25Retriever


CHUNK_WORDS = 300
TOP_CHUNKS = 8
BUDGET_WORDS = CHUNK_WORDS * TOP_CHUNKS
NUM_CTX = 16_384
METHODS = ("graph", "bm25", "truncate_middle", "no_context")
PROMPT = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}

Format your response as follows: "The correct answer is (insert answer here)"."""
NO_CONTEXT_PROMPT = """What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}

Format your response as follows: "The correct answer is (insert answer here)"."""
_WORD = re.compile(r"\S+")


def chunk(text: str, words: int = CHUNK_WORDS) -> list[str]:
    """Cut at every ``words``-th word, keeping the original whitespace and markup."""
    starts = [match.start() for match in _WORD.finditer(text)][::words]
    return [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)])]


def truncate_middle(text: str, words: int = BUDGET_WORDS) -> str:
    spans = [match.span() for match in _WORD.finditer(text)]
    if len(spans) <= words:
        return text
    half = words // 2
    return text[:spans[half - 1][1]] + "\n\n[...]\n\n" + text[spans[-half][0]:]


def extract_answer(response: str) -> str | None:
    """LongBench v2's official answer extraction."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    match = match or re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def ask_ollama(prompt: str, model: str, base_url: str) -> dict[str, Any]:
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
        "options": {"temperature": 0, "num_predict": 128, "num_ctx": NUM_CTX},
    }).encode("utf-8")
    request = Request(f"{base_url.rstrip('/')}/api/chat", data=body,
                      headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=600) as response:
        payload = json.load(response)
    return {"response": payload["message"]["content"],
            "prompt_tokens": payload.get("prompt_eval_count"),
            "seconds": time.perf_counter() - started}


def contexts(sample: dict) -> tuple[dict[str, str], dict[str, Any]]:
    """Evidence text per method, plus retrieval traces and timings."""
    sid, text = sample["_id"], sample["context"]
    query = "\n".join([sample["question"]] + [sample[f"choice_{c}"] for c in "ABCD"])
    chunks = chunk(text)
    documents = [Document(page_content=piece, metadata={
        "chunk_id": f"{sid}:chunk{i}", "row_id": sid, "reference_number": 1,
        "reference_chunk_number": i}) for i, piece in enumerate(chunks, start=1)]
    order = {document.metadata["chunk_id"]: i for i, document in enumerate(documents)}
    started = time.perf_counter()
    graph = StandaloneGraphRetriever(documents, top_k=TOP_CHUNKS)
    result = graph.retrieve(query)
    graph_seconds = time.perf_counter() - started
    graph_ids = [item.node_id for item in result.explanations]
    started = time.perf_counter()
    bm25 = BM25Retriever([{"doc_id": d.metadata["chunk_id"], "text": d.page_content} for d in documents],
                         k1=0.9, b=0.4)
    bm25_ids = [hit["doc_id"] for hit in bm25.search(query, top_k=TOP_CHUNKS)]
    bm25_seconds = time.perf_counter() - started

    def evidence(ids: list[str]) -> str:
        return "\n\n[...]\n\n".join(chunks[order[i]] for i in sorted(ids, key=order.get))

    texts = {"graph": evidence(graph_ids), "bm25": evidence(bm25_ids),
             "truncate_middle": truncate_middle(text), "no_context": ""}
    trace = {
        "chunks": len(chunks), "graph_chunks": graph_ids, "bm25_chunks": bm25_ids,
        "overlap_graph_bm25": len(set(graph_ids) & set(bm25_ids)),
        "graph_seconds": graph_seconds, "bm25_seconds": bm25_seconds,
        "graph_stats": {k: graph.stats[k] for k in ("concept_nodes", "heading_nodes", "incidence_edges")},
        "graph_top3": [{"node_id": e.node_id, "score": e.final_score,
                        "path": list(e.path_labels), "relations": list(e.path_relations)}
                       for e in result.explanations[:3]],
    }
    return texts, trace


def bootstrap(deltas: list[float], seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(10_000))
    return means[249], means[9749]


def evaluate(output_dir: Path, model: str, base_url: str, limit: int | None = None) -> dict:
    samples = load_dataset("zai-org/LongBench-v2", split="train").to_list()
    if limit:
        samples = samples[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = LLMCache(output_dir / "llm_cache.jsonl", model)
    per_question, traces = [], []
    for number, sample in enumerate(samples, start=1):
        texts, trace = contexts(sample)
        row = {"id": sample["_id"], "domain": sample["domain"], "difficulty": sample["difficulty"],
               "length": sample["length"], "answer": sample["answer"]}
        fields = {"question": sample["question"].strip(),
                  **{c: sample[f"choice_{c}"].strip() for c in "ABCD"}}
        for method in METHODS:
            prompt = (NO_CONTEXT_PROMPT.format(**fields) if method == "no_context"
                      else PROMPT.format(context=texts[method].strip(), **fields))
            output = cache.get("longbench-answer", prompt,
                               lambda prompt=prompt: ask_ollama(prompt, model, base_url),
                               label=json.dumps([sample["_id"], method]))
            prediction = extract_answer(output["response"])
            row[f"{method}_pred"] = prediction
            row[f"{method}_correct"] = float(prediction == sample["answer"])
            row[f"{method}_prompt_tokens"] = output["prompt_tokens"]
        per_question.append(row)
        traces.append({"id": sample["_id"], **trace})
        if number % 25 == 0:
            done = {m: statistics.fmean(r[f"{m}_correct"] for r in per_question) for m in METHODS}
            print(f"{number}/{len(samples)} " + " ".join(f"{m}={v:.3f}" for m, v in done.items()),
                  flush=True)

    groups = [("all", lambda r: True)]
    for key in ("difficulty", "length", "domain"):
        for value in sorted({r[key] for r in per_question}):
            groups.append((f"{key}={value}", lambda r, key=key, value=value: r[key] == value))
    accuracy = []
    for name, keep in groups:
        rows = [r for r in per_question if keep(r)]
        accuracy.append({"group": name, "n": len(rows),
                         **{m: statistics.fmean(r[f"{m}_correct"] for r in rows) for m in METHODS},
                         **{f"{m}_unparsed": statistics.fmean(r[f"{m}_pred"] is None for r in rows)
                            for m in METHODS}})
    paired = []
    for other in ("bm25", "truncate_middle", "no_context"):
        deltas = [r["graph_correct"] - r[f"{other}_correct"] for r in per_question]
        low, high = bootstrap(deltas)
        paired.append({"comparison": f"graph minus {other}", "n": len(deltas),
                       "accuracy_difference": statistics.fmean(deltas),
                       "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                       "bootstrap_95pct_low": low, "bootstrap_95pct_high": high})
    for filename, rows in (("per_question.csv", per_question), ("accuracy.csv", accuracy),
                           ("paired_accuracy_intervals.csv", paired)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    answers = [r["answer"] for r in per_question]
    summary = {
        "dataset": "LongBench v2 (zai-org/LongBench-v2, train split = all 503 questions)",
        "questions": len(per_question), "llm_model": model, "num_ctx": NUM_CTX,
        "temperature": 0, "chunk_words": CHUNK_WORDS, "top_chunks": TOP_CHUNKS,
        "budget_words": BUDGET_WORDS, "retrieval_query": "question + four choices",
        "chance": {"uniform": 0.25,
                   "majority_letter": max(answers.count(c) for c in "ABCD") / len(answers)},
        "max_prompt_tokens": {m: max(r[f"{m}_prompt_tokens"] or 0 for r in per_question)
                              for m in METHODS},
        "mean_retrieval_seconds": {
            "graph": statistics.fmean(t["graph_seconds"] for t in traces),
            "bm25": statistics.fmean(t["bm25_seconds"] for t in traces)},
        "mean_overlap_graph_bm25_chunks": statistics.fmean(t["overlap_graph_bm25"] for t in traces),
        "new_llm_calls_this_run": cache.new_calls,
        "protocol": ("Fixed configurations, nothing tuned on LongBench v2. Official 0-shot prompt and "
                     "answer extraction; temperature 0 instead of the official 0.1 for determinism."),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in accuracy[:1]:
        print(" ".join(f"{m}={row[m]:.4f}" for m in METHODS))
    for row in paired:
        print(f"{row['comparison']:32s} {row['accuracy_difference']:+.4f} "
              f"[{row['bootstrap_95pct_low']:+.4f}, {row['bootstrap_95pct_high']:+.4f}] "
              f"{row['wins']}W/{row['losses']}L")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output-dir", type=Path, default=Path("results/longbench_v2_graph"))
    parser.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    args = parser.parse_args()
    evaluate(args.output_dir, args.model, args.base_url, args.limit)


if __name__ == "__main__":
    main()
