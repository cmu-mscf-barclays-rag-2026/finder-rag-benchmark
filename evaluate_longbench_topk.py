"""Evidence-budget sweep on LongBench v2: top k chunks for k in 2, 4, 8, 16.

Runs the four retrieval methods of evaluate_longbench_methods.py at several
budgets (300-word chunks, so 600 to 4,800 words), with everything else fixed:

- ``bm25`` and ``graph``: the first k of each retriever's ranking (both
  rankings are prefix-stable, so one top-16 retrieval serves every k).
- ``multiquery``: LangChain's MultiQueryRetriever with BM25 top k per query,
  the union cut to k chunks by reciprocal rank fusion.
- ``dspy``: the saved DSPy program (results/longbench_v2_methods/
  dspy_program.json; MIPROv2 kept the default instruction) over BM25's top k.

The reader is gpt-4o-mini at temperature 0, with the official prompt for the
first three. Both splits of the earlier protocol are run: k is chosen per
method on the 101 dev questions (ties go to the smaller, cheaper k), and the
402 test questions give the curve and the comparisons. The LLM cache is shared
with evaluate_longbench_methods.py, so its k = 8 calls are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
import tiktoken
from datasets import load_dataset
from langchain_core.documents import Document

import evaluate_longbench_methods as M
from evaluate_longbench_graph import PROMPT, bootstrap, chunk, extract_answer
from openai_llm import CachedOpenAIChat, cost_usd
from standalone_graph import StandaloneGraphRetriever
from team_bm25 import BM25Retriever


K_VALUES = (2, 4, 8, 16)
OFFICIAL = ("bm25", "graph", "multiquery")
METHODS = OFFICIAL + ("dspy",)
SHARED_CACHE = Path("results/longbench_v2_methods/llm_cache.jsonl")
PROGRAM = Path("results/longbench_v2_methods/dspy_program.json")
# gpt-4o-mini has a 128k-token window. A few chunks of structured data hold very long
# whitespace-free "words", so 16 chunks can exceed it; such evidence keeps its first and
# last 60k tokens (middle truncation, as LongBench does). No evidence at k <= 8 is that long.
EVIDENCE_TOKEN_CAP = 120_000
ENCODING = tiktoken.get_encoding("o200k_base")   # gpt-4o-mini's tokenizer


def fit_window(text: str, cap: int = EVIDENCE_TOKEN_CAP) -> tuple[str, bool]:
    if len(text) <= cap:   # a token is at least one character, so short texts cannot exceed the cap
        return text, False
    tokens = ENCODING.encode(text, disallowed_special=())
    if len(tokens) <= cap:
        return text, False
    return ENCODING.decode(tokens[:cap // 2]) + "\n\n[...]\n\n" + ENCODING.decode(tokens[-(cap // 2):]), True


def retrieve(sample: dict, chain) -> dict:
    chunks = chunk(sample["context"])
    ids = M.chunk_ids(sample, chunks)
    query = M.retrieval_query(sample)
    kmax = max(K_VALUES)
    bm25 = BM25Retriever([{"doc_id": i, "text": t} for i, t in zip(ids, chunks)], k1=0.9, b=0.4)
    documents = [Document(page_content=text, metadata={
        "chunk_id": chunk_id, "row_id": sample["_id"], "reference_number": 1,
        "reference_chunk_number": i}) for i, (chunk_id, text) in enumerate(zip(ids, chunks), start=1)]
    graph = StandaloneGraphRetriever(documents, top_k=kmax)
    multiquery = {}
    for k in K_VALUES:
        base = M.ChunkBM25(index=bm25, texts=dict(zip(ids, chunks)), k=k)
        retriever = M.BudgetedMultiQueryRetriever(retriever=base, llm_chain=chain, include_original=True,
                                                  budget=k, verbose=False)
        multiquery[str(k)] = [d.metadata["chunk_id"] for d in retriever.invoke(query)]
    return {"id": sample["_id"], "chunks": len(chunks),
            "bm25": [hit["doc_id"] for hit in bm25.search(query, top_k=kmax)],
            "graph": [item.node_id for item in graph.retrieve(query).explanations],
            "multiquery": multiquery}


def ranked(record: dict, method: str, k: int) -> list[str]:
    if method == "multiquery":
        return record["multiquery"][str(k)]
    return record["bm25" if method == "dspy" else method][:k]


class TallyEngine(M.CachedEngine):
    """CachedEngine that also adds up tokens per label, cached calls included."""

    def __init__(self, chat: CachedOpenAIChat) -> None:
        super().__init__(chat)
        self.tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        self._lock = threading.Lock()

    def complete(self, request):
        response = super().complete(request)
        with self._lock:
            tally = self.tokens[self.label]
            tally[0] += 1
            tally[1] += response.usage.input_tokens
            tally[2] += response.usage.output_tokens
        return response


def evaluate(output_dir: Path, threads: int, max_new_calls: int, limit: int | None) -> dict:
    samples = load_dataset("zai-org/LongBench-v2", split="train").to_list()
    dev, test = M.split(samples)
    if limit:
        dev, test = dev[:limit], test[:limit]
    splits = {"dev": dev, "test": test}
    output_dir.mkdir(parents=True, exist_ok=True)
    chat = CachedOpenAIChat(M.MODEL, SHARED_CACHE, max_new_calls=max_new_calls,
                            max_tokens=M.ANSWER_MAX_TOKENS)
    chain = M.query_chain(chat)

    path = output_dir / "retrieval.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    todo = [s for s in dev + test if s["_id"] not in done]
    if todo:
        print(f"multi-query rewrites for {len(todo)} questions (cached ones are free) ...", flush=True)
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(lambda s: chain.invoke({"question": M.retrieval_query(s)}), todo))
        with path.open("a", encoding="utf-8") as handle:
            for number, sample in enumerate(todo, start=1):
                done[sample["_id"]] = retrieve(sample, chain)
                handle.write(json.dumps(done[sample["_id"]], ensure_ascii=False) + "\n")
                if number % 100 == 0:
                    print(f"retrieval {number}/{len(todo)}", flush=True)
    earlier = Path("results/longbench_v2_methods/retrieval.jsonl")
    if earlier.exists():   # k = 8 must reproduce the earlier run exactly
        old = {json.loads(l)["id"]: json.loads(l) for l in earlier.read_text(encoding="utf-8").splitlines()}
        mismatched = [sid for sid in (s["_id"] for s in test) if sid in old and "graph" in old[sid] and any(
            old[sid][m] != ranked(done[sid], m, 8) for m in OFFICIAL)]
        print(f"k=8 retrieval matches the earlier run on {len(test) - len(mismatched)}/{len(test)} test questions")

    texts: dict[tuple[str, str, int], str] = {}
    truncated: list[dict] = []
    for sample in dev + test:
        chunks = chunk(sample["context"])
        ids = M.chunk_ids(sample, chunks)
        for method in METHODS:
            for k in K_VALUES:
                text, cut = fit_window(M.evidence(chunks, ids, ranked(done[sample["_id"]], method, k)).strip())
                texts[(sample["_id"], method, k)] = text
                if cut:
                    truncated.append({"id": sample["_id"], "method": method, "k": k})

    tasks = []
    for name, rows in splits.items():
        for sample in rows:
            for method in OFFICIAL:
                for k in K_VALUES:
                    prompt = PROMPT.format(context=texts[(sample["_id"], method, k)], **M.fields(sample))
                    tasks.append((sample["_id"], method, k, [{"role": "user", "content": prompt}]))
    new = sum(not chat.is_cached(t[3]) for t in tasks)
    print(f"official-prompt calls: {len(tasks)} ({new} not cached)", flush=True)
    with ThreadPoolExecutor(threads) as pool:
        outputs = list(pool.map(lambda t: chat(t[3], label=json.dumps([t[0], t[1], t[2]])), tasks))
    answers = {(sid, method, k): output for (sid, method, k, _), output in zip(tasks, outputs)}

    engine = TallyEngine(chat)
    dspy.configure(lm=dspy.BaseLM(model=f"openai/{M.MODEL}", engine=engine, cache=False, temperature=0.0,
                                  max_tokens=M.DSPY_MAX_TOKENS, num_retries=0))
    program = M.LongBenchRAG()
    program.load(PROGRAM)
    dspy_rows: dict[tuple[str, int], dict] = {}
    for name, rows in splits.items():
        for k in K_VALUES:
            examples = []
            for sample in rows:
                key = f"{sample['_id']}@k{k}"   # not shown to the LM; prompts match the earlier run at k = 8
                M.EVIDENCE[key] = texts[(sample["_id"], "dspy", k)]
                example = M.dspy_example(sample)
                examples.append(dspy.Example(**{**example.toDict(), "sample_id": key}
                                             ).with_inputs("sample_id", "question", "choices"))
            engine.label = f"topk-dspy-{name}-k{k}"
            print(f"dspy {name} k={k} ...", flush=True)
            result = dspy.Evaluate(devset=examples, metric=M.exact_match, num_threads=threads,
                                   max_errors=len(examples) + 1)(program)
            for example, prediction, value in result.results:
                answer = getattr(prediction, "answer", None)
                dspy_rows[(example.sample_id.split("@")[0], k)] = {
                    "pred": answer if answer in ("A", "B", "C", "D") else None, "correct": float(value)}

    per_question = []
    for name, rows in splits.items():
        for sample in rows:
            sid = sample["_id"]
            row = {"id": sid, "split": name, "domain": sample["domain"], "difficulty": sample["difficulty"],
                   "length": sample["length"], "answer": sample["answer"]}
            for method in METHODS:
                for k in K_VALUES:
                    if method == "dspy":
                        prediction = dspy_rows[(sid, k)]["pred"]
                    else:
                        prediction = extract_answer(answers[(sid, method, k)]["text"])
                        row[f"{method}_k{k}_prompt_tokens"] = answers[(sid, method, k)]["prompt_tokens"]
                    row[f"{method}_k{k}_pred"] = prediction
                    row[f"{method}_k{k}_correct"] = float(prediction == sample["answer"])
            per_question.append(row)

    def rows_of(name):
        return [r for r in per_question if r["split"] == name]

    curve = []
    for name in splits:
        rows = rows_of(name)
        for method in METHODS:
            for k in K_VALUES:
                if method == "dspy":
                    calls, prompt, completion = engine.tokens[f"topk-dspy-{name}-k{k}"]
                else:
                    outs = [answers[(r["id"], method, k)] for r in rows]
                    calls = len(outs)
                    prompt = sum(o["prompt_tokens"] for o in outs)
                    completion = sum(o["completion_tokens"] for o in outs)
                cost = cost_usd(M.MODEL, prompt, completion)
                curve.append({"split": name, "method": method, "k": k, "words": 300 * k, "n": len(rows),
                              "accuracy": statistics.fmean(r[f"{method}_k{k}_correct"] for r in rows),
                              "unparsed": statistics.fmean(r[f"{method}_k{k}_pred"] is None for r in rows),
                              "mean_prompt_tokens": prompt / max(calls, 1),
                              "usd_per_1000_questions": 1000 * cost / len(rows)})

    def accuracy(name, method, k):
        return next(c["accuracy"] for c in curve if (c["split"], c["method"], c["k"]) == (name, method, k))

    selected = {m: max(K_VALUES, key=lambda k: (accuracy("dev", m, k), -k)) for m in METHODS}
    test_rows = rows_of("test")
    paired = []

    def compare(label, a, b):
        deltas = [r[a] - r[b] for r in test_rows]
        low, high = bootstrap(deltas)
        paired.append({"comparison": label, "n": len(deltas), "accuracy_difference": statistics.fmean(deltas),
                       "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                       "bootstrap_95pct_low": low, "bootstrap_95pct_high": high})

    for k in K_VALUES:
        for method in ("graph", "multiquery", "dspy"):
            compare(f"k={k}: {method} minus bm25", f"{method}_k{k}_correct", f"bm25_k{k}_correct")
    for method in METHODS:
        for k in K_VALUES:
            if k != 8:
                compare(f"{method}: k={k} minus k=8", f"{method}_k{k}_correct", f"{method}_k8_correct")
    for method in ("graph", "multiquery", "dspy"):
        compare(f"dev-selected k: {method} (k={selected[method]}) minus bm25 (k={selected['bm25']})",
                f"{method}_k{selected[method]}_correct", f"bm25_k{selected['bm25']}_correct")

    for filename, rows in (("per_question.csv", per_question), ("curve.csv", curve),
                           ("paired_accuracy_intervals.csv", paired)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "k_values": list(K_VALUES), "chunk_words": 300, "reader": M.MODEL, "temperature": 0,
        "split": {"dev": len(dev), "test": len(test)},
        "dev_selected_k": selected,
        "selection_rule": "highest dev accuracy per method; ties go to the smaller k",
        "test_accuracy_at_dev_selected_k": {m: accuracy("test", m, selected[m]) for m in METHODS},
        "new_llm_calls_this_run": chat.new_calls,
        "spend_this_run_usd": cost_usd(M.MODEL, chat.usage["prompt_tokens"], chat.usage["completion_tokens"]),
        "llm_cache": str(SHARED_CACHE),
        "evidence_token_cap": EVIDENCE_TOKEN_CAP,
        "evidence_truncated_to_fit_the_window": truncated,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name in splits:
        print(f"{name}: " + " | ".join(
            f"{m} " + " ".join(f"k{k}={accuracy(name, m, k):.3f}" for k in K_VALUES) for m in METHODS))
    print("dev-selected k:", selected)
    for row in paired:
        print(f"{row['comparison']:58s} {row['accuracy_difference']:+.4f} "
              f"[{row['bootstrap_95pct_low']:+.4f}, {row['bootstrap_95pct_high']:+.4f}]")
    print(f"new calls {chat.new_calls}, spend ${summary['spend_this_run_usd']:.3f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/longbench_v2_topk"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-new-calls", type=int, default=7500)
    parser.add_argument("--limit", type=int, help="first N dev and N test questions only (smoke test)")
    args = parser.parse_args()
    evaluate(args.output_dir, args.threads, args.max_new_calls, args.limit)


if __name__ == "__main__":
    main()
