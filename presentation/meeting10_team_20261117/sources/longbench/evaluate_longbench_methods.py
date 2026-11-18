"""LangChain's MultiQueryRetriever and a DSPy program on LongBench v2 (gpt-4o-mini reader).

Protocol, fixed before any run:

- LongBench v2 (``zai-org/LongBench-v2``, 503 questions) is split once with
  seed 42: 20% dev (101 questions, used only by DSPy's optimizer) and 80% test
  (402 questions, every reported number).
- Contexts are cut into 300-word chunks and every method gets the same
  evidence budget, 8 chunks = 2,400 words, as in evaluate_longbench_graph.py.
- Reader: gpt-4o-mini, temperature 0. LongBench has no relevance labels, so
  retrieval is judged by answer accuracy.

Methods (test split):

- ``no_context``, ``bm25``, ``graph``: the official 0-shot prompt with no
  evidence, the team BM25's top 8 chunks, or the improved graph's top 8,
  exactly as in evaluate_longbench_graph.py but read by gpt-4o-mini.
- ``multiquery``: ``langchain_classic.retrievers.MultiQueryRetriever`` with its
  default prompt (the LLM writes 3 versions of the question) and
  ``include_original=True``, over the team BM25 (top 8 per query). The unique
  union (up to 32 chunks) is cut to the 8-chunk budget by reciprocal rank
  fusion (k = 60); the stock union size is recorded. Official prompt.
- ``dspy``: a DSPy module, ``ChainOfThought(context, question, choices ->
  answer: A-D)`` over the same BM25 evidence as ``bm25``. Its instruction is
  optimized by MIPROv2 (auto="light", zero-shot, so test prompts stay the size
  of ``bm25``'s) on the dev split only, then run once on test.

All LLM calls go through openai_llm.CachedOpenAIChat (exact-prompt cache,
hard cap on new calls), so reruns are free and interrupted runs resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import dspy
from datasets import load_dataset
from dspy.lm15 import Message, Response, TextPart, Usage
from langchain_classic.retrievers.multi_query import (
    DEFAULT_QUERY_PROMPT, LineListOutputParser, MultiQueryRetriever)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from evaluate_longbench_graph import (
    NO_CONTEXT_PROMPT, PROMPT, TOP_CHUNKS, bootstrap, chunk, extract_answer)
from openai_llm import CachedOpenAIChat, cost_usd
from standalone_graph import StandaloneGraphRetriever
from team_bm25 import BM25Retriever


MODEL = "gpt-4o-mini"
SEED = 42
DEV_FRACTION = 0.2
RRF_K = 60
ANSWER_MAX_TOKENS = 128   # official LongBench v2 setting for direct answers
QUERY_MAX_TOKENS = 256
DSPY_MAX_TOKENS = 1024    # room for the chain of thought
ANSWER_METHODS = ("no_context", "bm25", "graph", "multiquery")
METHODS = ANSWER_METHODS + ("dspy",)


# ---------------------------------------------------------------- data

def split(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    ids = sorted(sample["_id"] for sample in samples)
    random.Random(SEED).shuffle(ids)
    dev = set(ids[:round(len(ids) * DEV_FRACTION)])
    return ([s for s in samples if s["_id"] in dev], [s for s in samples if s["_id"] not in dev])


def retrieval_query(sample: dict) -> str:
    """Question + four choices, the query used by evaluate_longbench_graph.py."""
    return "\n".join([sample["question"]] + [sample[f"choice_{c}"] for c in "ABCD"])


def fields(sample: dict) -> dict[str, str]:
    return {"question": sample["question"].strip(), **{c: sample[f"choice_{c}"].strip() for c in "ABCD"}}


def choices_text(sample: dict) -> str:
    return "\n".join(f"({c}) {sample[f'choice_{c}'].strip()}" for c in "ABCD")


def chunk_ids(sample: dict, chunks: list[str]) -> list[str]:
    return [f"{sample['_id']}:chunk{i}" for i in range(1, len(chunks) + 1)]


def evidence(chunks: list[str], ids: list[str], chosen: list[str]) -> str:
    """Chosen chunks in document order, as in evaluate_longbench_graph.py."""
    order = {chunk_id: i for i, chunk_id in enumerate(ids)}
    return "\n\n[...]\n\n".join(chunks[order[i]] for i in sorted(chosen, key=order.get))


# ---------------------------------------------------------------- MultiQueryRetriever

class ChunkBM25(BaseRetriever):
    """The team BM25 over one question's chunks, as a LangChain retriever."""

    index: Any
    texts: dict[str, str]
    k: int = TOP_CHUNKS

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun
                                ) -> list[Document]:
        return [Document(page_content=self.texts[hit["doc_id"]], metadata={"chunk_id": hit["doc_id"]})
                for hit in self.index.search(query, top_k=self.k)]


def rrf(ranked_lists: list[list[Document]], top_n: int, k: int = RRF_K) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    first: dict[str, tuple[int, int]] = {}
    documents: dict[str, Document] = {}
    for list_index, ranked in enumerate(ranked_lists):
        for rank, document in enumerate(ranked, start=1):
            chunk_id = document.metadata["chunk_id"]
            scores[chunk_id] += 1 / (k + rank)
            first.setdefault(chunk_id, (rank, list_index))
            documents.setdefault(chunk_id, document)
    best = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], first[chunk_id]))
    return [documents[chunk_id] for chunk_id in best[:top_n]]


class BudgetedMultiQueryRetriever(MultiQueryRetriever):
    """Stock MultiQueryRetriever, except the unique union is cut to ``budget`` chunks by RRF."""

    budget: int = TOP_CHUNKS

    def retrieve_documents(self, queries: list[str], run_manager: CallbackManagerForRetrieverRun
                           ) -> list[Document]:
        ranked = [self.retriever.invoke(query, config={"callbacks": run_manager.get_child()})
                  for query in queries]
        return rrf(ranked, self.budget)


def query_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": DEFAULT_QUERY_PROMPT.format(question=question)}]


def query_chain(chat: CachedOpenAIChat):
    """DEFAULT_QUERY_PROMPT | gpt-4o-mini (through the cached client) | LineListOutputParser."""
    roles = {"human": "user", "ai": "assistant", "system": "system"}

    def call(prompt_value) -> str:
        messages = [{"role": roles[m.type], "content": m.content} for m in prompt_value.to_messages()]
        return chat(messages, label="multiquery-queries", max_tokens=QUERY_MAX_TOKENS)["text"]

    return DEFAULT_QUERY_PROMPT | RunnableLambda(call) | LineListOutputParser()


# ---------------------------------------------------------------- retrieval

def retrieve(sample: dict, with_test_methods: bool, chain) -> dict[str, Any]:
    chunks = chunk(sample["context"])
    ids = chunk_ids(sample, chunks)
    query = retrieval_query(sample)
    bm25 = BM25Retriever([{"doc_id": i, "text": t} for i, t in zip(ids, chunks)], k1=0.9, b=0.4)
    record: dict[str, Any] = {"id": sample["_id"], "chunks": len(chunks),
                              "bm25": [hit["doc_id"] for hit in bm25.search(query, top_k=TOP_CHUNKS)]}
    if not with_test_methods:
        return record
    started = time.perf_counter()
    documents = [Document(page_content=text, metadata={
        "chunk_id": chunk_id, "row_id": sample["_id"], "reference_number": 1,
        "reference_chunk_number": i}) for i, (chunk_id, text) in enumerate(zip(ids, chunks), start=1)]
    graph = StandaloneGraphRetriever(documents, top_k=TOP_CHUNKS)
    record["graph"] = [item.node_id for item in graph.retrieve(query).explanations]
    record["graph_seconds"] = time.perf_counter() - started
    base = ChunkBM25(index=bm25, texts=dict(zip(ids, chunks)))
    multiquery = BudgetedMultiQueryRetriever(retriever=base, llm_chain=chain, include_original=True,
                                             verbose=False)
    record["multiquery"] = [d.metadata["chunk_id"] for d in multiquery.invoke(query)]
    queries = chain.invoke({"question": query}) + [query]   # cached: same call the retriever made
    record["multiquery_queries"] = queries[:-1]
    record["multiquery_stock_union"] = len({hit["doc_id"] for q in queries
                                            for hit in bm25.search(q, top_k=TOP_CHUNKS)})
    return record


def load_or_retrieve(path: Path, dev: list[dict], test: list[dict], chain, chat: CachedOpenAIChat,
                     threads: int) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    todo_test = [s for s in test if "multiquery" not in done.get(s["_id"], {})]
    todo_dev = [s for s in dev if s["_id"] not in done]
    if todo_test:   # write the rewrites in parallel first; the sequential pass then hits the cache
        print(f"multi-query rewrites for {len(todo_test)} questions ...", flush=True)
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(lambda s: chain.invoke({"question": retrieval_query(s)}), todo_test))
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as handle:
        for number, (sample, is_test) in enumerate(
                [(s, True) for s in todo_test] + [(s, False) for s in todo_dev], start=1):
            record = retrieve(sample, is_test, chain)
            done[sample["_id"]] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if number % 50 == 0:
                print(f"retrieval {number}/{len(todo_test) + len(todo_dev)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    return done


# ---------------------------------------------------------------- DSPy

class CachedEngine:
    """DSPy 3.5 engine that sends each request through the capped, cached OpenAI client."""

    def __init__(self, chat: CachedOpenAIChat) -> None:
        self.chat, self.label = chat, "dspy"

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, str):
            return content
        return "".join(part.text for part in content if isinstance(part, TextPart))

    def complete(self, request) -> Response:
        messages = [] if request.system is None else [{"role": "system", "content": self._text(request.system)}]
        messages += [{"role": m.role, "content": self._text(m.parts)} for m in request.messages]
        config = request.config
        output = self.chat(messages, label=self.label, max_tokens=config.max_tokens or DSPY_MAX_TOKENS,
                           temperature=config.temperature or 0.0)
        finish = output.get("finish_reason") or "stop"
        return Response(id=None, model=output["model"], message=Message.assistant([TextPart(output["text"])]),
                        finish_reason=finish if finish in ("stop", "length", "content_filter") else "stop",
                        usage=Usage(input_tokens=output["prompt_tokens"], output_tokens=output["completion_tokens"],
                                    total_tokens=output["prompt_tokens"] + output["completion_tokens"]))


class AnswerFromExcerpts(dspy.Signature):
    """Answer the multiple-choice question about a long text, using the excerpts retrieved from it."""

    context: str = dspy.InputField(desc="excerpts retrieved from the long text, in document order")
    question: str = dspy.InputField()
    choices: str = dspy.InputField()
    answer: Literal["A", "B", "C", "D"] = dspy.OutputField()


EVIDENCE: dict[str, str] = {}   # sample id -> the BM25 evidence string (filled before DSPy runs)


class LongBenchRAG(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.respond = dspy.ChainOfThought(AnswerFromExcerpts)

    def forward(self, sample_id: str, question: str, choices: str):
        return self.respond(context=EVIDENCE[sample_id], question=question, choices=choices)


def dspy_example(sample: dict) -> dspy.Example:
    return dspy.Example(sample_id=sample["_id"], question=sample["question"].strip(),
                        choices=choices_text(sample), answer=sample["answer"]
                        ).with_inputs("sample_id", "question", "choices")


def exact_match(example, prediction, trace=None) -> float:
    return float(getattr(prediction, "answer", None) == example.answer)


def run_dspy(output_dir: Path, dev: list[dict], test: list[dict], chat: CachedOpenAIChat,
             threads: int) -> tuple[dict[str, dict], dict[str, Any]]:
    engine = CachedEngine(chat)
    lm = dspy.BaseLM(model=f"openai/{MODEL}", engine=engine, cache=False, temperature=0.0,
                     max_tokens=DSPY_MAX_TOKENS, num_retries=0)
    dspy.configure(lm=lm)
    dev_examples, test_examples = [dspy_example(s) for s in dev], [dspy_example(s) for s in test]
    program_path = output_dir / "dspy_program.json"

    def score(program, examples, label):
        engine.label = label
        result = dspy.Evaluate(devset=examples, metric=exact_match, num_threads=threads,
                               max_errors=len(examples) + 1, display_progress=False)(program)
        return result

    default = LongBenchRAG()
    compiled = LongBenchRAG()
    if program_path.exists():
        compiled.load(program_path)
    else:
        engine.label = "dspy-compile"
        optimizer = dspy.MIPROv2(metric=exact_match, auto="light", max_bootstrapped_demos=0,
                                 max_labeled_demos=0, num_threads=threads, seed=SEED)
        compiled = optimizer.compile(LongBenchRAG(), trainset=dev_examples)
        compiled.save(program_path)
    dev_default = score(default, dev_examples, "dspy-dev-default")
    dev_compiled = score(compiled, dev_examples, "dspy-dev-compiled")
    test_result = score(compiled, test_examples, "dspy-test")
    rows = {}
    with (output_dir / "dspy_test_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for example, prediction, value in test_result.results:
            answer = getattr(prediction, "answer", None)
            reasoning = getattr(prediction, "reasoning", None)   # the prompt's <plan> field
            reasoning = None if reasoning is None else str(reasoning)
            rows[example.sample_id] = {"pred": answer if answer in ("A", "B", "C", "D") else None,
                                       "correct": float(value)}
            handle.write(json.dumps({"id": example.sample_id, "answer": example.answer, "pred": answer,
                                     "reasoning": reasoning}, ensure_ascii=False) + "\n")
    info = {
        "optimizer": "MIPROv2(auto='light', max_bootstrapped_demos=0, max_labeled_demos=0, seed=42) on dev",
        "default_instruction": default.respond.predict.signature.instructions,
        "selected_instruction": compiled.respond.predict.signature.instructions,
        "dev_accuracy_default_program": dev_default.score / 100,
        "dev_accuracy_compiled_program": dev_compiled.score / 100,
        "dev_note": "dev accuracy of the compiled program is optimistic: it was selected on 80 of these questions",
    }
    return rows, info


# ---------------------------------------------------------------- main

def evaluate(output_dir: Path, threads: int, max_new_calls: int, limit: int | None, skip_dspy: bool) -> dict:
    samples = load_dataset("zai-org/LongBench-v2", split="train").to_list()
    dev, test = split(samples)
    if limit:
        dev, test = dev[:limit], test[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    chat = CachedOpenAIChat(MODEL, output_dir / "llm_cache.jsonl", max_new_calls=max_new_calls,
                            max_tokens=ANSWER_MAX_TOKENS)
    chain = query_chain(chat)
    print(f"dev {len(dev)} / test {len(test)}", flush=True)
    retrieval = load_or_retrieve(output_dir / "retrieval.jsonl", dev, test, chain, chat, threads)

    texts: dict[str, dict[str, str]] = {}
    for sample in dev + test:
        chunks = chunk(sample["context"])
        ids = chunk_ids(sample, chunks)
        record = retrieval[sample["_id"]]
        texts[sample["_id"]] = {m: evidence(chunks, ids, record[m])
                                for m in ("bm25", "graph", "multiquery") if m in record}
        EVIDENCE[sample["_id"]] = texts[sample["_id"]]["bm25"].strip()

    tasks = []
    for sample in test:
        for method in ANSWER_METHODS:
            prompt = (NO_CONTEXT_PROMPT.format(**fields(sample)) if method == "no_context"
                      else PROMPT.format(context=texts[sample["_id"]][method].strip(), **fields(sample)))
            tasks.append((sample["_id"], method, [{"role": "user", "content": prompt}]))
    new = sum(not chat.is_cached(messages) for *_, messages in tasks)
    print(f"answer calls: {len(tasks)} ({new} not cached)", flush=True)
    with ThreadPoolExecutor(threads) as pool:
        outputs = list(pool.map(lambda t: chat(t[2], label=json.dumps([t[0], t[1]])), tasks))
    answers = {(sid, method): output for (sid, method, _), output in zip(tasks, outputs)}

    dspy_rows, dspy_info = ({}, {}) if skip_dspy else run_dspy(output_dir, dev, test, chat, threads)
    methods = METHODS if dspy_rows else ANSWER_METHODS

    per_question = []
    for sample in test:
        sid = sample["_id"]
        row = {"id": sid, "domain": sample["domain"], "difficulty": sample["difficulty"],
               "length": sample["length"], "answer": sample["answer"]}
        for method in ANSWER_METHODS:
            output = answers[(sid, method)]
            prediction = extract_answer(output["text"])
            row[f"{method}_pred"] = prediction
            row[f"{method}_correct"] = float(prediction == sample["answer"])
            row[f"{method}_prompt_tokens"] = output["prompt_tokens"]
        if dspy_rows:
            row["dspy_pred"] = dspy_rows[sid]["pred"]
            row["dspy_correct"] = dspy_rows[sid]["correct"]
        per_question.append(row)

    groups = [("all", lambda r: True)]
    for key in ("difficulty", "length", "domain"):
        for value in sorted({r[key] for r in per_question}):
            groups.append((f"{key}={value}", lambda r, key=key, value=value: r[key] == value))
    accuracy = []
    for name, keep in groups:
        rows = [r for r in per_question if keep(r)]
        accuracy.append({"group": name, "n": len(rows),
                         **{m: statistics.fmean(r[f"{m}_correct"] for r in rows) for m in methods},
                         **{f"{m}_unparsed": statistics.fmean(r[f"{m}_pred"] is None for r in rows)
                            for m in methods}})
    paired = []
    comparisons = [(m, "bm25") for m in methods if m != "bm25"]
    if dspy_rows:
        comparisons.append(("dspy", "multiquery"))
    for method, other in comparisons:
        deltas = [r[f"{method}_correct"] - r[f"{other}_correct"] for r in per_question]
        low, high = bootstrap(deltas)
        paired.append({"comparison": f"{method} minus {other}", "n": len(deltas),
                       "accuracy_difference": statistics.fmean(deltas),
                       "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                       "bootstrap_95pct_low": low, "bootstrap_95pct_high": high})

    # Cost per method as if run from scratch: tokens of every call the method makes, cached or not.
    labels: dict[str, list[dict]] = defaultdict(list)
    for line in (output_dir / "llm_cache.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["label"].startswith("dspy"):
            labels[entry["label"]].append(entry["output"])
    cost_rows = []

    def add_cost(name: str, outputs: list[dict], note: str) -> None:
        prompt = sum(o["prompt_tokens"] for o in outputs)
        completion = sum(o["completion_tokens"] for o in outputs)
        cost_rows.append({"component": name, "calls": len(outputs), "prompt_tokens": prompt,
                          "completion_tokens": completion, "usd": cost_usd(MODEL, prompt, completion),
                          "usd_per_test_question": (cost_usd(MODEL, prompt, completion) / len(test)
                                                    if not name.startswith("dspy compile") else None),
                          "note": note})

    for method in ANSWER_METHODS:
        add_cost(f"{method} answer", [answers[(s["_id"], method)] for s in test], "official prompt, test")
    add_cost("multiquery rewrites",
             [chat(query_messages(retrieval_query(s)), max_tokens=QUERY_MAX_TOKENS) for s in test],
             "LangChain default prompt, 3 rewrites per question")
    if dspy_rows:
        add_cost("dspy compile (MIPROv2 on dev)", labels["dspy-compile"], "one-off optimization cost")
        add_cost("dspy test", labels["dspy-test"], "ChainOfThought, compiled instruction")

    traces = [{"id": s["_id"], **retrieval[s["_id"]],
               "overlap_bm25": {m: len(set(retrieval[s["_id"]][m]) & set(retrieval[s["_id"]]["bm25"]))
                                for m in ("graph", "multiquery")}} for s in test]
    for filename, rows in (("per_question.csv", per_question), ("accuracy.csv", accuracy),
                           ("paired_accuracy_intervals.csv", paired), ("cost.csv", cost_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    gold = [r["answer"] for r in per_question]
    summary = {
        "dataset": "LongBench v2 (zai-org/LongBench-v2), 503 questions",
        "split": {"seed": SEED, "dev": len(dev), "test": len(test),
                  "dev_use": "DSPy's MIPROv2 only; every reported accuracy is on test"},
        "reader": MODEL, "temperature": 0, "chunk_words": 300, "top_chunks": TOP_CHUNKS, "budget_words": 2400,
        "retrieval_query": "question + four choices",
        "chance_on_test": {"uniform": 0.25,
                           "majority_letter": max(gold.count(c) for c in "ABCD") / len(gold)},
        "multiquery": {"class": "langchain_classic.retrievers.multi_query.MultiQueryRetriever",
                       "prompt": "DEFAULT_QUERY_PROMPT", "include_original": True, "per_query_k": TOP_CHUNKS,
                       "budget_fusion": f"reciprocal rank fusion, k={RRF_K}, top {TOP_CHUNKS}",
                       "mean_generated_queries": statistics.fmean(len(t["multiquery_queries"]) for t in traces),
                       "mean_stock_union_chunks": statistics.fmean(t["multiquery_stock_union"] for t in traces)},
        "mean_overlap_with_bm25_chunks": {m: statistics.fmean(t["overlap_bm25"][m] for t in traces)
                                          for m in ("graph", "multiquery")},
        "dspy": dspy_info,
        "new_llm_calls_this_run": chat.new_calls,
        "spend_this_run_usd": cost_usd(MODEL, chat.usage["prompt_tokens"], chat.usage["completion_tokens"]),
        "protocol": ("Fixed configurations; only DSPy's instruction was optimized, on dev. Official LongBench v2 "
                     "0-shot prompt and answer extraction for the non-DSPy methods; temperature 0."),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(" ".join(f"{m}={accuracy[0][m]:.4f}" for m in methods))
    for row in paired:
        print(f"{row['comparison']:26s} {row['accuracy_difference']:+.4f} "
              f"[{row['bootstrap_95pct_low']:+.4f}, {row['bootstrap_95pct_high']:+.4f}] "
              f"{row['wins']}W/{row['losses']}L")
    print(f"new calls {chat.new_calls}, spend ${summary['spend_this_run_usd']:.3f}")
    if chat.rate_limits:
        print("rate limits:", {k: v for k, v in chat.rate_limits.items() if "limit-" in k})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/longbench_v2_methods"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-calls", type=int, default=3500)
    parser.add_argument("--limit", type=int, help="first N dev and N test questions only (smoke test)")
    parser.add_argument("--skip-dspy", action="store_true")
    args = parser.parse_args()
    evaluate(args.output_dir, args.threads, args.max_new_calls, args.limit, args.skip_dspy)


if __name__ == "__main__":
    main()
