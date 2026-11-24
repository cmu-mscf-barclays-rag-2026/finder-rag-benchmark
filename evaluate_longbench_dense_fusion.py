"""Dense reference, dense-based MultiQuery/DSPy, and rank fusion on LongBench v2.

Follows up evaluate_longbench_methods.py with three changes suggested by the
team: compare against dense retrieval instead of BM25, put MultiQueryRetriever
and DSPy on a dense base, and fuse sparse and dense rankings with reciprocal
rank fusion and the rules compared by Franca et al. (2025, arXiv:2507.03761,
"Ranking-based Fusion Algorithms for Extreme Multi-label Text Classification").

Protocol (same as before): seed-42 split, 101 dev / 402 test questions,
300-word chunks, 8 chunks (2,400 words) of evidence, gpt-4o-mini at
temperature 0, official prompt except for DSPy. The query is the question
plus its four choices.

- ``dense``: all-MiniLM-L6-v2 (the team's dense model; it reads the first 256
  word pieces of each chunk), cosine similarity, top 8.
- ``multiquery_dense``: the same MultiQueryRetriever setup as before (default
  prompt, 3 rewrites + the original, union cut to 8 by RRF) over the dense
  retriever instead of BM25. The rewrites are the cached ones.
- ``dspy_dense``: the saved DSPy program (MIPROv2 kept its default
  instruction) over the dense top 8.
- ``hybrid_<rule>``: BM25 and dense fused over each one's top 64 candidates
  (the paper's pool size), top 8 kept. Rules: RRF (k = 60); CombSUM and
  CombMNZ on min-max normalized scores; CombMNZ on ZMUV (z-score) normalized
  scores, the paper's best combination; ISR (inverse square rank); BordaFuse.
  A document missing from one run contributes 0 to score fusion. The rule
  to report is chosen on dev; all are shown on test.
- ``graph_dense_rrf``: the graph and dense fused by RRF the same way.
- ``no_context``, ``bm25``, ``graph``, ``multiquery_bm25``, ``dspy_bm25``: the
  earlier methods, re-read from the shared cache for the same table.

The LLM cache is shared with evaluate_longbench_methods.py. Chunk embeddings
are cached outside the repository (~/.cache/finder-rag-baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import dspy
import numpy as np
from datasets import load_dataset
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import evaluate_longbench_methods as M
from evaluate_longbench_graph import NO_CONTEXT_PROMPT, PROMPT, TOP_CHUNKS, bootstrap, chunk, extract_answer
from evaluate_longbench_topk import PROGRAM, SHARED_CACHE, TallyEngine
from openai_llm import CachedOpenAIChat, cost_usd
from standalone_graph import StandaloneGraphRetriever
from team_bm25 import BM25Retriever


DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIR = Path.home() / ".cache" / "finder-rag-baseline" / "longbench_minilm"
POOL = 64          # candidates per retriever before fusion (as in the paper)
RRF_K = 60
FUSION_RULES = ("rrf", "combsum_minmax", "combmnz_minmax", "combmnz_zmuv", "isr", "bordafuse")
OFFICIAL = (("no_context", "bm25", "graph", "multiquery_bm25", "dense", "multiquery_dense")
            + tuple(f"hybrid_{rule}" for rule in FUSION_RULES) + ("graph_dense_rrf",))
DSPY = ("dspy_bm25", "dspy_dense")
METHODS = OFFICIAL + DSPY


# ---------------------------------------------------------------- fusion

def normalize(scores: list[float], how: str) -> list[float]:
    if how == "minmax":
        low, high = min(scores), max(scores)
        return [(s - low) / (high - low) if high > low else 1.0 for s in scores]
    if how == "zmuv":
        mean, spread = statistics.fmean(scores), statistics.pstdev(scores)
        return [(s - mean) / spread if spread else 0.0 for s in scores]
    raise ValueError(how)


def fuse(runs: list[list[tuple[str, float]]], rule: str, top_n: int = TOP_CHUNKS) -> list[str]:
    """Fuse ranked (id, score) lists, best first, with one of FUSION_RULES."""
    first: dict[str, tuple[int, int]] = {}
    for run_index, run in enumerate(runs):
        for rank, (doc, _) in enumerate(run, start=1):
            first[doc] = min(first.get(doc, (rank, run_index)), (rank, run_index))
    fused: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for run in runs:
        for doc, _ in run:
            count[doc] += 1
    if rule == "rrf":
        for run in runs:
            for rank, (doc, _) in enumerate(run, start=1):
                fused[doc] += 1 / (RRF_K + rank)
    elif rule.startswith("comb"):
        combiner, how = rule.split("_")
        for run in runs:
            for (doc, _), value in zip(run, normalize([s for _, s in run], how)):
                fused[doc] += value
        if combiner == "combmnz":
            fused = defaultdict(float, {doc: value * count[doc] for doc, value in fused.items()})
    elif rule == "isr":   # inverse square rank: |runs containing d| * sum of 1 / rank^2
        for run in runs:
            for rank, (doc, _) in enumerate(run, start=1):
                fused[doc] += 1 / rank ** 2
        fused = defaultdict(float, {doc: value * count[doc] for doc, value in fused.items()})
    elif rule == "bordafuse":   # unranked candidates share a run's leftover points equally
        pool = len(first)
        for run in runs:
            ranked = {doc for doc, _ in run}
            for rank, (doc, _) in enumerate(run, start=1):
                fused[doc] += pool - rank + 1
            for doc in first:
                if doc not in ranked:
                    fused[doc] += (pool - len(run) + 1) / 2
    else:
        raise ValueError(rule)
    order = sorted(first, key=lambda doc: (-fused[doc], first[doc], doc))
    return order[:top_n]


# ---------------------------------------------------------------- dense retrieval

class Encoder:
    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(DENSE_MODEL, device=self.device)

    def chunks(self, sample_id: str, texts: list[str]) -> np.ndarray:
        path = EMBED_DIR / f"{sample_id}.npy"
        if path.exists():
            return np.load(path).astype(np.float32)
        matrix = self.model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix.astype(np.float16))
        return matrix.astype(np.float32)

    def query(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def dense_ranking(matrix: np.ndarray, query: np.ndarray, ids: list[str], depth: int) -> list[tuple[str, float]]:
    scores = matrix @ query
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))[:depth]
    return [(ids[i], float(scores[i])) for i in order]


class ChunkDense(BaseRetriever):
    """MiniLM cosine search over one question's chunks, as a LangChain retriever."""

    matrix: Any
    ids: list[str]
    texts: dict[str, str]
    encoder: Any
    k: int = TOP_CHUNKS

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun
                                ) -> list[Document]:
        ranked = dense_ranking(self.matrix, self.encoder.query(query), self.ids, self.k)
        return [Document(page_content=self.texts[i], metadata={"chunk_id": i}) for i, _ in ranked]


def retrieve(sample: dict, encoder: Encoder, chain, earlier: dict) -> dict:
    chunks = chunk(sample["context"])
    ids = M.chunk_ids(sample, chunks)
    query = M.retrieval_query(sample)
    started = time.perf_counter()
    matrix = encoder.chunks(sample["_id"], chunks)
    dense = dense_ranking(matrix, encoder.query(query), ids, POOL)
    dense_seconds = time.perf_counter() - started
    bm25_index = BM25Retriever([{"doc_id": i, "text": t} for i, t in zip(ids, chunks)], k1=0.9, b=0.4)
    bm25 = [(hit["doc_id"], hit["score"]) for hit in bm25_index.search(query, top_k=POOL)]
    documents = [Document(page_content=text, metadata={
        "chunk_id": chunk_id, "row_id": sample["_id"], "reference_number": 1,
        "reference_chunk_number": i}) for i, (chunk_id, text) in enumerate(zip(ids, chunks), start=1)]
    graph = [(e.node_id, e.final_score)
             for e in StandaloneGraphRetriever(documents, top_k=POOL).retrieve(query).explanations]
    base = ChunkDense(matrix=matrix, ids=ids, texts=dict(zip(ids, chunks)), encoder=encoder)
    multiquery = M.BudgetedMultiQueryRetriever(retriever=base, llm_chain=chain, include_original=True,
                                               verbose=False)
    record = {"id": sample["_id"], "chunks": len(chunks), "dense_seconds": dense_seconds,
              "dense": dense, "bm25": bm25, "graph": graph,
              "multiquery_dense": [d.metadata["chunk_id"] for d in multiquery.invoke(query)],
              "multiquery_bm25": earlier[sample["_id"]]["multiquery"]["8"]}
    for rule in FUSION_RULES:
        record[f"hybrid_{rule}"] = fuse([bm25, dense], rule)
    record["graph_dense_rrf"] = fuse([graph, dense], "rrf")
    return record


def evidence_ids(record: dict, method: str) -> list[str]:
    if method in ("bm25", "graph", "dense", "dspy_bm25", "dspy_dense"):
        source = {"dspy_bm25": "bm25", "dspy_dense": "dense"}.get(method, method)
        return [doc for doc, _ in record[source][:TOP_CHUNKS]]
    return record[method]


# ---------------------------------------------------------------- main

def evaluate(output_dir: Path, threads: int, max_new_calls: int, limit: int | None) -> dict:
    samples = load_dataset("zai-org/LongBench-v2", split="train").to_list()
    dev, test = M.split(samples)
    if limit:
        dev, test = dev[:limit], test[:limit]
    splits = {"dev": dev, "test": test}
    output_dir.mkdir(parents=True, exist_ok=True)
    chat = CachedOpenAIChat(M.MODEL, SHARED_CACHE, max_new_calls=max_new_calls, max_tokens=M.ANSWER_MAX_TOKENS)
    chain = M.query_chain(chat)
    earlier = {json.loads(l)["id"]: json.loads(l) for l in
               Path("results/longbench_v2_topk/retrieval.jsonl").read_text(encoding="utf-8").splitlines()}

    path = output_dir / "retrieval.jsonl"
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    todo = [s for s in dev + test if s["_id"] not in done]
    if todo:
        encoder = Encoder()
        print(f"retrieval for {len(todo)} questions on {encoder.device} ...", flush=True)
        with ThreadPoolExecutor(threads) as pool:   # rewrites are cached; this only fills gaps
            list(pool.map(lambda s: chain.invoke({"question": M.retrieval_query(s)}), todo))
        started = time.perf_counter()
        with path.open("a", encoding="utf-8") as handle:
            for number, sample in enumerate(todo, start=1):
                done[sample["_id"]] = retrieve(sample, encoder, chain, earlier)
                handle.write(json.dumps(done[sample["_id"]], ensure_ascii=False) + "\n")
                if number % 50 == 0:
                    print(f"retrieval {number}/{len(todo)} ({time.perf_counter() - started:.0f}s)", flush=True)
    for sample in test:   # the earlier methods must reproduce the earlier run's evidence
        record, old = done[sample["_id"]], earlier[sample["_id"]]
        assert evidence_ids(record, "bm25") == old["bm25"][:8] and evidence_ids(record, "graph") == old["graph"][:8]

    texts = {}
    for sample in dev + test:
        chunks = chunk(sample["context"])
        ids = M.chunk_ids(sample, chunks)
        for method in METHODS:
            if method != "no_context":
                texts[(sample["_id"], method)] = M.evidence(chunks, ids, evidence_ids(done[sample["_id"]], method)).strip()

    tasks = []
    for rows in splits.values():
        for sample in rows:
            for method in OFFICIAL:
                prompt = (NO_CONTEXT_PROMPT.format(**M.fields(sample)) if method == "no_context"
                          else PROMPT.format(context=texts[(sample["_id"], method)], **M.fields(sample)))
                tasks.append((sample["_id"], method, [{"role": "user", "content": prompt}]))
    new = sum(not chat.is_cached(t[2]) for t in tasks)
    print(f"official-prompt calls: {len(tasks)} ({new} not cached)", flush=True)
    with ThreadPoolExecutor(threads) as pool:
        outputs = list(pool.map(lambda t: chat(t[2], label=json.dumps([t[0], t[1], "dense_fusion"])), tasks))
    answers = {(sid, method): output for (sid, method, _), output in zip(tasks, outputs)}

    engine = TallyEngine(chat)
    dspy.configure(lm=dspy.BaseLM(model=f"openai/{M.MODEL}", engine=engine, cache=False, temperature=0.0,
                                  max_tokens=M.DSPY_MAX_TOKENS, num_retries=0))
    program = M.LongBenchRAG()
    program.load(PROGRAM)
    dspy_rows = {}
    for name, rows in splits.items():
        for method in DSPY:
            examples = []
            for sample in rows:
                key = f"{sample['_id']}@{method}"   # not shown to the LM
                M.EVIDENCE[key] = texts[(sample["_id"], method)]
                examples.append(dspy.Example(**{**M.dspy_example(sample).toDict(), "sample_id": key}
                                             ).with_inputs("sample_id", "question", "choices"))
            engine.label = f"fusion-{method}-{name}"
            print(f"{method} {name} ...", flush=True)
            result = dspy.Evaluate(devset=examples, metric=M.exact_match, num_threads=threads,
                                   max_errors=len(examples) + 1)(program)
            for example, prediction, value in result.results:
                answer = getattr(prediction, "answer", None)
                dspy_rows[(example.sample_id.split("@")[0], method)] = answer if answer in ("A", "B", "C", "D") else None

    per_question = []
    for name, rows in splits.items():
        for sample in rows:
            sid = sample["_id"]
            row = {"id": sid, "split": name, "domain": sample["domain"], "difficulty": sample["difficulty"],
                   "length": sample["length"], "answer": sample["answer"]}
            for method in METHODS:
                prediction = (dspy_rows[(sid, method)] if method in DSPY
                              else extract_answer(answers[(sid, method)]["text"]))
                row[f"{method}_pred"] = prediction
                row[f"{method}_correct"] = float(prediction == sample["answer"])
            per_question.append(row)

    accuracy_rows = []
    for name in splits:
        rows = [r for r in per_question if r["split"] == name]
        groups = [("all", lambda r: True)]
        if name == "test":
            for key in ("difficulty", "length", "domain"):
                for value in sorted({r[key] for r in rows}):
                    groups.append((f"{key}={value}", lambda r, key=key, value=value: r[key] == value))
        for group, keep in groups:
            subset = [r for r in rows if keep(r)]
            accuracy_rows.append({"split": name, "group": group, "n": len(subset),
                                  **{m: statistics.fmean(r[f"{m}_correct"] for r in subset) for m in METHODS},
                                  **{f"{m}_unparsed": statistics.fmean(r[f"{m}_pred"] is None for r in subset)
                                     for m in METHODS}})

    def accuracy(name: str, method: str) -> float:
        return next(r[method] for r in accuracy_rows if r["split"] == name and r["group"] == "all")

    selected_rule = max(FUSION_RULES, key=lambda rule: (accuracy("dev", f"hybrid_{rule}"),
                                                        -FUSION_RULES.index(rule)))
    test_rows = [r for r in per_question if r["split"] == "test"]
    paired = []
    for reference in ("dense", "bm25"):
        for method in METHODS:
            if method == reference:
                continue
            deltas = [r[f"{method}_correct"] - r[f"{reference}_correct"] for r in test_rows]
            low, high = bootstrap(deltas)
            paired.append({"comparison": f"{method} minus {reference}", "n": len(deltas),
                           "accuracy_difference": statistics.fmean(deltas),
                           "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                           "bootstrap_95pct_low": low, "bootstrap_95pct_high": high})

    cost_rows = []
    for method in METHODS:
        if method in DSPY:
            calls, prompt, completion = engine.tokens[f"fusion-{method}-test"]
        else:
            outs = [answers[(r["id"], method)] for r in test_rows]
            calls = len(outs)
            prompt = sum(o["prompt_tokens"] for o in outs)
            completion = sum(o["completion_tokens"] for o in outs)
        cost_rows.append({"method": method, "test_calls": calls, "mean_prompt_tokens": prompt / max(calls, 1),
                          "usd_per_1000_questions": 1000 * cost_usd(M.MODEL, prompt, completion) / len(test_rows)})

    def overlap(method: str, other: str) -> float:
        return statistics.fmean(len(set(evidence_ids(done[s["_id"]], method)) & set(evidence_ids(done[s["_id"]], other)))
                                for s in test)

    for filename, rows in (("per_question.csv", per_question), ("accuracy.csv", accuracy_rows),
                           ("paired_accuracy_intervals.csv", paired), ("cost.csv", cost_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "dense_model": DENSE_MODEL, "dense_max_word_pieces": 256, "fusion_pool_per_retriever": POOL,
        "rrf_k": RRF_K, "fusion_rules": list(FUSION_RULES), "split": {"dev": len(dev), "test": len(test)},
        "top_chunks": TOP_CHUNKS, "reader": M.MODEL,
        "dev_selected_fusion_rule": selected_rule,
        "dev_accuracy": {m: accuracy("dev", m) for m in METHODS},
        "test_accuracy": {m: accuracy("test", m) for m in METHODS},
        "mean_overlap_of_8_chunks": {f"{a} vs {b}": overlap(a, b) for a, b in (
            ("dense", "bm25"), ("dense", "graph"), ("multiquery_dense", "dense"),
            (f"hybrid_{selected_rule}", "dense"), (f"hybrid_{selected_rule}", "bm25"))},
        "mean_dense_seconds_per_question": statistics.fmean(done[s["_id"]]["dense_seconds"] for s in dev + test),
        "new_llm_calls_this_run": chat.new_calls,
        "spend_this_run_usd": cost_usd(M.MODEL, chat.usage["prompt_tokens"], chat.usage["completion_tokens"]),
        "reference": "Franca et al. 2025, arXiv:2507.03761 (fusion rules and normalizations)",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name in splits:
        print(f"{name}: " + " ".join(f"{m}={accuracy(name, m):.3f}" for m in METHODS))
    print("dev-selected fusion rule:", selected_rule)
    for row in paired:
        if row["comparison"].endswith("minus dense"):
            print(f"{row['comparison']:34s} {row['accuracy_difference']:+.4f} "
                  f"[{row['bootstrap_95pct_low']:+.4f}, {row['bootstrap_95pct_high']:+.4f}]")
    print(f"new calls {chat.new_calls}, spend ${summary['spend_this_run_usd']:.3f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/longbench_v2_dense_fusion"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-new-calls", type=int, default=7000)
    parser.add_argument("--limit", type=int, help="first N dev and N test questions only (smoke test)")
    args = parser.parse_args()
    evaluate(args.output_dir, args.threads, args.max_new_calls, args.limit)


if __name__ == "__main__":
    main()
