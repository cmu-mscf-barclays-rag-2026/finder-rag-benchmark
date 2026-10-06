"""Evaluate LLM-augmented graph retrieval under the team Legal RAG protocol.

Uses legal_protocol.py: passage text only, the published 20 dev / 80 held-out
split, legal_rag metric files, and paired bootstrap intervals. Every method
is a fixed configuration, so nothing is selected on development:

- ``standalone_graph``: the graph walk alone (baseline). This is the improved
  graph; ``--legacy-graph`` reruns everything on the original graph.
- ``llm_keywords_concat`` and ``llm_keywords_mixed``: LLM keyword expansion.
  The phrases are either appended to the question, or share the walk's seeds
  with the question keeping a fixed 50%.
- ``llm_rerank_graph``: the LLM reranks the graph's top 20 (the app's
  ``graph_llm`` mode).
- ``llm_expand_rerank``: the LLM reranks the union of the question walk's and
  the 50%-mixed expanded walk's top 20.

Metrics use cached LLM outputs, so reruns need Ollama only for new calls. The
graph is timed with the legal_rag CPU harness. ``--live-latency`` also times
the LLM methods with the cache bypassed: one shuffled pass over held-out
questions, with Ollama on the local GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from langchain_ollama import ChatOllama

from legal_data import (
    LEGAL_DATASET_ID, legal_records_to_documents, legal_records_to_questions,
    load_legal_corpus, load_legal_questions,
)
from legal_protocol import KS, SPLIT_ID, load_reference_hits, split_queries, warm_latency, write_results
from llm_graph import (
    EXPANSION_PROMPT, EXPANSION_PROMPT_VERSION, RERANK_PROMPT_VERSION, RERANK_SYSTEM,
    LLMExpandedGraphRetriever, LLMRerankedGraphRetriever, OllamaRelevanceScorer,
    expansion_chain, rerank_messages,
)
from standalone_graph import LEGACY_SETTINGS, StandaloneGraphRetriever, StandaloneGraphSettings


BASELINE = "standalone_graph"
QUERY_WEIGHT = 0.5  # fixed a priori: equal seed mass for question and expansion
POOL_DEPTH = 20  # per-walk candidates judged by the LLM reranker
CONFIG_IDS = {
    BASELINE: "graph_improved_ppr_r0.35_s0.15",
    "llm_keywords_concat": "graph_llmkw_concat",
    "llm_keywords_mixed": "graph_llmkw_q0.5",
    "llm_rerank_graph": "graph_llmrerank_pool20",
    "llm_expand_rerank": "graph_llmkw_q0.5_llmrerank_pool20",
}


class LLMCache:
    """Append-only JSONL cache of LLM outputs keyed by task, model, and exact input."""

    def __init__(self, path: Path, model: str) -> None:
        self.path, self.model = path, model
        self.entries: dict[str, Any] = {}
        self.new_calls: dict[str, int] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.entries[row["key"]] = row["output"]

    def key(self, task: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{task}|{self.model}|{digest}"

    def get(self, task: str, text: str, compute: Callable[[], Any], label: str | None = None) -> Any:
        """``text`` is the exact model input; ``label`` is a short readable record of it."""
        key = self.key(task, text)
        if key not in self.entries:
            self.new_calls[task] = self.new_calls.get(task, 0) + 1
            self.entries[key] = compute()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "input": label or text,
                                         "output": self.entries[key]}, ensure_ascii=False) + "\n")
        return self.entries[key]


def build_methods(
    graph: StandaloneGraphRetriever,
    expand: Callable[[str], str],
    score: Callable[[str, Any], float],
) -> dict[str, Callable[..., Any]]:
    expanded = LLMExpandedGraphRetriever(graph, expand, query_weight=QUERY_WEIGHT)
    return {
        BASELINE: graph.retrieve,
        "llm_keywords_concat": LLMExpandedGraphRetriever(graph, expand).retrieve,
        "llm_keywords_mixed": expanded.retrieve,
        "llm_rerank_graph": LLMRerankedGraphRetriever(graph, score, pool_depth=POOL_DEPTH).retrieve,
        "llm_expand_rerank": LLMRerankedGraphRetriever(
            graph, score, expanded=expanded, pool_depth=POOL_DEPTH).retrieve,
    }


def evaluate(corpus: list[dict], qa: list[dict], output_dir: Path, model: str, base_url: str,
             reference_per_query: Path | None = None, live_latency: bool = False,
             graph_settings: StandaloneGraphSettings | None = None) -> dict[str, Any]:
    dev_ids, test_ids = split_queries(qa, corpus)
    documents = legal_records_to_documents(corpus, text_only=True)
    questions = legal_records_to_questions(qa)
    graph = StandaloneGraphRetriever(documents, top_k=max(KS), settings=graph_settings)
    config_ids = dict(CONFIG_IDS)
    if graph.settings == LEGACY_SETTINGS:
        config_ids = {name: config.replace("graph_improved_", "graph_")
                      for name, config in config_ids.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = LLMCache(output_dir / "llm_cache.jsonl", model)
    chain = expansion_chain(ChatOllama(model=model, base_url=base_url, temperature=0, num_predict=256))
    scorer = OllamaRelevanceScorer(model, base_url)

    def live_expand(question: str) -> str:
        return chain.invoke({"question": question})

    def cached_expand(question: str) -> str:
        return cache.get(f"expand:{EXPANSION_PROMPT_VERSION}", question, lambda: live_expand(question))

    def cached_score(question: str, document: Any) -> float:
        return cache.get(
            f"rerank:{RERANK_PROMPT_VERSION}",
            json.dumps(rerank_messages(question, document), ensure_ascii=False),
            lambda: scorer(question, document),
            label=json.dumps([question, document.metadata["passage_id"]], ensure_ascii=False),
        )

    methods = build_methods(graph, cached_expand, cached_score)
    rankings: dict[str, dict[str, list[str]]] = {}
    traces: dict[str, dict[str, Any]] = {
        q.question_id: {"question_id": q.question_id,
                        "split": "dev" if q.question_id in dev_ids else "test",
                        "question": q.question, "gold_passage_id": q.relevant_passage_id}
        for q in questions
    }
    for name, retrieve in methods.items():
        rankings[name] = {}
        for question in questions:
            result = retrieve(question.question, top_k=max(KS))
            ranking = [item.node_id for item in result.explanations]
            rankings[name][question.question_id] = ranking
            gold = question.relevant_passage_id
            trace = {
                "gold_rank": ranking.index(gold) + 1 if gold in ranking else None,
                "top5": ranking[:5],
                "seed_concepts": sorted(result.query_links, key=result.query_links.get,
                                        reverse=True)[:15],
            }
            if result.explanations and hasattr(result.explanations[0], "llm_relevance"):
                trace["top5_llm_relevance"] = [item.llm_relevance for item in result.explanations[:5]]
            traces[question.question_id][name] = trace
        print(f"{name}: new LLM calls so far {cache.new_calls}", flush=True)
    for question in questions:
        traces[question.question_id]["keyword_expansion"] = cache.entries.get(
            cache.key(f"expand:{EXPANSION_PROMPT_VERSION}", question.question))

    ordered = sorted(questions, key=lambda q: int(q.question_id))
    dev = [q for q in ordered if q.question_id in dev_ids]
    test = [q for q in ordered if q.question_id in test_ids]
    latency = warm_latency({BASELINE: graph.retrieve}, dev, test)
    if live_latency:
        live = build_methods(graph, live_expand, scorer)
        latency.update(warm_latency({name: live[name] for name in methods if name != BASELINE},
                                    dev, test, repeats=1))
    results = write_results(
        output_dir, questions=questions, dev_ids=dev_ids, test_ids=test_ids, rankings=rankings,
        config_ids=config_ids, latency=latency, baseline=BASELINE,
        references=load_reference_hits(reference_per_query) if reference_per_query else None,
    )
    summary = {
        "dataset_id": LEGAL_DATASET_ID, "corpus_passages": len(documents),
        "corpus_representation": "passage text only (no titles, no appended footnotes)",
        "questions": len(questions), "dev_count": len(dev_ids), "test_count": len(test_ids),
        "split": f"{SPLIT_ID} (legal_rag, feature/person2-bm25)",
        "llm_model": model, "configs": config_ids,
        "expansion_prompt": {"version": EXPANSION_PROMPT_VERSION,
                             "messages": [m.prompt.template for m in EXPANSION_PROMPT.messages],
                             "query_weight_mixed": QUERY_WEIGHT},
        "rerank": {"version": RERANK_PROMPT_VERSION, "system": RERANK_SYSTEM,
                   "score": "P(Yes) over Yes/No in first-token top-10 logprobs",
                   "pool_depth_per_walk": POOL_DEPTH, "tie_break": "reciprocal-rank fusion, k=60"},
        "graph_settings": asdict(graph.settings), "graph": graph.stats,
        "new_llm_calls_this_run": cache.new_calls,
        "timing": {
            BASELINE: "legal_rag harness: 4 BLAS threads, 3 dev warmups, 3 shuffled interleaved "
                      "repeats of held-out queries, top-20 retrieval, CPU",
            "llm_methods": ("live, cache bypassed: 3 dev warmups, 1 shuffled interleaved pass of "
                            "held-out queries; llama3.2 via Ollama on an RTX 4060 laptop GPU")
                           if live_latency else "not timed in this run (cached LLM outputs)",
        },
        "reference_per_query": str(reference_per_query) if reference_per_query else None,
        "protocol": ("Team legal_rag protocol; fixed configurations, no development selection. "
                     "The LLM variants were designed earlier from failure analysis on a different "
                     "50-question half that overlaps this held-out set, so their held-out "
                     "numbers are somewhat optimistic."),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {**summary, **results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--output-dir", type=Path, default=Path("results/legal_llm_graph"))
    parser.add_argument("--reference-per-query", type=Path,
                        help="legal_rag results/initial/per_query.csv for paired BM25/dense/hybrid comparisons")
    parser.add_argument("--live-latency", action="store_true",
                        help="also time LLM methods live with the cache bypassed (needs Ollama)")
    parser.add_argument("--legacy-graph", action="store_true",
                        help="run on the original keyword graph instead of the improved graph")
    args = parser.parse_args()
    evaluate(load_legal_corpus(), load_legal_questions(), args.output_dir, args.model,
             args.base_url, args.reference_per_query, args.live_latency,
             LEGACY_SETTINGS if args.legacy_graph else None)


if __name__ == "__main__":
    main()
