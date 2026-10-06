"""Evaluate the improved graph construction against the original graph.

Follows the team Legal RAG protocol in legal_protocol.py: passage text only,
the published 20 dev / 80 held-out split, legal_rag metric files, and the
legal_rag timing harness. No embedding model, vector store, or LLM is loaded.

Methods, fixed before any held-out evaluation:

- ``original_graph``: the original keyword graph (``LEGACY_SETTINGS``).
- ``volume_links``: + query links weighted by IDF x concept edge volume.
- ``volume_heading_words``: + passages inherit their headings' words
  (consecutive-passage edges kept).
- ``heading_tree_only``: volume links + the section/heading tree, without
  heading words.
- ``improved_graph``: volume links + heading words + heading tree. Selected on
  development (the app default); ``improved_graph_one_step`` is its
  no-propagation control.
- ``improved_graph_word_forms``: + corpus word-form merging (rejected on
  development).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from legal_data import (
    LEGAL_DATASET_ID, legal_records_to_documents, legal_records_to_questions,
    load_legal_corpus, load_legal_questions,
)
from legal_protocol import KS, SPLIT_ID, load_reference_hits, split_queries, warm_latency, write_results
from standalone_graph import LEGACY_SETTINGS, StandaloneGraphRetriever, StandaloneGraphSettings


IMPROVED = StandaloneGraphSettings()
SELECTED = "improved_graph"
# method -> (config_id, settings, propagate)
METHODS = {
    "original_graph": ("graph_legacy_ppr_r0.35_s0.15", LEGACY_SETTINGS, True),
    "volume_links": ("graph_volume", replace(LEGACY_SETTINGS, query_links="volume"), True),
    "volume_heading_words": ("graph_volume_headwords_consecutive",
                             replace(LEGACY_SETTINGS, query_links="volume", heading_words=True), True),
    "heading_tree_only": ("graph_volume_tree",
                          replace(LEGACY_SETTINGS, query_links="volume", structure="headings"), True),
    SELECTED: ("graph_improved_ppr_r0.35_s0.15", IMPROVED, True),
    "improved_graph_one_step": ("graph_improved_onestep", IMPROVED, False),
    "improved_graph_word_forms": ("graph_improved_wordforms",
                                  replace(IMPROVED, word_forms="corpus"), True),
}
BASELINES = ("original_graph", "volume_links", "improved_graph_one_step")


def evaluate(corpus: list[dict], qa: list[dict], output_dir: Path,
             reference_per_query: Path | None = None) -> dict[str, Any]:
    dev_ids, test_ids = split_queries(qa, corpus)
    documents = legal_records_to_documents(corpus, text_only=True)
    questions = legal_records_to_questions(qa)
    graphs: dict[StandaloneGraphSettings, StandaloneGraphRetriever] = {}
    build_seconds: dict[str, float] = {}
    for name, (_, settings, _) in METHODS.items():
        if settings not in graphs:
            started = time.perf_counter()
            graphs[settings] = StandaloneGraphRetriever(documents, top_k=max(KS), settings=settings)
            build_seconds[name] = time.perf_counter() - started

    rankings: dict[str, dict[str, list[str]]] = {}
    traces = []
    for name, (_, settings, propagate) in METHODS.items():
        rankings[name] = {}
        for question in questions:
            result = graphs[settings].retrieve(question.question, propagate=propagate)
            rankings[name][question.question_id] = [item.node_id for item in result.explanations]
            if name == SELECTED:
                traces.append({
                    "question_id": question.question_id,
                    "split": "dev" if question.question_id in dev_ids else "test",
                    "question": question.question,
                    "gold_passage_id": question.relevant_passage_id,
                    "query_links": result.query_links,
                    "iterations": result.iterations,
                    "residual": result.residual,
                    "converged": result.converged,
                    "trace": [item.to_dict() for item in result.explanations[:10]],
                })
    for row in traces:
        row["original_graph_ranking"] = rankings["original_graph"][row["question_id"]][:10]

    ordered = sorted(questions, key=lambda q: int(q.question_id))
    dev = [q for q in ordered if q.question_id in dev_ids]
    test = [q for q in ordered if q.question_id in test_ids]
    latency = warm_latency({
        name: (lambda text, graph=graphs[settings], propagate=propagate:
               graph.retrieve(text, propagate=propagate))
        for name, (_, settings, propagate) in METHODS.items()
    }, dev, test)
    # legal_rag Top-100 diagnostic: the reranking ceiling of each candidate list.
    recall_100 = {
        name: statistics.fmean(
            q.relevant_passage_id in [item.node_id for item in
                                      graphs[METHODS[name][1]].retrieve(q.question, top_k=100).explanations]
            for q in test
        )
        for name in ("original_graph", SELECTED)
    }
    results = write_results(
        output_dir, questions=questions, dev_ids=dev_ids, test_ids=test_ids, rankings=rankings,
        config_ids={name: config for name, (config, _, _) in METHODS.items()}, latency=latency,
        baseline=BASELINES,
        references=load_reference_hits(reference_per_query) if reference_per_query else None,
    )

    corpus_payload = sorted((str(doc.metadata["passage_id"]), doc.page_content) for doc in documents)
    summary = {
        "dataset_id": LEGAL_DATASET_ID, "corpus_passages": len(documents),
        "corpus_representation": "passage text only (no titles, no appended footnotes)",
        "corpus_sha256": hashlib.sha256(json.dumps(corpus_payload, ensure_ascii=False).encode()).hexdigest(),
        "questions": len(questions), "dev_count": len(dev_ids), "test_count": len(test_ids),
        "split": f"{SPLIT_ID} (legal_rag, feature/person2-bm25)",
        "retriever": "standalone concept/passage/heading Personalized PageRank",
        "embedding_model": None, "dense_retrieval_used": False,
        "selected_method": SELECTED,
        "selection": ("Construction choices were made on the 20 development questions before any "
                      "held-out run. Heading words + volume links led on dev nDCG@5; the heading tree "
                      "(0.372) and consecutive edges (0.376) differed by one rank on one question, and "
                      "the tree was kept for its interpretable structure. Corpus word forms lowered dev "
                      "nDCG@5 (0.303) and were rejected."),
        "settings": {name: asdict(settings) for name, (_, settings, _) in METHODS.items()},
        "graphs": {name: graphs[METHODS[name][1]].stats for name in build_seconds},
        "build_seconds": build_seconds,
        "converged_queries": sum(row["converged"] for row in traces),
        "unmatched_queries": sum(not row["query_links"] for row in traces),
        "test_candidate_recall_at_100": recall_100,
        "reference_per_query": str(reference_per_query) if reference_per_query else None,
        "protocol": "Team legal_rag protocol; every method is a fixed configuration.",
        "timing": ("legal_rag harness: 4 BLAS threads, 3 dev warmups, 3 shuffled interleaved repeats "
                   "of held-out queries; top-20 retrieval including explanation construction; CPU."),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Held-out candidate Recall@100:", {k: round(v, 4) for k, v in recall_100.items()})
    return {**summary, **results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/legal_standalone_graph"))
    parser.add_argument("--reference-per-query", type=Path,
                        help="legal_rag results/initial/per_query.csv for paired BM25/dense/hybrid comparisons")
    args = parser.parse_args()
    evaluate(load_legal_corpus(), load_legal_questions(), args.output_dir, args.reference_per_query)


if __name__ == "__main__":
    main()
