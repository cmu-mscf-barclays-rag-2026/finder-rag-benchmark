"""Classify FinDER MultiQuery misses using saved Week 11 rankings and Week 10 searches."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from week3_bm25.finder_bm25.bm25 import BM25Retriever
from week3_bm25.finder_bm25.data import load_bundle, read_jsonl
from week10_followup.finder_query_expansion import rrf

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "week3_bm25/data/team_shared"
RANKINGS = ROOT / "week11_followup/results/finder_backends/rankings.jsonl"
TRACES = ROOT / "week10_followup/results/finder_gpt200/multiquery/traces.jsonl"
SELECTED = ROOT / "week3_bm25/results/b_bm25_common/selected_config.json"
OUTPUT = ROOT / "week11_followup/results/finder_backends"


def first_gold(ids: list[str], gold: set[str]) -> int | None:
    return next((rank for rank, doc_id in enumerate(ids, 1) if doc_id in gold), None)


def change(before: bool, after: bool) -> str:
    return "gain" if after and not before else "loss" if before and not after else "same"


def main() -> None:
    bundle = load_bundle(BUNDLE)
    queries = {q["query_id"]: q for q in bundle["queries"]}
    rankings = read_jsonl(RANKINGS)
    traces = {t["query_id"]: t for t in read_jsonl(TRACES)}
    if len(rankings) != 200 or {r["query_id"] for r in rankings} != set(traces):
        raise ValueError("Expected the same 200 questions in rankings and saved traces")
    selected = json.loads(SELECTED.read_text())
    if selected["bundle_fingerprint"] != bundle["manifest"]["fingerprint"]:
        raise ValueError("Tuned BM25 configuration does not match the FinDER bundle")
    index = BM25Retriever(bundle["corpus"], k1=selected["k1"], b=selected["b"])

    rows = []
    counts = Counter()
    for entry in rankings:
        qid = entry["query_id"]
        q = queries[qid]
        gold = set(bundle["qrels"][qid])
        rank = entry["rankings"]
        trace = traces[qid]
        searches = trace["searches"]
        if searches[-1]["query"] != q["text"]:
            raise ValueError(f"Original search does not match the question: {qid}")
        all_bm25_candidates = set().union(*(set(s["doc_ids"]) for s in searches))
        deeper_searches = []
        for search in searches:
            ids = [hit["doc_id"] for hit in index.search(search["query"], 100)]
            if ids[:20] != search["doc_ids"]:
                raise ValueError(f"Depth-100 replay differs from saved Top-20: {qid}")
            deeper_searches.append({"doc_ids": ids})
        deeper_candidates = set().union(*(set(s["doc_ids"]) for s in deeper_searches))
        deeper_fused, _ = rrf(deeper_searches)
        backend_top20 = set(rank["bm25_multiquery"]) | set(rank["dense_multiquery"])
        original = first_gold(rank["bm25_original"][:10], gold) is not None
        bm25 = first_gold(rank["bm25_multiquery"][:10], gold) is not None
        dense = first_gold(rank["dense_multiquery"][:10], gold) is not None
        hybrid = first_gold(rank["hybrid_multiquery"][:10], gold) is not None
        deeper_hit10 = first_gold(deeper_fused[:10], gold) is not None
        bm25_failure = ("hit" if bm25 else "ranking_or_fusion" if gold & all_bm25_candidates
                        else "candidate_recall")
        hybrid_failure = ("hit" if hybrid else "ranking_within_backend_top20"
                          if gold & backend_top20 else "absent_from_backend_top20")
        bm25_change = change(original, bm25)
        hybrid_change = change(bm25, hybrid)
        counts[f"bm25_{bm25_failure}"] += 1
        counts[f"hybrid_{hybrid_failure}"] += 1
        counts[f"bm25_change_{bm25_change}"] += 1
        counts[f"hybrid_vs_bm25_{hybrid_change}"] += 1
        counts["bm25_depth100_candidate_hit" if gold & deeper_candidates else
               "bm25_depth100_candidate_miss"] += 1
        counts["bm25_depth100_hit5"] += int(first_gold(deeper_fused[:5], gold) is not None)
        counts["bm25_depth100_hit10"] += int(deeper_hit10)
        counts[f"bm25_depth100_vs_depth20_{change(bm25, deeper_hit10)}"] += 1
        if not gold & all_bm25_candidates:
            counts["bm25_depth20_miss_recovered_at100" if gold & deeper_candidates
                   else "bm25_depth20_miss_still_absent_at100"] += 1
        if bm25_change == "gain":
            counts["bm25_gain_original_top20_present" if gold & set(searches[-1]["doc_ids"])
                   else "bm25_gain_new_from_rewrites"] += 1
        if bm25_change == "loss":
            counts["bm25_loss_still_top20" if gold & set(rank["bm25_multiquery"])
                   else "bm25_loss_below_top20"] += 1
        if hybrid_change == "loss":
            counts["hybrid_loss_still_top20" if gold & set(rank["hybrid_multiquery"])
                   else "hybrid_loss_below_top20"] += 1
        rows.append({
            "query_id": qid, "category": q["category"], "question": q["text"],
            "rewrite_1": trace["generated_queries"][0] if trace["generated_queries"] else "",
            "rewrite_2": trace["generated_queries"][1] if len(trace["generated_queries"]) > 1 else "",
            "gold_count": len(gold), "original_bm25_gold_rank": first_gold(rank["bm25_original"], gold),
            "bm25_multiquery_gold_rank": first_gold(rank["bm25_multiquery"], gold),
            "dense_multiquery_gold_rank": first_gold(rank["dense_multiquery"], gold),
            "hybrid_multiquery_gold_rank": first_gold(rank["hybrid_multiquery"], gold),
            "bm25_depth100_multiquery_gold_rank": first_gold(deeper_fused, gold),
            "bm25_gold_in_three_top20_searches": bool(gold & all_bm25_candidates),
            "bm25_gold_in_three_top100_searches": bool(gold & deeper_candidates),
            "bm25_failure_type": bm25_failure, "hybrid_failure_type": hybrid_failure,
            "bm25_change": bm25_change, "hybrid_change_vs_bm25": hybrid_change,
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "error_analysis.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "error_analysis_counts.json").write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n")
    lines = ["# FinDER error analysis", "",
             "The same 200 held-out questions, pooled reference-passage labels, and saved Week 10 GPT rewrites are used. "
             "`error_analysis.csv` contains the question, rewrites, first judged-reference rank, and failure class for each question.", "",
             "## Where BM25 MultiQuery misses", "",
             f"At Top-10 it hits {counts['bm25_hit']}/200 questions and misses {200-counts['bm25_hit']}/200. "
             f"For {counts['bm25_candidate_recall']} misses, **none** of the original or two rewritten BM25 searches returns a judged reference in its Top-20. "
             f"For the other {counts['bm25_ranking_or_fusion']} misses, a judged reference enters at least one Top-20 candidate list but does not reach the fused Top-10. "
             "The first group needs better candidate retrieval; the second can potentially benefit from ranking or fusion changes.", "",
             "## What happens at candidate depth 100", "",
             "The tuned BM25 index replays every saved Top-20 source ranking exactly. Retrieving 100 instead of 20 from each of the same three formulations raises candidate-level question coverage "
             f"from {200-counts['bm25_candidate_recall']}/200 to {counts['bm25_depth100_candidate_hit']}/200. "
             f"Of the {counts['bm25_candidate_recall']} questions with no judged reference in any Top-20 source list, "
             f"{counts['bm25_depth20_miss_recovered_at100']} gain a candidate by depth 100 and {counts['bm25_depth20_miss_still_absent_at100']} still have none. "
             f"But applying the same equal-weight RRF to all Top-100 lists yields {counts['bm25_depth100_hit10']}/200 Hit@10 versus {counts['bm25_hit']}/200 at source depth 20: "
             f"{counts['bm25_depth100_vs_depth20_gain']} gains and {counts['bm25_depth100_vs_depth20_loss']} losses. "
             "Candidate recall improves, but the extra candidates do not improve the final Top-10 under this fusion rule.", "",
             "## What the rewrites change", "",
             f"Against the original BM25 Top-10, MultiQuery gains {counts['bm25_change_gain']} question hits and loses {counts['bm25_change_loss']}. "
             f"In {counts['bm25_gain_original_top20_present']} of the gains, the original BM25 Top-20 already contained a judged reference, so the rewrites helped promote evidence. "
             f"The other {counts['bm25_gain_new_from_rewrites']} gains brought judged evidence absent from the original Top-20. "
             f"Of the {counts['bm25_change_loss']} losses, {counts['bm25_loss_still_top20']} still have a judged reference in the fused Top-20, showing a ranking problem.", "",
             "Two examples for manual review: question `632f159a` uses the term *input metrics* and the judged passage says *input method*; its rewrites use *advancement* and *status*, moving the reference from rank 9 to 18. "
             "Question `a0adb935` uses *Dec 23*, which a rewrite renders as *December 23* even though the judged reference is a fourth-quarter 2023 table; that may alter the intended date. The reference moves from rank 5 to 17. "
             "These examples suggest mechanisms; they do not establish a general causal pattern.", "",
             "## What hybrid adds", "",
             f"Hybrid MultiQuery gains {counts['hybrid_vs_bm25_gain']} Top-10 hits relative to BM25 MultiQuery and loses {counts['hybrid_vs_bm25_loss']}. "
             f"All {counts['hybrid_loss_still_top20']} losses retain a judged reference within hybrid's Top-20, so their failure is rank placement. "
             f"Among its {200-counts['hybrid_hit']} Top-10 misses, {counts['hybrid_ranking_within_backend_top20']} have a judged reference in the union of the fused BM25 and dense Top-20 lists; "
             f"{counts['hybrid_absent_from_backend_top20']} do not. This is a diagnostic of those two backend lists, not every original per-formulation candidate.", "",
             "## Next controlled tests", "",
             "1. Audit a small sample from each failure class in `error_analysis.csv`, including whether the benchmark reference supports the question. Record abbreviation, entity, date, and financial-term drift in the rewrites.",
             "2. On a development split, test a fusion or reranking rule that gives the original query more weight or protects its high-ranked evidence. Check whether it can promote the newly found depth-100 candidates without losing current Top-10 hits. Evaluate the frozen rule on a fresh test split and compare Top-5 with Top-10.",
             "3. For references still absent from all three Top-100 searches, test a stronger dense encoder or domain-aware reformulations. A fusion adjustment alone cannot recover references absent from every source list.", "",
             "The depth-100 comparison uses the same 200 test questions and is exploratory; choose any new ranking rule on development data and validate it on fresh held-out questions. "
             "These are retrieval labels only. They do not establish whether the passages fully answer the question.", ""]
    (OUTPUT / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} question diagnostics to {OUTPUT}")


if __name__ == "__main__":
    main()
