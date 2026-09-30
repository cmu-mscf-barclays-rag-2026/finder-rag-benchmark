# Week 5 — Florence Liu: hierarchical retrieval results

Same 80 test questions from Kevin’s handoff; all settings selected on its 20 development questions.
Shared BM25: k1=1.2, b=0.75. Selected hybrid BM25 weight: 0.75.

Hit means a complete original gold child is present in the delivered context. Coverage measures gold-body tokens, not answer correctness.

## Context budget: unbounded

| Method | Mode | Hit@1 | Hit@5 | Hit@10 | Coverage@10 | Mean tokens@10 | P95 tokens@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | child | 11/80 | 23/80 | 29/80 | 36.2% | 3823 | 4628 |
| bm25 | parent | 15/80 | 31/80 | 37/80 | 46.2% | 10569 | 19027 |
| bm25 | child_to_parent | 13/80 | 26/80 | 32/80 | 40.0% | 8956 | 15774 |
| hybrid_w0.75 | child | 12/80 | 24/80 | 33/80 | 41.2% | 3719 | 4494 |
| hybrid_w0.75 | parent | 16/80 | 32/80 | 49/80 | 61.3% | 9677 | 15217 |
| hybrid_w0.75 | child_to_parent | 14/80 | 29/80 | 40/80 | 50.0% | 8757 | 16028 |

## Context budget: 2048

| Method | Mode | Hit@1 | Hit@5 | Hit@10 | Coverage@10 | Mean tokens@10 | P95 tokens@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | child | 11/80 | 22/80 | 22/80 | 28.2% | 2039 | 2048 |
| bm25 | parent | 14/80 | 21/80 | 21/80 | 28.1% | 2039 | 2048 |
| bm25 | child_to_parent | 11/80 | 21/80 | 21/80 | 26.3% | 2039 | 2048 |
| hybrid_w0.75 | child | 12/80 | 23/80 | 23/80 | 30.0% | 2045 | 2048 |
| hybrid_w0.75 | parent | 15/80 | 22/80 | 22/80 | 27.6% | 2048 | 2048 |
| hybrid_w0.75 | child_to_parent | 13/80 | 20/80 | 20/80 | 25.6% | 2046 | 2048 |

## Interpretation limits

- K counts child seeds for expansion and parent results for direct parent retrieval. Expansion reuses exactly the child ranking and does not backfill.
- Unbounded expansion can recover evidence through a sibling. It cannot improve the original child ranking.
- MiniLM truncates encoder inputs at 256 tokens, separately from the final 2,048-token context cap. See index_audit.csv.
- The split differs from Florence’s initial 20/80 run (63 shared test questions). Three gold parents span Kevin’s dev/test split. This is exploratory query-held-out evaluation, not parent-disjoint validation.
- The original 44 → 56 configuration is unverified; this controlled experiment is not a reproduction claim.
- summary.csv also reports dense and all three hybrid weights. Test scores do not select the weight.
- No answer generation, correctness judging, or latency benchmark was performed.
