# Legal RAG Bench — held-out retrieval results

20 development questions; 80 held-out questions. Configurations selected on development nDCG@5.

| Method | Recall@5 | Hit@5 | All evidence hit@5 | Gold-text coverage@5 | MRR@5 | nDCG@5 | Mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.3750 | 0.3750 | 0.3750 | 0.3750 | 0.2540 | 0.2843 | 11.25 |
| dense | 0.2875 | 0.2875 | 0.2875 | 0.1311 | 0.1844 | 0.2109 | 13.74 |
| hybrid | 0.4000 | 0.4000 | 0.4000 | 0.1911 | 0.2712 | 0.3037 | 44.71 |

Recall, Hit and All Evidence Hit coincide because there is one gold passage per question.
Coverage is union character coverage of the gold passage, NOT semantic evidence completeness.
K counts chunks, including repeated parents. Rank credit is awarded only at the first gold-parent occurrence.
Each method selects its own chunk size; context lengths differ. See selected.json and generation_inputs.jsonl.
Latency is warm CPU top-100 candidate retrieval, not generation or separate per-K latency.
Original-passage dense embeddings can truncate; inspect manifest.json before interpreting comparisons.
Answer correctness and groundedness: NOT EVALUATED. No answer model has been selected.
This is a small internal split, not the official full-100 benchmark score.

## Interpretation of this run

- bm25: 30/80 labeled-parent hits at K=5; selected configuration `bm25_s0_k0.8_b1.0`.
- dense: 23/80 labeled-parent hits at K=5; selected configuration `dense_s128`.
- hybrid: 32/80 labeled-parent hits at K=5; selected configuration `hybrid_s128_w0.25`.

Hybrid rescues 9 BM25 misses and loses 7 BM25 hits; all three miss 41 questions.
The hybrid-minus-BM25 Hit@5 difference is 2.5%; its exploratory paired gold-passage-group bootstrap interval is [-7.4%, 12.3%]. This does not establish a clear hybrid advantage.
BM25 returns whole passages while the selected dense and hybrid configurations return smaller chunks. Gold-text coverage and context size therefore matter alongside the hit count.
See question_comparison.csv for concrete rescues/regressions and paired_hit5_intervals.csv for uncertainty estimates.
