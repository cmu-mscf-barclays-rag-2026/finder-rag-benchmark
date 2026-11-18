# Legal and LongBench framework comparison

Meeting 10 · 17 November 2026

This table brings the Legal results together with the teammate’s published LongBench results. It compares each method with its own dataset baseline. **Legal token-F1 and LongBench multiple-choice accuracy measure different outcomes; their absolute scores and deltas must not be ranked or averaged across datasets.**

| Dataset | Test n | Answer metric | Method | Score | Δ vs BM25 | Marginal 95% interval for Δ | USD / 1,000 | Setup USD |
|---|---:|---|---|---:|---:|---|---:|---:|
| Legal RAG Bench | 80 | Token-F1, lexical proxy | BM25 + reader | 0.2788 | +0.0000 | — | 0.4984 | 0.0000 |
| Legal RAG Bench | 80 | Token-F1, lexical proxy | MultiQuery + BM25 + reader | 0.3085 | +0.0297 | [+0.0039, +0.0574] | 0.5914 | 0.0000 |
| Legal RAG Bench | 80 | Token-F1, lexical proxy | BM25 + DSPy CoT | 0.2518 | -0.0271 | [-0.0618, +0.0065] | 0.5909 | 0.0647 |
| LongBench v2 | 402 | MC accuracy | BM25 + reader | 0.3557 | +0.0000 | — | 0.7425 | 0.0000 |
| LongBench v2 | 402 | MC accuracy | MultiQuery + BM25 + reader | 0.3085 | -0.0473 | [-0.0846, -0.0100] | 0.8374 | 0.0000 |
| LongBench v2 | 402 | MC accuracy | BM25 + DSPy CoT | 0.3458 | -0.0100 | [-0.0498, +0.0299] | 0.8343 | 0.3534 |

Intervals are marginal paired question-bootstrap intervals, not simultaneous or multiplicity-adjusted: 20,000 draws for Legal and 10,000 for LongBench. **None of the three Legal reader comparisons passes Holm3 correction.** Legal MultiQuery token-F1 versus BM25 has Holm3 p=0.09471, despite its unadjusted interval excluding zero. Legal optimized DSPy versus BM25 has p=0.25042; versus unoptimized CoT p=0.60423.

Legal has a separate primary retrieval finding: passage Hit@5 rises from 32.5% to 42.5% with MultiQuery, exact paired McNemar p=0.02148, exploratory. This is not an answer-accuracy result.

## What changed in each dataset

- Both use gpt-4o-mini; readers and MultiQuery generation use temperature 0. Native optimizer proposal sampling may differ. MultiQueryRetriever requests three rewrites plus the original query, with explicit RRF60 to retain a matched evidence allowance. DSPy changes the reader over unchanged BM25 evidence.
- Legal groups the 100 questions by their 95 labeled passages into optimizer train10 / validation10 / test80. It uses the original 4,876 passages, not 300-word rechunking. Readers receive at most Top8 / 2,400 document words. The benchmark was previously inspected, so this split isolates only current optimization.
- LongBench uses seed42 dev101 / test402, 300-word chunks, Top8 / 2,400 document words, and includes the four choices in the retrieval query. It uses official reader prompting except for DSPy. The selected DSPy instruction equals the default; optimization did not produce a better selected instruction.
- Legal F1 measures lexical overlap with a free-text reference; it does not validate legal correctness. LongBench accuracy measures whether the selected choice matches the reference label. Evidence granularity, task format and prompts differ.
- Legal cost estimates model fresh test inference and include test-only MultiQuery generation; setup is the newly incurred optimizer-call estimate. LongBench costs are derived from the teammate’s saved usage, also including rewrites. Development generation and one-time optimization are excluded from per-query inference costs. These are estimates, not billing statements.

## Sources and reproduction

[Legal report and validation](../meeting10_legal_20261117/README.md) · [Pinned teammate report](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/blob/8738b543b8770a29acb5d82c481b1a67fe530140/GRAPH_RAG_README.md#longbench-v2-with-gpt-4o-mini-multiqueryretriever-and-dspy) · [Provenance](provenance.json) · [Six-row answer table](team_primary_comparison.csv) · [All 43 metric rows](team_all_metrics_long.csv)

Legal results were independently validated locally. LongBench values were read from the teammate’s published CSV/JSON and checked against the pinned GitHub blob hashes; their predictions were not independently rerun. The source aggregates and download receipt are retained in `sources/longbench/`. No pooled metric, cross-dataset significance test or new model call was produced.

Run from the repository root after the two source packages are available:

```bash
../work/meeting10_venv/bin/python person3_meeting10_legal/combine_team.py
```
