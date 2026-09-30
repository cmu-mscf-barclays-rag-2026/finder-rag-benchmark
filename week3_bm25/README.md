# Week 3 — BM25 retrieval on FinDER

**Question:** How well does a keyword-based retriever find the relevant financial passages?

Florence's BM25 implementation achieved **31.12% Recall@5** on the common team
split, compared with **30.22%** for the fixed default BM25 configuration.
Parameters were chosen using 1,128 development questions and then evaluated on
4,575 test questions over 5,830 reference passages.

| Selected BM25 metric | Result |
|---|---:|
| Recall@5 | 31.12% |
| MRR@5 | 0.2340 |
| nDCG@5 | 0.2497 |

This is retrieval over the benchmark's pooled reference passages, not all original
company filings. The result does not measure generated-answer correctness.

## Results to review

- [Florence's results](results/b_bm25_common/report.md) — configurations and metrics at each cutoff.
- [Week 3 team comparison](results/team_comparison/README.md) — matched results from all four contributors.
- [Controlled timing study](results/common_latency/report.md) — the same queries and CPU timing protocol.
- [Detailed analysis](ANALYSIS.md) — earlier experiments and failure patterns, labeled separately from the common-split run.

## Code and reproduction

[Reproduction guide](REPRODUCING.md) · [Methodology](METHODOLOGY.md) ·
[Walkthrough notebook](bm25_walkthrough.ipynb)

`finder_bm25/` contains Florence's retriever and evaluation code.
`run_shared_benchmark.py` runs the shared-split experiment; `common_latency.py`
measures the controlled timing comparison. `results/` contains the saved outputs.
`shared_reference/` holds the inherited team baseline, source data, contracts and
export utilities needed to reproduce the comparison.
