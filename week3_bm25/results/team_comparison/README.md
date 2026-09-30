# Week 3 — FinDER team comparison

This is the **Week 3** team comparison. Florence contributed the BM25 retriever
and the controlled timing study; the other methods are teammate submissions.
For Florence's own work, see [the Week 3 summary](../../README.md).
Source commits and export provenance are recorded in
[provenance.json](../../shared_reference/team_metrics/provenance.json).

All included rows passed the shared benchmark identifier checks for dataset,
corpus, and split. This is a comparison of the included submissions, not a claim
that unreported metrics were measured. Blank metrics are unreported, not zero.
Florence's submission uses the same shared split as the other methods; her older
seed-42 experiments are excluded.

## Primary comparison at K=5

| Contributor | Method | Test questions | Recall@5 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| Florence | Florence BM25 k1=1.6, b=1 | 4,575 | 31.12% | 0.2340 | 0.2497 |
| Cheryl | Hybrid RRF alpha=0.25 | 4,575 | 30.56% | 0.2205 | 0.2386 |
| Yuchen | Dense MiniLM, chunk 500 / overlap 75 | 4,575 | 22.84% | Not reported | Not reported |
| Kevin | MMR (lambda 1.0, 10 candidates) | 4,575 | 21.18% | 0.1583 | 0.1689 |
| Kevin | Similarity threshold 0.3 | 4,575 | 21.17% | 0.1583 | 0.1688 |

Among the included submissions, the highest Recall@5 is **Florence BM25 k1=1.6, b=1** at **0.3112**.
The same row has nDCG@5 of **0.2497**.
This is a descriptive comparison, not a claim of statistical significance.

## Controlled query latency

[Controlled latency comparison](latency_comparison.csv): all selected methods were timed on one Mac CPU with the same 100 held-out questions, 3 warmups, and 5 interleaved repetitions (500 measurements per method). Index preparation is excluded. See the [timing report](../common_latency/report.md) for scope, hardware, model revision, and limitations. The original mixed-hardware timings remain only in the combined CSV for provenance.

## Answer evaluation

Answer scores are not yet comparable. No answer table is published until every method is marked final under one shared protocol.
