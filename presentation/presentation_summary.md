# FinDER retrieval comparison

This table covers the compatible CSV submissions currently in `team_metrics/`.
See [the all-four team overview](team_overview.md) for every contribution,
including the historical experiments. Source commits are recorded in
[provenance.json](../team_metrics/provenance.json).

All included rows passed the shared benchmark identifier checks for dataset,
corpus, and split. This is a comparison of the included submissions, not a claim
that unreported metrics were measured. Blank metrics are unreported, not zero.
Florence's current B submission uses the same shared split as A/C/D; her older
seed-42 experiments are excluded.

## Primary comparison at K=5

| owner | method | n_queries | precision_at_k | recall_at_k | hit_rate_at_k | mrr_at_k | ndcg_at_k | answer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | Florence BM25 k1=1.6, b=1 | 4575.0000 | 0.0655 | 0.3112 | 0.3211 | 0.2340 | 0.2497 | not_run |
| C | Hybrid RRF alpha=0.25 | 4575.0000 | 0.0643 | 0.3056 | 0.3145 | 0.2205 | 0.2386 | pilot |
| A | Dense MiniLM, chunk 500 / overlap 75 | 4575.0000 | 0.0479 | 0.2284 | 0.2341 |  |  | not_run |
| D | d_mmr_1.0_10 | 4575.0000 | 0.0451 | 0.2118 | 0.2195 | 0.1583 | 0.1689 | not_run |
| D | d_threshold_0.3 | 4575.0000 | 0.0450 | 0.2117 | 0.2195 | 0.1583 | 0.1688 | not_run |

Among the included submissions, the highest Recall@5 is **Florence BM25 k1=1.6, b=1** at **0.3112**.
The same row has nDCG@5 of **0.2497**.

## Controlled query latency

[Controlled latency comparison](latency_comparison.csv): all selected methods were timed on one Mac CPU with the same 100 held-out questions, 3 warmups, and 5 interleaved repetitions (500 measurements per method). Index preparation is excluded. See the [timing report](../results/common_latency/report.md) for scope, hardware, model revision, and limitations. The original mixed-hardware timings remain only in the combined CSV for provenance.

Answer scores are not yet comparable. The answer table stays empty until every method is marked final under one shared protocol.
