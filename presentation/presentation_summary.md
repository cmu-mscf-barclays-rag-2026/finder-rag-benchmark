# FinDER retrieval comparison

This table covers the compatible CSV submissions currently in `team_metrics/`.
See [the all-four team overview](team_overview.md) for every contribution,
including results awaiting a compatible export. Source commits are recorded in
[provenance.json](../team_metrics/provenance.json).

All included rows passed the shared benchmark identifier checks for dataset,
corpus, and split. This is a comparison of the included submissions, not a claim
that every team member has a compatible result. Blank metrics are unreported, not zero.
Latency values use different hardware/protocols and must not be ranked as speed.

## Primary comparison at K=5

| owner | method | n_queries | precision_at_k | recall_at_k | hit_rate_at_k | mrr_at_k | ndcg_at_k | latency_ms_per_query | answer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | Hybrid RRF alpha=0.25 | 4575.0000 | 0.0643 | 0.3056 | 0.3145 | 0.2205 | 0.2386 | 3.8719 | pilot |
| A | Dense MiniLM, chunk 500 / overlap 75 | 4575.0000 | 0.0479 | 0.2284 | 0.2341 |  |  | 0.6634 | not_run |
| D | d_mmr_1.0_10 | 4575.0000 | 0.0451 | 0.2118 | 0.2195 | 0.1583 | 0.1689 | 6.8894 | not_run |
| D | d_threshold_0.3 | 4575.0000 | 0.0450 | 0.2117 | 0.2195 | 0.1583 | 0.1688 | 6.7480 | not_run |

Among the included submissions, the highest Recall@5 is **Hybrid RRF alpha=0.25** at **0.3056**.
The same row has nDCG@5 of **0.2386** and mean retrieval
latency of **3.87 ms/query** on the contributor's hardware.

Answer scores are not yet comparable. The answer table stays empty until every method is marked final under one shared protocol.
