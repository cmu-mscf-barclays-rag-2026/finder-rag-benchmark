# FinDER retrieval comparison

All rows passed the shared benchmark check for dataset, corpus, and split.

## Primary comparison at K=5

| owner | method | n_queries | precision_at_k | recall_at_k | hit_rate_at_k | mrr_at_k | ndcg_at_k | latency_ms_per_query | answer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | Hybrid RRF alpha=0.25 | 4575.0000 | 0.0643 | 0.3056 | 0.3145 | 0.2205 | 0.2386 | 3.8719 | pilot |

The highest Recall@5 is **Hybrid RRF alpha=0.25** at **0.3056**.
The same row has nDCG@5 of **0.2386** and mean retrieval
latency of **3.87 ms/query** on the contributor's hardware.

Answer scores are not yet comparable. The answer table stays empty until every method is marked final under one shared protocol.
