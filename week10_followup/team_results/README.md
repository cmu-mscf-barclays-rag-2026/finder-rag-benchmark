# Team result handoff

Place a JSON file here for each teammate's completed method. The aggregator
reads every `*.json` in this directory. Example:

```json
{
  "schema": "week10_retrieval_summary_v1",
  "rows": [{
    "dataset": "ViDoRe_finance_en",
    "corpus": "corpus SHA-256 or stable corpus ID",
    "split": "test_english",
    "query_set_fingerprint": "stable SHA-256 of the evaluated query IDs",
    "query_count": 309,
    "retrieval_unit": "page",
    "method": "method name and configuration",
    "scope": "ranked",
    "k": 5,
    "budget": "",
    "hit": 0.0,
    "recall": 0.0,
    "mrr": 0.0,
    "ndcg": 0.0,
    "mean_delivered_units": "",
    "mean_tokens": "",
    "latency_ms": "",
    "model_calls": "",
    "input_tokens": "",
    "output_tokens": ""
  }]
}
```

Use the exact corpus and query-set fingerprints of the corresponding existing
row in `results/comparison/comparison.csv` when the benchmark is matched.
Record a different fingerprint when the question set or corpus changes.
Use an empty string for an unmeasured field, not zero. Include a link or path
to the underlying per-question rankings in the committed result folder so
paired differences can be checked before interpreting gains.

Run `.venv-week6/bin/python -m week10_followup.aggregate` after adding files.
The generated report keeps datasets separate and computes deltas only for
matching corpus, questions, cutoff, budget and metric scope.
