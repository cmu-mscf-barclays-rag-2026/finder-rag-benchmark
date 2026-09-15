# Team metric submissions

Each person adds one CSV file to this directory. Do not edit another person's
file. Use the exporter so that every submission has the same columns.

Example for part C:

```bash
python scripts/export_method_metrics.py \
  --retrieval-csv results/retrieval_metrics.csv \
  --answer-csv results/answer_correctness.csv \
  --method "Hybrid RRF alpha=0.25" \
  --method-id hybrid_rrf_75bm25_25dense \
  --owner C \
  --answer-status pilot \
  --answer-protocol-id flan-t5-small-top3-50-pilot-v1 \
  --output team_metrics/c_hybrid_rrf.csv
```

Only held-out `test` rows are exported. `answer_*` columns may be blank when a
method has not yet completed the shared answer-generation experiment.
Only use `--answer-status final` after the entire team uses the same answer
sample, prompt, generator, context count, and token limit.
