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

## Current shared submissions

| Owner | Submission | Status |
| --- | --- | --- |
| A / Yuchen | [a_dense.csv](a_dense.csv) | Adapted from held-out dense CSV after dataset-content and split verification; MRR/nDCG unreported |
| B / Person 2 | No shared CSV yet | 4,563-query seed-42 test split; results are reported separately in the team overview |
| C / Cheryl | [c_hybrid_rrf.csv](c_hybrid_rrf.csv) | Common 4,575-query split; answer scores remain pilot |
| D / Kevin | [d_threshold_mmr.csv](d_threshold_mmr.csv) | Owner's unchanged CSV for selected threshold and MMR configurations |

See [the all-four overview](../presentation/team_overview.md) and
[provenance.json](provenance.json) for source commits and verification details.
A's original CSV and run manifest are retained in `results/team_sources/`.
Missing measurements are blank, not zero. A uses CUDA; C/D use CPU with different
latency protocols, so their recorded timings must not be ranked as speed results.

The exporter above expects the root experiment's raw schema. A's different schema
was explicitly mapped into the common columns: `top_k` to `k`, `questions` to
`n_queries`, and measured Precision/Recall/Hit Rate/latency without alteration.
Its dataset/corpus/split identifiers were assigned only after checking its content
fingerprint and corpus/split rules against the shared Parquet and configuration.

B must use the shared corpus and exact hash split and repeat development selection
before exporting a final shared result. Relabeling the current B results with the
shared split identifier would be incorrect. Final answer evaluation is still
pending across the team.
