# Week 3 — shared FinDER reference material

These are supporting materials for the Week 3 team comparison, including the
inherited hybrid baseline and saved contributions from other teammates. They are
not additional weekly deliverables from Florence.

For review, start with [Florence's Week 3 summary](../README.md) or the
[Week 3 team comparison](../results/team_comparison/README.md).

| Material | Purpose |
|---|---|
| `config/`, `data/`, [data license](DATA_LICENSE.md) | Pinned shared dataset and evaluation configuration |
| `team_metrics/`, [metric contract](TEAM_METRICS_SPEC.md) | Team submissions and their provenance |
| `scripts/`, `tests/` | Export, aggregation and validation utilities |
| `finder_hybrid_experiment.py`, `run_retrieval.py`, `reproduce_c_metrics.py` | Inherited Part C reference implementation |
| `results/`, `FinDER_Hybrid_C.ipynb`, `docs/` | Historical baseline results and technical explanation |

From the repository root, check the saved reference data/results or rebuild the
Week 3 comparison:

```sh
.venv/bin/python week3_bm25/shared_reference/scripts/validate_saved_results.py
.venv/bin/python week3_bm25/shared_reference/scripts/aggregate_team_metrics.py --k 5
```

To rerun the inherited Part C pipeline, install this folder's `requirements.txt`
and execute `reproduce_c_metrics.py`. Its subprocesses run from this folder.
Florence's own BM25 runner and controlled timing harness remain one folder up.
Other contributors' Git branches are unchanged by this branch's organization.
