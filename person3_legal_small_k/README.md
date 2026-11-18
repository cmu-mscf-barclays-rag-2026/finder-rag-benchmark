# Legal Top-3 follow-up

Read the saved original BM25(Q), BM25(H) and BM25(Q+H) Top-100 lists for the same 100 Legal questions. No API, embedding or retrieval calls are made.

Three exact paired McNemar comparisons at Top 3 are Holm-adjusted together. Historical Top-5 rates and paired counts are replay checks, not additional Top-3 tests.

[Combined original/fresh results and mechanisms](../presentation/legal_qh_replication_20261118/README.md)

With the original local files restored, run from the repository root:

```bash
python person3_legal_small_k/run.py prepare
OPENBLAS_NUM_THREADS=1 python person3_legal_small_k/run.py evaluate
```

For a check using only the published tables, run `presentation/legal_qh_replication_20261118/reproduce.py` instead. All comparisons are post-hoc and exploratory; the Holm3 family does not adjust other historical tests.
