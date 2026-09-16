# Controlled retrieval latency — all four workstreams

One Mac, CPU only, the same 100 held-out questions, 3 warmup calls per method, and 5 randomly interleaved repetitions. Each method has 500 measurements.

| Method | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| a_dense_minilm_500_75 | 18.354 | 15.730 | 30.368 |
| b_bm25_tuned | 7.323 | 6.944 | 13.829 |
| hybrid_rrf_75bm25_25dense | 24.189 | 22.345 | 36.301 |
| d_threshold_0.3 | 15.062 | 13.206 | 25.876 |
| d_mmr_1.0_10 | 14.800 | 13.089 | 22.954 |

Index embeddings were prepared on mps outside the timed region; the model was then moved to CPU. Measured from raw question to top-five evidence text, including query encoding where needed, search, source pooling, fusion/refinement, and text lookup. Indexing, downloads, and generation are excluded. All model calls use CPU with four intra-op threads and one inter-op thread; BLAS pools are limited to four threads.

These measurements support a local comparison of these implementations and selected configurations, not a general claim about hardware or algorithm speed. A indexes 44,679 chunks while C/D index 5,830 whole passages; this is part of each selected method. B is Florence’s implementation, not the BM25 control inside C. Raw repetitions, query IDs, source commits, model revision, and package versions are retained. Retrieval scores in team_metrics remain the owners’ full-test results; this timing run is not a quality rerun.

Reproduce after running the shared BM25 benchmark: `python scripts/common_latency.py --model-cache PATH --cache PATH`. Download the pinned model revision first. The original mixed-hardware latencies remain in team_metrics for provenance and must not be used as a speed ranking.
