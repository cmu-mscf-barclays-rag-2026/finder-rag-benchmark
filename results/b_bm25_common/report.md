# Florence / Person 2 — common-split BM25

Verified 5,830 passages, 1128 development queries, and 4575 test queries.
Parameters were selected on development queries and frozen before test evaluation.

| Method | k | Queries | P@k | R@k | MRR@k | nDCG@k | Evidence coverage@k | Answer EM | Answer correctness | Mean search ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b_bm25_default | 5 | 4575 | 0.0635 | 0.3022 | 0.2297 | 0.2449 | 0.3036 | N/A | N/A | 3.4114 |
| b_bm25_tuned | 1 | 4575 | 0.1832 | 0.1766 | 0.1832 | 0.1832 | 0.1767 | N/A | N/A | 3.6059 |
| b_bm25_tuned | 3 | 4575 | 0.0935 | 0.2676 | 0.2243 | 0.2318 | 0.2680 | N/A | N/A | 3.5272 |
| b_bm25_tuned | 5 | 4575 | 0.0655 | 0.3112 | 0.2340 | 0.2497 | 0.3116 | N/A | N/A | 3.6631 |
| b_bm25_tuned | 10 | 4575 | 0.0388 | 0.3661 | 0.2413 | 0.2678 | 0.3672 | N/A | N/A | 4.9629 |

The tokenizer and BM25 implementation are unchanged; the partition protocol changes. Hashed document IDs map to the same normalized passages and exact qrels. Original seed-42 experiments remain historical, separate results. The nine-point sweep, split IDs, per-query metrics, and top-10 test rankings are retained here. No answer generation was run. These single-run timings are not a cross-method speed ranking.
