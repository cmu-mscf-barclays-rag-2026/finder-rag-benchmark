# FinDER MultiQuery across retrieval backends

Matched pilot: 200 held-out questions, 5,830 pooled reference passages, two saved GPT-4o mini rewrites per question plus the original. All formulations retrieve 20 candidates. Query rankings use equal-weight RRF (k=60); hybrid combines the already fused BM25 and MiniLM lists using weighted RRF with the Week 3 dev-selected 25% dense weight. The weight was fixed before this test.

| Backend | Query | Hit@5 | Hit@8 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | original | 29.0% | 32.0% | 34.0% | 33.5% | 0.220 | 0.244 |
| bm25 | multiquery | 28.5% | 34.5% | 38.0% | 37.2% | 0.228 | 0.260 |
| dense | original | 19.0% | 23.5% | 24.5% | 23.8% | 0.152 | 0.171 |
| dense | multiquery | 22.0% | 24.5% | 26.5% | 25.5% | 0.162 | 0.182 |
| hybrid | original | 30.0% | 33.0% | 34.0% | 33.5% | 0.216 | 0.242 |
| hybrid | multiquery | 30.0% | 35.5% | 38.5% | 38.0% | 0.232 | 0.265 |

## Paired changes at Top-10

Positive values favor the method on the left of the comparison. Intervals are descriptive 95% paired question-bootstrap intervals (10,000 fixed-seed draws).

| Comparison | Δ Hit@10 | 95% interval | Wins / losses | Δ nDCG@10 |
|---|---:|---|---:|---:|
| bm25_multiquery − bm25_original | +4.0% | [+0.0%, +8.0%] | 12 / 4 | +0.016 |
| dense_multiquery − dense_original | +2.0% | [-1.5%, +5.5%] | 8 / 4 | +0.011 |
| hybrid_multiquery − hybrid_original | +4.5% | [+0.5%, +9.0%] | 14 / 5 | +0.022 |
| dense_original − bm25_original | -9.5% | [-15.5%, -3.5%] | 11 / 30 | -0.073 |
| hybrid_original − bm25_original | +0.0% | [-2.0%, +2.0%] | 2 / 2 | -0.002 |
| dense_multiquery − bm25_multiquery | -11.5% | [-18.0%, -5.0%] | 12 / 35 | -0.078 |
| hybrid_multiquery − bm25_multiquery | +0.5% | [-3.5%, +4.5%] | 9 / 8 | +0.005 |

## Fixed backend fusion check

The equal-weight backend alternative uses the same BM25 and dense candidate rankings. It is an exploratory test-set comparison, not a tuned replacement for the prior 25% dense weight.

| Query | Dense weight | Hit@10 | nDCG@10 | Δ Hit vs 25% | 95% interval |
|---|---:|---:|---:|---:|---:|
| original | 25% | 34.0% | 0.242 | — | — |
| original | 50% | 33.5% | 0.241 | -0.5% | [-3.5%, +2.5%] |
| multiquery | 25% | 38.5% | 0.265 | — | — |
| multiquery | 50% | 36.0% | 0.245 | -2.5% | [-6.0%, +1.0%] |

## Interpretation

MiniLM dense retrieval is weaker than BM25 on this reference-passage sample. MultiQuery raises dense Hit@10 by 2.0 points, with an interval spanning zero. The preset hybrid MultiQuery reaches 38.5% Hit@10 versus 38.0% for BM25 MultiQuery; its paired difference is only 0.5 points, with an interval spanning zero. Equal backend weights reduce hybrid MultiQuery Hit@10 to 36.0%, so fusion choice matters here. For this FinDER setup, BM25 MultiQuery is the practical default among these tested methods: the hybrid's small observed edge does not establish a reliable benefit from maintaining a dense index. This does not settle whether MultiQuery helps a stronger dense encoder on a different corpus.


## Evidence and cost diagnostics

Mean unique candidate IDs per question: BM25 MultiQuery 41.7, dense MultiQuery 36.4, hybrid MultiQuery 72.3. The fused dense Top-20 list contributes 13 query–reference matches absent from the fused BM25 Top-20 list; BM25 contributes 39 absent from dense.

Local MiniLM corpus embedding took 85.4 s. Batched encoding and ranking of 600 query formulations took 1.2 s. These batch times exclude GPT query generation and are not directly comparable with the earlier sequential BM25 latency. The saved GPT MultiQuery run used one generation call per question (200 total); no new generation calls were made here.

This benchmark scores reference-passage retrieval, not answer correctness. Its 200-question results are a pilot and should not be extrapolated to the full 4,575-question test set. The hybrid uses two fusion stages, so its gains can reflect candidate pooling and ranking together. See `summary.csv`, `paired.csv`, `candidate_diagnostics.csv`, and `rankings.jsonl` for the full matched results.
