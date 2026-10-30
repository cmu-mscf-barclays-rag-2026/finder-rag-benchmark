# FiQA Task 2 / BEIR FiQA baseline

The official BEIR FiQA archive (`fiqa.zip`, MD5
`17918ed23cd04fb15047f73e6c3bd9d9`) was downloaded into the ignored
local cache and evaluated on its test qrels. There are 57,638 answer passages
and 648 test queries; **428 queries have more than one judged relevant
answer** (maximum 15). This supports distinct Hit@k and Recall@k measures.

The baseline uses this repo's BM25 implementation with `k1=1.2`, `b=0.75`.
No parameters were tuned on test. The full results are in `summary.csv`,
`per_query.csv`, and `rankings.jsonl`.

| Depth and budget | Hit@k | Recall@k | All judged relevant | MRR@k | nDCG@k |
|---|---:|---:|---:|---:|---:|
| Top-5, 4,000 tokens | 38.9% | 23.5% | 13.0% | 0.278 | 0.213 |
| Top-10, 4,000 tokens | 48.3% | 29.7% | 16.5% | 0.290 | 0.235 |
| Top-10, 6,000 tokens | 48.5% | 29.7% | 16.5% | 0.290 | 0.235 |
| Top-20, 4,000 tokens | 54.2% | 34.4% | 19.9% | 0.295 | 0.250 |
| Top-20, 8,000 tokens | 56.0% | 36.1% | 21.1% | 0.296 | 0.255 |

These are **delivered** whole-answer metrics after the token budget. At
Top-10 with no binding budget, mean Hit@10 is 48.5% while mean Recall@10 is
29.7%, unlike the single-label Legal RAG setting. The mean 10-answer context
uses about 2,393 measurement tokens. At Top-20, 4,000 tokens admits about
16.8 answers on average, so the budget limits retrieval; 8,000 tokens admits
almost all 20.

FiQA labels say which answer passages are relevant. They do not establish
that every relevant answer is required to answer a question. Accordingly,
`all_judged_relevant` is reported as a retrieval diagnostic, not as verified
complete reasoning evidence. The [original FiQA Task 2 page](https://sites.google.com/view/fiqa/home)
states that its training and test data are for noncommercial use.
