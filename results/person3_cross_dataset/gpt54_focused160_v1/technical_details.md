# GPT-5.4 focused HyDE transfer to FiQA and LongBench v2

## Main findings

Both complete collections finished: **648 FiQA test queries against 57,638 documents**, and **503 LongBench v2 questions across all six domains and all three length groups**. No test questions were sampled away, and each LongBench question searches its own complete context before taking the top five chunks. All 5,030 LongBench predictions, evidence spans, reader prompts, accuracy tables and 2,515 RRF rankings passed independent checks. Dense ranks were recomputed for 219 of 1,509 query/method pairs using available caches. Earlier cache files were lost during environment resets; exhaustive dense-rank rechecking is not complete. This limitation concerns the additional audit, not missing benchmark questions or missing predictions.

On FiQA, the highest observed nDCG@10 was **0.3687** using **Original question dense**; original question dense scored **0.3687**, and standard hybrid at dense weight .25 scored **0.3050**. The legal dataset's large HyDE improvement did not repeat: pure HyDE was lower, and mixed HyDE was close to question-only dense rather than better on the primary metric.

On LongBench v2, the highest observed retrieval-backed answer accuracy was **46.72%** using **BM25**; original question dense scored **44.53%**, standard hybrid at dense weight .25 scored **43.54%**, and the same reader without retrieved context scored **37.77%**. These are descriptive best results, not independently selected methods or official leaderboard numbers.

## FiQA: retrieval evaluation

| Method | Hit@5 | Recall@5 | MRR@5 | nDCG@10 |
| --- | --- | --- | --- | --- |
| Original question dense | 57.25% | 36.71% | 0.4341 | 0.3687 |
| BM25 | 37.65% | 22.55% | 0.2702 | 0.2286 |
| Standard hybrid (dense .25) | 49.85% | 30.38% | 0.3742 | 0.3050 |
| Standard hybrid (dense .50) | 53.86% | 33.06% | 0.4055 | 0.3452 |
| Pure HyDE dense | 46.91% | 29.94% | 0.3611 | 0.3124 |
| Question + HyDE dense | 57.41% | 37.07% | 0.4274 | 0.3668 |
| Pure HyDE hybrid (dense .25) | 48.92% | 29.62% | 0.3675 | 0.3006 |
| Question + HyDE hybrid (dense .25) | 49.85% | 30.43% | 0.3737 | 0.3068 |
| Question + HyDE hybrid (dense .50) | 56.17% | 33.99% | 0.4183 | 0.3522 |

FiQA has multiple relevant answers per query. Hit@5 means at least one positive judgment was retrieved; Recall@5 counts the fraction of all positive judgments retrieved. nDCG@10 is the primary metric and is reported on a 0–1 scale. Unjudged documents are treated as nonrelevant. Dense vectors represent the first 256 MiniLM tokens of each document; BM25 uses the full document. Coincident query/document ID strings are not filtered, so this is not an exact reproduction of BEIR evaluator defaults.

## LongBench v2: retrieval plus fixed reader

| Method | Correct / 503 | Answer accuracy |
| --- | --- | --- |
| Original question dense | 224/503 | 44.53% |
| BM25 | 235/503 | 46.72% |
| Standard hybrid (dense .25) | 219/503 | 43.54% |
| Standard hybrid (dense .50) | 219/503 | 43.54% |
| Pure HyDE dense | 209/503 | 41.55% |
| Question + HyDE dense | 210/503 | 41.75% |
| Pure HyDE hybrid (dense .25) | 225/503 | 44.73% |
| Question + HyDE hybrid (dense .25) | 224/503 | 44.53% |
| Question + HyDE hybrid (dense .50) | 224/503 | 44.53% |
| Reader without context | 190/503 | 37.77% |

These are final multiple-choice answer accuracies, not evidence retrieval Hit@5. LongBench v2 does not supply chunk relevance labels. The same GPT-5.4 reader sees only the selected five chunks, question and choices. HyDE generation and retrieval see only the question. Gold labels are used only after predictions are recorded. The no-context control uses an empty text section in the same reader template.

This is a custom RAG adaptation with 224 MiniLM token chunks and 32-token overlap; it does not replicate the official full-context or 512-token RAG leaderboard setting. The public Hugging Face distribution calls its single split `train`; all 503 benchmark items are used only for evaluation here, with no fitting. Full contexts are searched locally; they are not passed whole to the API. Tasks requiring information from many locations or many demonstrations can be poorly served by five short chunks. A reader can also answer from prior knowledge, so final accuracy alone cannot isolate evidence retrieval quality.

## Planned paired comparisons

| Dataset / primary metric | Candidate | Control | Delta | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- |
| FiQA / nDCG@10 | Pure HyDE dense | Original question dense | -0.0563 | 9.9999e-06 | 0.000119999 |
| FiQA / nDCG@10 | Question + HyDE dense | Original question dense | -0.0019 | 0.763302 | 1 |
| FiQA / nDCG@10 | Pure HyDE hybrid (dense .25) | Standard hybrid (dense .25) | -0.0045 | 0.330447 | 1 |
| FiQA / nDCG@10 | Question + HyDE hybrid (dense .25) | Standard hybrid (dense .25) | +0.0018 | 0.562594 | 1 |
| FiQA / nDCG@10 | Question + HyDE hybrid (dense .50) | Standard hybrid (dense .50) | +0.0069 | 0.101769 | 0.92461 |
| FiQA / nDCG@10 | Question + HyDE hybrid (dense .50) | Standard hybrid (dense .25) | +0.0471 | 9.9999e-06 | 0.000119999 |
| LongBench / accuracy | Pure HyDE dense | Original question dense | -0.0298 | 0.0959236 | 0.92461 |
| LongBench / accuracy | Question + HyDE dense | Original question dense | -0.0278 | 0.092461 | 0.92461 |
| LongBench / accuracy | Pure HyDE hybrid (dense .25) | Standard hybrid (dense .25) | +0.0119 | 0.488456 | 1 |
| LongBench / accuracy | Question + HyDE hybrid (dense .25) | Standard hybrid (dense .25) | +0.0099 | 0.568172 | 1 |
| LongBench / accuracy | Question + HyDE hybrid (dense .50) | Standard hybrid (dense .50) | +0.0099 | 0.568172 | 1 |
| LongBench / accuracy | Question + HyDE hybrid (dense .50) | Standard hybrid (dense .25) | +0.0099 | 0.635308 | 1 |

The six candidate/control comparisons for each dataset were fixed in the protocol before outcomes. FiQA tests use two-sided paired sign-flip permutation tests of nDCG@10 (100,000 draws, seed 20261028); LongBench tests use exact two-sided paired McNemar tests of answer accuracy. Holm adjustment covers all 12 tests jointly. Other metrics and subgroup results are descriptive. This family does not cover the earlier legal experiments, nor a post-hoc comparison of the observed best methods. LongBench has 462 distinct contexts for 503 questions; question-level tests do not model correlation among questions sharing a context, so their p-values should be treated as exploratory benchmark evidence.

Planned comparisons below Holm-adjusted p=.05:

- fiqa: Pure HyDE dense versus Original question dense, delta -0.0563, adjusted p=0.000119999.
- fiqa: Question + HyDE hybrid (dense .50) versus Standard hybrid (dense .25), delta +0.0471, adjusted p=0.000119999.

No planned LongBench comparison was significant after Holm correction. The significant FiQA comparison of mixed HyDE hybrid at dense weight .50 against standard hybrid at .25 changes both the query and the fusion weight; it cannot isolate a HyDE benefit. The corresponding matched-.50 comparison was not significant.

## LongBench subgroups

### domain

| Method | Long In-context Learning (n=81) | Single-Document QA (n=175) | Multi-Document QA (n=125) | Long-dialogue History Understanding (n=39) | Code Repository Understanding (n=50) | Long Structured Data Understanding (n=33) |
| --- | --- | --- | --- | --- | --- | --- |
| Original question dense | 39.51% | 44.00% | 42.40% | 46.15% | 62.00% | 39.39% |
| BM25 | 48.15% | 44.57% | 43.20% | 35.90% | 68.00% | 48.48% |
| Standard hybrid (dense .25) | 41.98% | 42.29% | 42.40% | 35.90% | 60.00% | 42.42% |
| Standard hybrid (dense .50) | 45.68% | 44.00% | 38.40% | 33.33% | 58.00% | 45.45% |
| Pure HyDE dense | 44.44% | 40.57% | 40.00% | 28.21% | 62.00% | 30.30% |
| Question + HyDE dense | 39.51% | 39.43% | 39.20% | 43.59% | 58.00% | 42.42% |
| Pure HyDE hybrid (dense .25) | 46.91% | 41.71% | 41.60% | 35.90% | 68.00% | 42.42% |
| Question + HyDE hybrid (dense .25) | 48.15% | 41.71% | 41.60% | 35.90% | 66.00% | 39.39% |
| Question + HyDE hybrid (dense .50) | 46.91% | 44.57% | 40.80% | 38.46% | 56.00% | 42.42% |
| Reader without context | 43.21% | 36.57% | 38.40% | 25.64% | 54.00% | 18.18% |

### difficulty

| Method | hard (n=311) | easy (n=192) |
| --- | --- | --- |
| Original question dense | 42.44% | 47.92% |
| BM25 | 44.69% | 50.00% |
| Standard hybrid (dense .25) | 41.16% | 47.40% |
| Standard hybrid (dense .50) | 41.16% | 47.40% |
| Pure HyDE dense | 39.87% | 44.27% |
| Question + HyDE dense | 38.91% | 46.35% |
| Pure HyDE hybrid (dense .25) | 41.80% | 49.48% |
| Question + HyDE hybrid (dense .25) | 42.44% | 47.92% |
| Question + HyDE hybrid (dense .50) | 43.09% | 46.88% |
| Reader without context | 36.98% | 39.06% |

### length

| Method | long (n=108) | short (n=180) | medium (n=215) |
| --- | --- | --- | --- |
| Original question dense | 51.85% | 46.11% | 39.53% |
| BM25 | 56.48% | 43.33% | 44.65% |
| Standard hybrid (dense .25) | 51.85% | 43.33% | 39.53% |
| Standard hybrid (dense .50) | 53.70% | 41.67% | 40.00% |
| Pure HyDE dense | 50.93% | 41.67% | 36.74% |
| Question + HyDE dense | 50.00% | 42.22% | 37.21% |
| Pure HyDE hybrid (dense .25) | 58.33% | 42.78% | 39.53% |
| Question + HyDE hybrid (dense .25) | 53.70% | 42.78% | 41.40% |
| Question + HyDE hybrid (dense .50) | 51.85% | 42.78% | 42.33% |
| Reader without context | 42.59% | 37.22% | 35.81% |

## Execution and costs


- Environment recovery restored a persistent snapshot at 353 retrieval questions, after the old environment had reached approximately 422. A later reset restored 454 questions after 460 had been observed. Outputs after the snapshots were lost and some requests had to be repeated. The budget includes an additional $3.00 allowance for spending absent from the restored database. This is an estimated allowance, not verified usage or a billing statement.


- HyDE outputs: 1151 (648 financial questions + 503 long-context questions); 1 length-capped/incomplete outputs retained under the fixed 160-token cap.
- Logical LongBench predictions: 5,030 (10 methods/controls × 503 questions); 4686 unique reader requests after sharing identical complete prompts.
- Total successful API calls: 5837. Estimated usage cost **$17.843340** (HyDE $2.233727, readers $15.609613). This is a token-based estimate at $2.50/$15 per million input/output tokens, without cache discounts, not a billing statement.
- Transport recovery: 6 prior connection-failed attempts had no OpenAI response or token usage. They were retained separately and the missing logical calls were manually reissued after a successful read-only connectivity check. Their possible additional cost is conservatively reserved at **$0.097505**, giving a budget-accounting estimate, including the environment-recovery allowance, of **$20.940845**. All intended logical outputs are available. Successful calls retained in the checkpoints were reused; workers were refreshed between complete batches to avoid stale long-running connections.
- API model: gpt-5.4-2026-03-05, reasoning none, temperature 0, no tools, store false; HyDE cap 160, answer cap 32; no intended prediction remains missing.
- Encoder: pinned all-MiniLM-L6-v2, normalized float32 embeddings; question/HyDE mixed 50/50 before normalization. Equivalent PyTorch/ONNX float32 caches were checked against each other. After a runtime safety review rejected an unknown Microsoft telemetry connection, encoding resumed with entirely offline PyTorch, no ONNX Runtime import and explicit Hugging Face telemetry opt-outs; existing vectors were reused and compatibility rechecked. No quantized outputs were used. Indexing processes were also resumed after a memory limit; bounded text loading/tokenization preserved chunk spans and all contexts.
- BM25 k1=1.2, b=.75, binary query counts. RRF k=60, top-100 input lists, fixed .25/.50 dense weights. BM25 always receives the original question.

## Interpretation limits

The focused prompt was transferred with domain-specific wording before new test outcomes. The original96 profile was intentionally omitted because 94/100 earlier legal generations hit its cap. Thus this is a focused-profile transfer study, not a full repeat of every earlier prompt ablation. No prompt, weight, model, or chunk budget was tuned on these new test labels. However, both datasets are public and can overlap model pretraining; these are not contamination-free private holdouts. The legal, financial and long-context numbers measure different tasks and must not be placed in one undifferentiated accuracy column. The retriever is still a small fixed MiniLM model, and the LongBench result additionally depends on the reader and five-chunk budget.

## Artifacts

`protocol.json` and source hashes fix the experiment; `fiqa_metrics.csv`, `fiqa_per_query_metrics.csv` and `fiqa_rankings.npz` support financial retrieval checks; `longbench2_accuracy.csv`, `longbench2_predictions.csv`, per-query retrieval JSON and answer mappings support long-context checks. `api_outputs.jsonl` records generated text, exact prompts and usage; `api_checkpoint.sqlite` supports resume without duplicate requests. `paired_tests.csv`, `cost_summary.csv`, backend checks and `validation.json` record inference and verification. Source datasets and large corpus/index caches are excluded from the result ZIP and can be restored from pinned downloads using the supplied code. No API key is included.
