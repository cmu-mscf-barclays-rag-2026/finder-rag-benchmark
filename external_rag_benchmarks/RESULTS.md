# External benchmark pilot results

BM25, Qwen dense and reciprocal-rank fusion were run locally. FiQA measures document retrieval; LongBench v2 measures final multiple-choice answers from retrieved context. These are different outcomes.

## FiQA

Full corpus: 57,638 entries. Query pilot: 50 official development and 100 official test questions, selected by a fixed hash before scoring. Of the selected test questions, 71 have multiple judged relevant documents.

| Method | Hit@10 | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 41.0% | 25.3% | 0.2173 | 0.2866 |
| Qwen dense | 81.0% | 56.0% | 0.4920 | 0.5890 |
| Hybrid equal | 67.0% | 46.9% | 0.3769 | 0.4449 |
| Hybrid selected | 75.0% | 53.7% | 0.4432 | 0.5138 |

The highest observed test nDCG@10 is from **Qwen dense**. The development-selected fusion configuration is evaluated as selected even when its test result is weaker than an untuned baseline.

Development-selected BM25 fusion weight: **0.25**. Selection maximizes development nDCG@10, then Recall@10. Equal fusion is shown separately. A weight of zero is identical to the dense-only endpoint; a weight of one is sparse-only.

Measured corpus encoding: 797.7 seconds on NVIDIA GeForce RTX 5070 Laptop GPU. Dense input truncation rate at the fixed 512-token Qwen window: 4.0%. This is a fixed-window model comparison, not a new input-window sweep.

## LongBench v2

This follow-up reuses the original twenty short QA items: six project development and fourteen previously inspected test questions. It is exploratory, not a new untouched evaluation. Each question searches its own supplied context. For every retriever, the same top-forty ranking is used at 2,048 tokens, 8,192 tokens and all retrieved text. The answer model remains Qwen2.5-1.5B-Instruct.

**All retrieved text means all forty complete retrieved chunks, not the entire original source.** No additional body cutoff is applied in that condition. Every complete prompt, including the question and choices, is checked with the reader tokenizer and space is reserved for generation.

| Method | Budget | Correct / 14 | Accuracy | Mean body tokens |
|---|---|---:|---:|---:|
| No context | none | 2/14 | 14.3% | 0.0 |
| BM25 | 2048 | 4/14 | 28.6% | 2048.0 |
| BM25 | 8192 | 4/14 | 28.6% | 8192.0 |
| BM25 | all_retrieved | 5/14 | 35.7% | 10239.1 |
| Qwen dense | 2048 | 3/14 | 21.4% | 2048.0 |
| Qwen dense | 8192 | 3/14 | 21.4% | 8192.0 |
| Qwen dense | all_retrieved | 3/14 | 21.4% | 10221.2 |
| Hybrid equal | 2048 | 3/14 | 21.4% | 2048.0 |
| Hybrid equal | 8192 | 3/14 | 21.4% | 8192.0 |
| Hybrid equal | all_retrieved | 3/14 | 21.4% | 10236.9 |

Highest observed pilot test accuracy: 35.7%, obtained by **BM25 / all_retrieved**. This is descriptive of fourteen questions, not a reliable ordering of the methods on the full benchmark.

### Effect of adding context

- BM25: 4/14 at 2,048; 4/14 at 8,192; 5/14 with all forty chunks.
- Qwen dense: 3/14 at 2,048; 3/14 at 8,192; 3/14 with all forty chunks.
- Hybrid equal: 3/14 at 2,048; 3/14 at 8,192; 3/14 with all forty chunks.

Larger delivered contexts did not produce a consistent accuracy gain in this pilot. This does not establish why errors occurred: useful evidence may remain outside the top forty, or the reader may fail to use evidence that is present. There are no annotated support locations to adjudicate those explanations automatically.

All 200 generation conditions ran locally. Maximum complete prompt: 14,047 reader tokens; model capacity: 32,768. Prompt truncations: **0**.

The no-context control uses the same questions and answer options without source text. A correct response can reflect evidence, prior knowledge or guessing. Four choices imply 25% expected uniform-random accuracy. This sample cannot reliably establish a winner; one question changes its test-subset accuracy by about 7.1 percentage points.

Paired wins and losses for each larger budget versus 2,048 are saved in results/longbench_budget_sweep/paired_changes.csv. The original top-10 results remain in results/longbench as a baseline. The small-budget delivered text is checked to be identical across the original and follow-up protocols.

One BM25 8,192-token generation had an unexplained 299-second runtime outlier. Raw timings are preserved. timing_summary.csv reports median and maximum as well as mean; do not use that single-run mean to infer an inherent latency penalty for BM25 or its budget.

## Interpretation

- FiQA lets Hit and Recall differ because queries can have multiple relevant answers. Those answers may be alternatives; retrieving every relevance label is not proof of complete required evidence.
- LongBench provides correct answer letters but no exact supporting-span labels. No retrieval Recall or span-coverage score is claimed for it.
- All three retrievers use the same input questions within a benchmark. FiQA fusion tuning uses development judgments only; LongBench fusion is fixed at equal weight.
- Results are exploratory samples. They do not establish superiority across financial or legal document collections. The small local reader is a practical pilot model, not a frontier-model comparison.
- These experiments complement the Legal RAG span prototype. They do not validate its draft evidence annotations.

## Execution and checks

Answer generation ran on NVIDIA GeForce RTX 5070 Laptop GPU. Paid API calls: **0**. Models and raw datasets were downloaded locally; GPU vectors, weights and raw downloads are omitted from the upload.

See results/audit.json and results/longbench_budget_sweep/audit.json for independent score recomputation, source-hash checks, fixed-ranking checks, prompt capacity and context-slice verification. Timing files report component costs from one pass, not repeated end-to-end latency benchmarks.
