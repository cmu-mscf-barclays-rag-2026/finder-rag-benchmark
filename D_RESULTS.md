# FinDER Retrieval Refinement

## Objective and approach

Evaluated similarity-threshold filtering and maximal marginal relevance (MMR) as refinements to dense retrieval. The work includes development-set parameter selection, held-out retrieval evaluation, evidence-change analysis, and latency measurement across five retrieval configurations.

The corpus contains 5,830 deduplicated reference passages from 5,703 FinDER questions. The split contains 1,128 development and 4,575 test questions. Passages are embedded with `all-MiniLM-L6-v2` and ranked by cosine similarity. The initial candidate pool contains 100 passages; the primary evaluation uses the top five.

## Parameter selection

Thresholds of 0.3, 0.5, and 0.7 were evaluated. MMR weights of 0.3, 0.5, 0.7, and 1.0 were crossed with candidate counts of 10, 20, 50, and 100. Selection used development Recall@5, followed by nDCG@5.

Threshold 0.3 was selected. Higher thresholds reduced development recall: from 20.46% for dense retrieval to 14.85% at 0.5 and 0.75% at 0.7.

MMR selected λ = 1.0 and 10 candidates. This disables the diversity penalty and reproduces dense ranking. The best configuration with an active diversity penalty, λ = 0.7 and 10 candidates, achieved development Recall@5 of 19.95%, below the dense baseline.

## Test results

| Method | Precision@5 | Recall@5 | MRR@5 | nDCG@5 | Mean passages |
|---|---:|---:|---:|---:|---:|
| Dense | 0.04507 | 0.21177 | 0.15833 | 0.16889 | 5.000 |
| Dense + threshold (0.3) | 0.04503 | 0.21169 | 0.15833 | 0.16885 | 4.972 |
| MMR (λ = 1, fetch = 10) | 0.04507 | 0.21177 | 0.15833 | 0.16889 | 5.000 |

Threshold filtering slightly reduced context volume, but removed gold evidence for one question and returned no passages for six questions. Gold coverage was unchanged for the other 4,574 questions. Selected MMR left gold coverage unchanged for all 4,575 questions.

## Latency

Measured on the same CPU and 100 test questions, with three warm-up calls per method and three randomly interleaved repetitions. Timing includes question encoding where applicable, search, fusion/refinement, and evidence lookup.

| Method | Mean (ms) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|
| BM25 | 17.17 | 16.90 | 19.42 |
| Dense | 19.93 | 19.49 | 24.60 |
| Hybrid | 37.29 | 36.80 | 43.31 |
| Dense + threshold (0.3) | 20.32 | 19.95 | 24.51 |
| MMR (λ = 1, fetch = 10) | 20.05 | 19.75 | 24.46 |

Each method has 300 measurements. Differences among the dense variants are small and do not establish a reliable speed advantage. Exported batch-average latency uses a separate measurement protocol.

## Interpretation

Neither refinement improved held-out retrieval quality. Low-threshold filtering removed little context, while stronger filtering reduced evidence recall. Diversity-active MMR configurations underperformed dense retrieval on development questions.

The findings apply to pooled reference passages with MiniLM embeddings. Long passages are subject to the model's 256-token input limit. Full-filing retrieval and generated-answer quality were not evaluated.

Detailed parameter results, latency samples, evidence changes, and environment settings are saved in `results/d_refinement/`.
