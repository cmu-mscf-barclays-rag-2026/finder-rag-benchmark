# FinDER query-expansion findings

Held-out hash-selected test questions: **200** from the fixed 4,575-question FinDER test partition. All methods search the same 5,830 pooled reference passages with the tuned BM25 index. The index contains benchmark reference passages, not full financial filings. DSPy receives 16 development examples and no test labels; its actual optimization model-call count is in each manifest. The historical tuned-BM25 full-test Hit@5 is 0.321 over 4575 questions, versus 0.290 on this 200-question sample, so sample scores should not be extrapolated to the full test set.

| Generator | Method | K | Hit | Recall | MRR | nDCG | Alternatives | Unique candidates | Mean latency ms | Model calls | Input / output tokens | API USD estimate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt | single_query | 5 | 0.290 | 0.280 | 0.214 | 0.226 | 0.00 | — | 14.3 | 0 | 0 / 0 | $0.00000 |
| gpt | multiquery | 5 | 0.285 | 0.280 | 0.216 | 0.230 | 2.00 | 41.7 | 1023.1 | 200 | 14314 / 7292 | $0.00652 |
| gpt | dspy | 5 | 0.305 | 0.297 | 0.219 | 0.236 | 2.00 | 37.0 | 1453.5 | 200 | 71314 / 9784 | $0.01657 |
| gpt | single_query | 10 | 0.340 | 0.335 | 0.220 | 0.244 | 0.00 | — | 14.3 | 0 | 0 / 0 | $0.00000 |
| gpt | multiquery | 10 | 0.380 | 0.372 | 0.228 | 0.260 | 2.00 | 41.7 | 1023.1 | 200 | 14314 / 7292 | $0.00652 |
| gpt | dspy | 10 | 0.380 | 0.372 | 0.229 | 0.261 | 2.00 | 37.0 | 1453.5 | 200 | 71314 / 9784 | $0.01657 |
| qwen | single_query | 5 | 0.290 | 0.280 | 0.214 | 0.226 | 0.00 | — | 3.7 | 0 | 0 / 0 | local |
| qwen | multiquery | 5 | 0.305 | 0.297 | 0.213 | 0.231 | 2.00 | 42.1 | 3748.9 | 200 | 15472 / 8597 | local |
| qwen | dspy | 5 | 0.270 | 0.263 | 0.196 | 0.209 | 1.98 | 31.8 | 6971.4 | 200 | 74886 / 12315 | local |
| qwen | single_query | 10 | 0.340 | 0.335 | 0.220 | 0.244 | 0.00 | — | 3.7 | 0 | 0 / 0 | local |
| qwen | multiquery | 10 | 0.355 | 0.347 | 0.219 | 0.247 | 2.00 | 42.1 | 3748.9 | 200 | 15472 / 8597 | local |
| qwen | dspy | 10 | 0.330 | 0.325 | 0.204 | 0.230 | 1.98 | 31.8 | 6971.4 | 200 | 74886 / 12315 | local |

Intervals are 10,000 fixed-seed paired question-bootstrap draws, reported as descriptive marginal 95% intervals. A single run per generator was made.

## Paired Top-5 changes

| Comparison | Δ Hit | 95% interval | Wins / losses | Δ Recall | Δ nDCG |
|---|---:|---|---:|---:|---:|
| gpt multiquery minus single_query | -0.005 | [-0.040, +0.030] | 6 / 7 | +0.000 | +0.003 |
| gpt dspy minus single_query | +0.015 | [-0.015, +0.050] | 7 / 4 | +0.018 | +0.010 |
| qwen multiquery minus single_query | +0.015 | [-0.020, +0.050] | 7 / 4 | +0.018 | +0.005 |
| qwen dspy minus single_query | -0.020 | [-0.045, +0.005] | 1 / 5 | -0.018 | -0.017 |
| gpt_vs_qwen multiquery gpt minus qwen | -0.020 | [-0.060, +0.020] | 6 / 10 | -0.018 | -0.001 |
| gpt_vs_qwen dspy gpt minus qwen | +0.035 | [+0.005, +0.070] | 9 / 2 | +0.035 | +0.027 |

## Paired Top-10 changes

| Comparison | Δ Hit | 95% interval | Wins / losses | Δ Recall | Δ nDCG |
|---|---:|---|---:|---:|---:|
| gpt multiquery minus single_query | +0.040 | [+0.000, +0.080] | 12 / 4 | +0.037 | +0.016 |
| gpt dspy minus single_query | +0.040 | [+0.010, +0.075] | 10 / 2 | +0.037 | +0.017 |
| qwen multiquery minus single_query | +0.015 | [-0.020, +0.050] | 7 / 4 | +0.013 | +0.003 |
| qwen dspy minus single_query | -0.010 | [-0.040, +0.020] | 3 / 5 | -0.010 | -0.014 |
| gpt_vs_qwen multiquery gpt minus qwen | +0.025 | [-0.010, +0.060] | 9 / 4 | +0.025 | +0.013 |
| gpt_vs_qwen dspy gpt minus qwen | +0.050 | [+0.015, +0.085] | 12 / 2 | +0.048 | +0.031 |

## Interpretation

At Top-10, the paired BM25 baseline hits 0.340 of questions. GPT MultiQuery and DSPy both reach 0.380; Qwen MultiQuery reaches 0.355 and Qwen DSPy 0.330. The GPT DSPy-versus-Qwen DSPy paired Hit interval is above zero on this sample; the GPT-versus-Qwen MultiQuery interval crosses zero. GPT DSPy uses more input tokens and takes longer per query than GPT MultiQuery, while the local Qwen DSPy path is slower than local Qwen MultiQuery. These are single-run pilot findings, not full-test or answer-quality estimates.

Model calls and input/output tokens cover the test query-generation stage; Top-5 and Top-10 are scored from the same generated queries, so their displayed model calls and costs are not additive. The separate DSPy development optimization cost is in each manifest. Qwen runs locally; GPT-4o mini API estimates use $0.15/M input and $0.60/M output tokens from the [model pricing page](https://developers.openai.com/api/docs/models/gpt-4o-mini), without cached-input discounts or taxes. They are not billing records. Qwen and GPT token counts come from different tokenizers. Local Qwen and remote GPT latencies have different execution paths and are descriptive, not a hardware-normalized speed ranking. Retrieval metrics do not score generated answer correctness.
