# Cheryl: additional small-model HyDE search — recovered run summary

Status: **aggregate results recovered from execution logs; original per-query artifacts unavailable after workspace cleanup.** This folder does not contain the 400 generated passages or saved ranking arrays. These observations were produced by the completed runs, but cannot currently be independently revalidated from their raw outputs. This is not a fully reproducible result package. Earlier Qwen/FLAN files are unchanged.

## Main observations

Same 100 previously inspected Legal RAG Bench questions and 4,876 passages; fixed MiniLM encoder and gold passage IDs. Hit@5 measures evidence retrieval, not final-answer accuracy.

| Configuration | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| Original-question dense control | 29% | 0.1610 | 0.1933 |
| SmolLM3, original prompt / 96 tokens, pure HyDE dense | 29% | 0.2102 | 0.2298 |
| SmolLM3, original prompt / 96 tokens, question + HyDE dense | 35% | 0.2250 | 0.2563 |
| SmolLM3, focused prompt / 160 tokens, question + HyDE dense | **37%** | 0.1975 | 0.2403 |
| Standard hybrid control | 36% | 0.2490 | 0.2764 |
| SmolLM3, focused / 160, question + HyDE hybrid, 50% dense | **40%** | 0.2555 | 0.2918 |
| LFM2.5, focused / 160, question + HyDE hybrid, 50% dense | **40%** | 0.2407 | 0.2805 |
| LFM2.5, focused / 160, question + HyDE hybrid, 25% dense | 37% | **0.2698** | **0.2948** |

Highest observed dense Hit@5: 37%. Highest observed hybrid Hit@5: 40%, tied between two models. SmolLM3 has higher MRR/nDCG within that 40% tie. These maxima were selected from 32 configurations, not a fresh held-out test. The 35% dense configuration ranks evidence better than the 37% dense configuration, so there is no single best configuration across all metrics.

Earlier selected FLAN multi-answer hybrid reached 39% using another prompt/aggregation; the new 40% is one percentage point higher than that exploratory historical maximum. It is four percentage points above the standard 36% hybrid control. See [earlier results](../person3_hyde_robustness/findings.md).

## Paired Hit@5 tests recovered from the logs

| Comparison | Candidate-only hits | Control-only hits | Exact McNemar p | Holm p, 32 comparisons |
|---|---:|---:|---:|---:|
| SmolLM3 focused question-mix dense vs question dense | 11 | 3 | 0.057373 | 1.0000 |
| SmolLM3 focused question-mix hybrid, 50% dense vs standard hybrid | 10 | 6 | 0.454498 | 1.0000 |

Neither difference reaches the 5% significance level. MRR/nDCG differences above are descriptive; they were not tested for significance.

## What was tested

- Generators: SmolLM3-3B and LFM2.5-1.2B-Instruct.
- Two generation settings each: original prompt / 96-token cap; focused prompt / 160-token cap.
- Pure hypothetical-passage embedding or equal normalized-vector mixture of question and passage.
- Dense-only retrieval, or BM25 fusion at 25%, 50%, 75% dense weight.
- Total: 400 generated passages and 32 new retrieval configurations in the completed run. Their original per-query files were subsequently lost; only logged aggregate observations are restored here.
- Same data, encoder, top-100 candidate depth, and RRF constant 60. CPU BF16; greedy; non-thinking; repetition penalty 1.2; no-repeat 3-gram.
- The focused setting changes both prompt and length. The best combined result cannot be attributed to changing the generator alone.

## Files

- [Recovered aggregate CSV](recovered_comparison_k5.csv): 32 new configurations and two controls, rounded from log output. Precision is derived from Hit@5 / 5 because each question has one gold passage.
- [Recorded protocol](protocol_recovered.json): model revisions, prompts and retrieval settings.
- [Technical walkthrough](technical_walkthrough.md): generation, vector mixture and BM25/RRF equations.

Next: recover or regenerate raw outputs before treating these observations as a verified reproducible benchmark; then freeze a configuration and evaluate on new questions. Additional experimentation has not been authorized by this documentation update.
