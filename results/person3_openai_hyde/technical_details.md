# Cheryl — OpenAI GPT-5.4 HyDE experiment

**Status: completed and independently replay-validated.** 200 generated passages, 14 retrieval methods, 56 metric rows, and 1,400 per-question rows. No final-answer experiment was performed.

## Main findings

On the same previously inspected 100 Legal RAG Bench questions and 4,876 passages, GPT-5.4 focused pure HyDE reached **47% Hit@5**, versus **29%** for original-question dense. Equal question/HyDE vector mixing reached **45%**. The highest hybrid Hit@5 was **43%**, from focused pure HyDE with 25% dense / 75% BM25 RRF.

| Method | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| Original-question dense | 29% | 0.1610 | 0.1933 |
| BM25 | 31% | 0.2295 | 0.2500 |
| Standard hybrid, 25% dense | 36% | 0.2490 | 0.2764 |
| Standard hybrid, 50% dense | 37% | 0.2362 | 0.2690 |
| Original prompt / 96, pure HyDE dense | 39% | 0.2510 | 0.2859 |
| Original prompt / 96, question + HyDE dense | 40% | 0.2557 | 0.2917 |
| Original / 96, pure HyDE hybrid, 25% dense | 41% | 0.2832 | 0.3148 |
| Original / 96, question + HyDE hybrid, 25% dense | 39% | 0.2587 | 0.2906 |
| Original / 96, question + HyDE hybrid, 50% dense | 42% | 0.2660 | 0.3042 |
| Focused prompt / 160, pure HyDE dense | 47% | 0.3018 | 0.3430 |
| Focused prompt / 160, question + HyDE dense | 45% | 0.2583 | 0.3055 |
| Focused / 160, pure HyDE hybrid, 25% dense | 43% | 0.2777 | 0.3156 |
| Focused / 160, question + HyDE hybrid, 25% dense | 39% | 0.2568 | 0.2897 |
| Focused / 160, question + HyDE hybrid, 50% dense | 41% | 0.2753 | 0.3088 |

**Interpretation:** With this stronger generator, pure HyDE is numerically better than equal question/HyDE mixing in the focused profile (47% vs 45%). Mixing and BM25 are not automatically beneficial. Pure focused dense also exceeds the best hybrid configuration (47% vs 43%). These are descriptive comparisons, not a significance claim for pure versus mixed retrieval.

## Paired Hit@5 inference

Twelve comparisons were specified in the previously prepared notebook before these outputs were generated. Exact two-sided McNemar tests use discordant per-query hits; Holm adjusts across all twelve comparisons. No significance tests were performed for MRR/nDCG.

| Selected comparison | Gained | Lost | Raw p | Holm p |
|---|---:|---:|---:|---:|
| Original prompt / 96, pure HyDE dense vs Original-question dense | 20 | 10 | 0.098737 | 0.830566 |
| Focused prompt / 160, pure HyDE dense vs Original-question dense | 29 | 11 | 0.006427 | 0.070692 |
| Focused prompt / 160, question + HyDE dense vs Original-question dense | 20 | 4 | 0.001544 | 0.018527 |
| Focused / 160, pure HyDE hybrid, 25% dense vs Standard hybrid, 25% dense | 10 | 3 | 0.092285 | 0.830566 |
| Focused / 160, question + HyDE hybrid, 50% dense vs Standard hybrid, 50% dense | 8 | 4 | 0.387695 | 1.000000 |

**45% mixed dense vs 29% question dense remains significant within this twelve-test family (Holm p≈0.0185).** The 47% pure dense result has raw p≈0.0064 but Holm p≈0.0707, so it does not meet 5% after adjustment. Statistical significance depends on paired wins/losses, not just the aggregate percentage difference. The hybrid improvements are not significant after adjustment.

This adjustment does not correct for all historical explorations across previous weeks. The test set was previously inspected and repeatedly used for development; the result supports a follow-up on new questions, not a claim of held-out generalization.

## Comparison with previous work

The earlier Qwen3-1.7B pure dense result was 28% under original prompt / 96 tokens. GPT-5.4 original / 96 pure dense is 39%; GPT-5.4 focused / 160 pure dense is 47%. OpenAI tokenization, chat formatting and decoding controls differ from local-model execution. These comparisons cannot isolate architecture/model capability perfectly.

Previously reported SmolLM3 question-mix dense 37% and hybrid 40% were recovered aggregate observations whose raw outputs are unavailable. The new results are fully retained, but no paired significance comparison against those lost raw records can be made. Do not reconstruct or invent those missing records.

Original-question dense, BM25 and standard hybrid d25 baselines were replayed against the earlier saved rankings; see historical_baseline_replay.json for exact top-100 comparisons. Standard hybrid at d50 is 37%, so new d50 hybrid must be compared to 37% at the same weights as well as to the historical 36% d25 baseline.

## Protocol and generation quality

- Model: `gpt-5.4-2026-03-05`; reasoning effort `none`; temperature 0; no tools; `store=false`; no SDK automatic retries. Temperature 0 does not guarantee identical future outputs.
- The API receives question and prompt only. Gold answers and evidence are used locally for retrieval evaluation, never in generation prompts.
- MiniLM: fixed `all-MiniLM-L6-v2` revision, CPU float32, max length 256; normalized corpus/query vectors. Mixing uses normalize(0.5q + 0.5h).
- BM25 inputs original questions. RRF constant 60, input depth 100, stable corpus-order ties.
- Original / 96: **94/100 outputs incomplete at the length cap**. Focused / 160: **100/100 completed**. None of the generated passages was truncated by MiniLM. The focused change affects both prompt and length; gains cannot be attributed to prompt alone.
- Four concurrent local API requests. Saved request durations are not isolated hardware/latency benchmarks. First five hash-selected question IDs form the connection pilot and are reused in the 100-question run.

**Estimated API cost: US$0.364565** for the 200 saved generations, using recorded token usage and standard input/output prices without cache discounts. Actual organization billing is authoritative. No final-answer or OpenAI embedding calls were made.

Generation fluency is not legal factual verification. Retrieval Hit@5 measures whether the annotated passage appears among the five results; it is not final-answer correctness.

## Files and reproduction

- `protocol.json`: exact prompts, model snapshot, dataset hashes and parameters.
- `generation_checkpoint.json` and `generated_passages.jsonl`: all prompts, generated text, token usage, response IDs and statuses.
- `comparison_k5.csv` / `retrieval_metrics.csv`: 14 methods at k=5 / k=1,3,5,10.
- `saved_rankings.npz`, `baseline_rankings.npz`, `per_query_metrics.csv`: raw rankings and per-query outcomes.
- `paired_tests.csv`: all twelve tests and Holm adjustments.
- `generation_audit.csv`, `generation_summary.csv`: truncation, output status, cost and timing.
- `validation.json`, `historical_baseline_replay.json`: independent replay checks.
- `embeddings.npz`: normalized corpus and original-question embeddings.
- `evaluation_queries.csv`, `passage_index.csv`: question IDs, references, and corpus-index mapping.
- See `person3_openai_hyde/README.md` for running stages; `validate.py RESULT_DIR` independently replays saved metrics without an API key or model download.

**Next step:** freeze a candidate before testing new questions, and separately evaluate final answers with a capable answer model. Retain the original96 comparison to separate the model change from the focused prompt/length profile.
