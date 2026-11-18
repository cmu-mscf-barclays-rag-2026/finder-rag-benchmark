# Cross-dataset retrieval comparison

Each row keeps its corpus, test-query set, retrieval unit, cutoff and context budget. Deltas compare only within an identical setting; ViDoRe deltas use BM25 OCR as a descriptive reference and mix retriever/window differences. Raw scores across FinDER, FiQA and ViDoRe do not share relevance definitions.

| Dataset | Method | K | Budget | Queries | Hit | Recall | nDCG | Δ Hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FinDER | single_query_bm25 | 5 | — | 200 | 0.290 | 0.280 | 0.226 | +0.000 |
| FinDER | single_query_bm25 | 10 | — | 200 | 0.340 | 0.335 | 0.244 | +0.000 |
| FinDER | multiquery_gpt | 5 | — | 200 | 0.285 | 0.280 | 0.230 | -0.005 |
| FinDER | multiquery_gpt | 10 | — | 200 | 0.380 | 0.372 | 0.260 | +0.040 |
| FinDER | dspy_gpt | 5 | — | 200 | 0.305 | 0.297 | 0.236 | +0.015 |
| FinDER | dspy_gpt | 10 | — | 200 | 0.380 | 0.372 | 0.261 | +0.040 |
| FinDER | multiquery_qwen | 5 | — | 200 | 0.305 | 0.297 | 0.231 | +0.015 |
| FinDER | multiquery_qwen | 10 | — | 200 | 0.355 | 0.347 | 0.247 | +0.015 |
| FinDER | dspy_qwen | 5 | — | 200 | 0.270 | 0.263 | 0.209 | -0.020 |
| FinDER | dspy_qwen | 10 | — | 200 | 0.330 | 0.325 | 0.230 | -0.010 |
| ViDoRe_finance_en | bm25_neighbors_0/0 | 10 | 6000 | 309 | 0.790 | 0.496 | 0.503 | +0.000 |
| ViDoRe_finance_en | bm25_neighbors_0/0 | 10 | 12000 | 309 | 0.896 | 0.586 | 0.533 | +0.000 |
| ViDoRe_finance_en | bm25_neighbors_0/1 | 10 | 6000 | 309 | 0.783 | 0.464 | 0.467 | -0.006 |
| ViDoRe_finance_en | bm25_neighbors_0/1 | 10 | 12000 | 309 | 0.845 | 0.559 | 0.498 | -0.052 |
| ViDoRe_finance_en | bm25_neighbors_1/0 | 10 | 6000 | 309 | 0.754 | 0.460 | 0.460 | -0.036 |
| ViDoRe_finance_en | bm25_neighbors_1/0 | 10 | 12000 | 309 | 0.825 | 0.558 | 0.492 | -0.071 |
| ViDoRe_finance_en | bm25_neighbors_0/0 | 20 | 6000 | 309 | 0.790 | 0.497 | 0.502 | +0.000 |
| ViDoRe_finance_en | bm25_neighbors_0/0 | 20 | 12000 | 309 | 0.903 | 0.608 | 0.538 | +0.000 |
| ViDoRe_finance_en | bm25_neighbors_0/1 | 20 | 6000 | 309 | 0.783 | 0.464 | 0.466 | -0.006 |
| ViDoRe_finance_en | bm25_neighbors_0/1 | 20 | 12000 | 309 | 0.845 | 0.560 | 0.499 | -0.058 |
| ViDoRe_finance_en | bm25_neighbors_1/0 | 20 | 6000 | 309 | 0.754 | 0.461 | 0.459 | -0.036 |
| ViDoRe_finance_en | bm25_neighbors_1/0 | 20 | 12000 | 309 | 0.828 | 0.564 | 0.493 | -0.074 |
| ViDoRe_finance_en | minilm_neighbors_0/0 | 10 | 6000 | 309 | 0.576 | 0.295 | 0.297 | -0.214 |
| ViDoRe_finance_en | minilm_neighbors_0/0 | 10 | 12000 | 309 | 0.628 | 0.341 | 0.311 | -0.269 |
| ViDoRe_finance_en | minilm_neighbors_0/1 | 10 | 6000 | 309 | 0.521 | 0.266 | 0.276 | -0.269 |
| ViDoRe_finance_en | minilm_neighbors_0/1 | 10 | 12000 | 309 | 0.631 | 0.347 | 0.300 | -0.265 |
| ViDoRe_finance_en | minilm_neighbors_1/0 | 10 | 6000 | 309 | 0.531 | 0.294 | 0.291 | -0.259 |
| ViDoRe_finance_en | minilm_neighbors_1/0 | 10 | 12000 | 309 | 0.650 | 0.376 | 0.318 | -0.246 |
| ViDoRe_finance_en | minilm_neighbors_0/0 | 20 | 6000 | 309 | 0.579 | 0.297 | 0.297 | -0.210 |
| ViDoRe_finance_en | minilm_neighbors_0/0 | 20 | 12000 | 309 | 0.654 | 0.364 | 0.316 | -0.249 |
| ViDoRe_finance_en | minilm_neighbors_0/1 | 20 | 6000 | 309 | 0.521 | 0.266 | 0.275 | -0.269 |
| ViDoRe_finance_en | minilm_neighbors_0/1 | 20 | 12000 | 309 | 0.638 | 0.352 | 0.302 | -0.265 |
| ViDoRe_finance_en | minilm_neighbors_1/0 | 20 | 6000 | 309 | 0.534 | 0.297 | 0.291 | -0.256 |
| ViDoRe_finance_en | minilm_neighbors_1/0 | 20 | 12000 | 309 | 0.657 | 0.378 | 0.318 | -0.246 |
| ViDoRe_finance_en | qwen256_neighbors_0/0 | 10 | 6000 | 309 | 0.673 | 0.381 | 0.377 | -0.117 |
| ViDoRe_finance_en | qwen256_neighbors_0/0 | 10 | 12000 | 309 | 0.744 | 0.442 | 0.394 | -0.152 |
| ViDoRe_finance_en | qwen256_neighbors_0/1 | 10 | 6000 | 309 | 0.660 | 0.367 | 0.357 | -0.129 |
| ViDoRe_finance_en | qwen256_neighbors_0/1 | 10 | 12000 | 309 | 0.735 | 0.455 | 0.381 | -0.162 |
| ViDoRe_finance_en | qwen256_neighbors_1/0 | 10 | 6000 | 309 | 0.654 | 0.375 | 0.360 | -0.136 |
| ViDoRe_finance_en | qwen256_neighbors_1/0 | 10 | 12000 | 309 | 0.741 | 0.458 | 0.386 | -0.155 |
| ViDoRe_finance_en | qwen256_neighbors_0/0 | 20 | 6000 | 309 | 0.676 | 0.386 | 0.377 | -0.113 |
| ViDoRe_finance_en | qwen256_neighbors_0/0 | 20 | 12000 | 309 | 0.777 | 0.472 | 0.402 | -0.126 |
| ViDoRe_finance_en | qwen256_neighbors_0/1 | 20 | 6000 | 309 | 0.660 | 0.367 | 0.355 | -0.129 |
| ViDoRe_finance_en | qwen256_neighbors_0/1 | 20 | 12000 | 309 | 0.741 | 0.462 | 0.383 | -0.162 |
| ViDoRe_finance_en | qwen256_neighbors_1/0 | 20 | 6000 | 309 | 0.654 | 0.375 | 0.360 | -0.136 |
| ViDoRe_finance_en | qwen256_neighbors_1/0 | 20 | 12000 | 309 | 0.741 | 0.461 | 0.386 | -0.162 |
| ViDoRe_finance_en | qwen1024_neighbors_0/0 | 10 | 6000 | 309 | 0.773 | 0.460 | 0.458 | -0.016 |
| ViDoRe_finance_en | qwen1024_neighbors_0/0 | 10 | 12000 | 309 | 0.822 | 0.540 | 0.485 | -0.074 |
| ViDoRe_finance_en | qwen1024_neighbors_0/1 | 10 | 6000 | 309 | 0.741 | 0.429 | 0.423 | -0.049 |
| ViDoRe_finance_en | qwen1024_neighbors_0/1 | 10 | 12000 | 309 | 0.812 | 0.523 | 0.453 | -0.084 |
| ViDoRe_finance_en | qwen1024_neighbors_1/0 | 10 | 6000 | 309 | 0.748 | 0.439 | 0.429 | -0.042 |
| ViDoRe_finance_en | qwen1024_neighbors_1/0 | 10 | 12000 | 309 | 0.816 | 0.529 | 0.455 | -0.081 |
| ViDoRe_finance_en | qwen1024_neighbors_0/0 | 20 | 6000 | 309 | 0.777 | 0.465 | 0.459 | -0.013 |
| ViDoRe_finance_en | qwen1024_neighbors_0/0 | 20 | 12000 | 309 | 0.841 | 0.576 | 0.495 | -0.061 |
| ViDoRe_finance_en | qwen1024_neighbors_0/1 | 20 | 6000 | 309 | 0.741 | 0.430 | 0.422 | -0.049 |
| ViDoRe_finance_en | qwen1024_neighbors_0/1 | 20 | 12000 | 309 | 0.816 | 0.529 | 0.454 | -0.087 |
| ViDoRe_finance_en | qwen1024_neighbors_1/0 | 20 | 6000 | 309 | 0.748 | 0.439 | 0.427 | -0.042 |
| ViDoRe_finance_en | qwen1024_neighbors_1/0 | 20 | 12000 | 309 | 0.819 | 0.539 | 0.458 | -0.084 |
| FiQA | bm25 | 10 | 6000 | 648 | 0.485 | 0.297 | 0.235 | +0.000 |
| FiQA | bm25 | 10 | 12000 | 648 | 0.485 | 0.297 | 0.235 | +0.000 |
| FiQA | bm25 | 20 | 6000 | 648 | 0.559 | 0.359 | 0.255 | +0.000 |
| FiQA | bm25 | 20 | 12000 | 648 | 0.560 | 0.361 | 0.255 | +0.000 |

## Completed teammate methods

The committed Legal and LongBench studies change different pipeline stages and score different outcomes. Their full 43 long-form metric rows are preserved in `team_method_comparison.csv`. FinDER fuses branch Top-20 lists once and scores their Top-5/Top-10 prefixes; the Legal study truncates each branch at each K before fusion and generates three alternatives, versus two on FinDER. Legal DSPy optimizes the answer reader; FinDER DSPy optimizes queries. LongBench measures multiple-choice answer accuracy, while FinDER and ViDoRe rows here measure retrieval.

| Dataset | Method | Stage changed | Metric | Score | Δ from matched baseline |
|---|---|---|---|---:|---:|
| Legal RAG Bench | BM25 + fixed reader | baseline | reference_token_f1_lexical_proxy | 0.279 | +0.000 |
| Legal RAG Bench | MultiQuery + fixed reader | retrieval | reference_token_f1_lexical_proxy | 0.308 | +0.030 |
| Legal RAG Bench | BM25 + DSPy CoT, optimized | reader_instruction_optimization_and_CoT | reference_token_f1_lexical_proxy | 0.252 | -0.027 |
| LongBench v2 | BM25 + fixed reader | baseline | multiple_choice_answer_accuracy | 0.356 | +0.000 |
| LongBench v2 | MultiQuery + fixed reader | retrieval | multiple_choice_answer_accuracy | 0.308 | -0.047 |
| LongBench v2 | BM25 + DSPy CoT, optimized | reader_prompt_and_CoT_default_instruction_retained | multiple_choice_answer_accuracy | 0.346 | -0.010 |

## New teammate ViDoRe query-expansion study

Kevin’s committed study uses 247 optimizer-held-out English questions from the same 2,942-page ViDoRe corpus. It uses a different query subset, tuned BM25 settings, Qwen3-VL-Embedding-2B OCR/image backends, and GPT-4o mini query generation. Deltas below are paired within each backend; they are not paired with Florence’s 309-question context grid. DSPy uses MIPROv2 query instruction optimization, unlike Florence’s FinDER BootstrapFewShot program. The full 64 English test rows at K=1/5/10/20 are in `team_vidore_query_expansion.csv`.

| Backend | Method | K | Hit | Recall | nDCG | Δ Hit | Δ nDCG |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | original | 10 | 0.874 | 0.564 | 0.504 | +0.000 | +0.000 |
| bm25 | multiquery | 10 | 0.883 | 0.588 | 0.519 | +0.008 | +0.015 |
| bm25 | dspy_initial | 10 | 0.883 | 0.581 | 0.524 | +0.008 | +0.021 |
| bm25 | dspy_optimized | 10 | 0.891 | 0.582 | 0.519 | +0.016 | +0.015 |
| ocr | original | 10 | 0.854 | 0.574 | 0.517 | +0.000 | +0.000 |
| ocr | multiquery | 10 | 0.846 | 0.569 | 0.507 | -0.008 | -0.009 |
| ocr | dspy_initial | 10 | 0.850 | 0.567 | 0.510 | -0.004 | -0.006 |
| ocr | dspy_optimized | 10 | 0.850 | 0.566 | 0.507 | -0.004 | -0.010 |
| image | original | 10 | 0.907 | 0.628 | 0.579 | +0.000 | +0.000 |
| image | multiquery | 10 | 0.915 | 0.623 | 0.572 | +0.008 | -0.007 |
| image | dspy_initial | 10 | 0.895 | 0.623 | 0.575 | -0.012 | -0.004 |
| image | dspy_optimized | 10 | 0.907 | 0.618 | 0.569 | +0.000 | -0.010 |
| combined | original | 10 | 0.915 | 0.636 | 0.576 | +0.000 | +0.000 |
| combined | multiquery | 10 | 0.907 | 0.619 | 0.554 | -0.008 | -0.022 |
| combined | dspy_initial | 10 | 0.903 | 0.628 | 0.574 | -0.012 | -0.002 |
| combined | dspy_optimized | 10 | 0.915 | 0.628 | 0.571 | +0.000 | -0.005 |

Team contribution files loaded: 0.
Additional teammates’ results can be added under `week10_followup/team_results/` with the schema in its README, then this script can be rerun.
Legal/LongBench source commit: `18bf882b612b5fa3c0d83a4b861603fd8b424446`. ViDoRe query-expansion source commit: `8df85985ea1ea3f3b7e7508d7390f2970f4dab6c`.
