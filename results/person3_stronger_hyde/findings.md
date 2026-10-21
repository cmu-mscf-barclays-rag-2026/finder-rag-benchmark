# Cheryl — stronger HyDE generators

**Question:** Does replacing FLAN-T5-base with a newer local instruction model improve evidence retrieval and final answers?

100 previously inspected Legal RAG Bench questions;4,876 passages. MiniLM fixed; one hypothesis; identical evidence-passage prompt;96-token generation cap;greedy;BM25/dense weights75%/25%;RRF60;depth100. Qwen uses official non-thinking chat templates and CPU bfloat16. FLAN outputs are reused, not regenerated.

## 1. Retrieval comparison

| Method | Precision@5 | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| Original-question dense | 5.8% | 29% | 0.1610 | 0.1933 |
| BM25 | 6.2% | 31% | 0.2295 | 0.2500 |
| Standard hybrid | 7.2% | 36% | 0.2490 | 0.2764 |
| Reference-answer oracle (diagnostic) | 14.4% | 72% | 0.5498 | 0.5929 |
| FLAN-T5-base HyDE dense (matched) | 3.8% | 19% | 0.0982 | 0.1209 |
| FLAN-T5-base HyDE hybrid (matched) | 6.2% | 31% | 0.2328 | 0.2520 |
| Qwen3-0.6B HyDE dense | 5.0% | 25% | 0.1422 | 0.1689 |
| Qwen3-0.6B HyDE hybrid | 6.0% | 30% | 0.2075 | 0.2307 |
| Qwen3-1.7B HyDE dense | 5.6% | 28% | 0.1793 | 0.2043 |
| Qwen3-1.7B HyDE hybrid | 7.0% | 35% | 0.2702 | 0.2898 |

Matched FLAN means the same single-passage/96-token/greedy setup. The earlier39% multi-HyDE result used a different selected setup and is not a matched control. The72% oracle uses real reference answers and is diagnostic only.

**Finding:** Qwen3-1.7B raises matched HyDE dense Hit@5 from19% to28%, while question dense is29%. Its hybrid Hit@5 is35% versus36% for standard hybrid; MRR@5 (0.2702 vs0.2490) and nDCG@5 (0.2898 vs0.2764) are numerically higher. This is a ranking/coverage tradeoff, not a clear overall win.

## 2. Paired evidence gains/losses

| Comparison | Gained | Lost | McNemar p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Qwen3-0.6B HyDE dense vs Original-question dense | 8 | 12 | 0.5034 | 1.0000 |
| Qwen3-0.6B HyDE dense vs FLAN-T5-base HyDE dense (matched) | 14 | 8 | 0.2863 | 1.0000 |
| Qwen3-0.6B HyDE hybrid vs Standard hybrid | 3 | 9 | 0.1460 | 1.0000 |
| Qwen3-0.6B HyDE hybrid vs FLAN-T5-base HyDE hybrid (matched) | 5 | 6 | 1.0000 | 1.0000 |
| Qwen3-1.7B HyDE dense vs Original-question dense | 8 | 9 | 1.0000 | 1.0000 |
| Qwen3-1.7B HyDE dense vs FLAN-T5-base HyDE dense (matched) | 17 | 8 | 0.1078 | 0.8620 |
| Qwen3-1.7B HyDE hybrid vs Standard hybrid | 5 | 6 | 1.0000 | 1.0000 |
| Qwen3-1.7B HyDE hybrid vs FLAN-T5-base HyDE hybrid (matched) | 8 | 4 | 0.3877 | 1.0000 |

Eight planned comparisons; all exploratory. Higher Hit@5 alone does not establish higher ranking quality or general superiority.

## 3. Generation quality and cost

| Model | Median generation seconds/query | P95 seconds/query | Mean words | Reached96-token cap | Truncated by MiniLM |
|---|---:|---:|---:|---:|---:|
| qwen06 | 8.77 | 9.69 | 73.9 | 99% | 0% |
| qwen17 | 18.93 | 20.75 | 70.1 | 100% | 0% |

Serial CPU query measurements exclude model loading, warm-up, downloads and corpus encoding. The two generation jobs shared the same CPU host, so these are indicative timings with contention, not isolated hardware benchmarks. FLAN's old batched timing is not directly comparable. More words and fewer short outputs are mechanical quality checks, not semantic correctness.

**Limit:** 99% of0.6B and100% of1.7B outputs reached the96-token generation cap; none was truncated by MiniLM. The fixed generation budget may prevent complete hypotheses. Do not interpret these as unconstrained model-quality results.

## 4. Final-answer pilot

Fixed eight IDs selected by hash before the new outcomes; same Qwen3-0.6B answer model, prompt,128-token cap,top5 packing and2048 passage-token budget for all methods. One question is12.5 percentage points. These scores come from **model-assisted qualitative review**, not independent legal-expert annotation or an automatic truth metric.

| Retrieved context | Questions | Fully correct | At least partly correct | Fully grounded | Citations present and all labels valid |
|---|---:|---:|---:|---:|---:|
| standard_hybrid | 8 | 0% | 25% | 0% | 50% |
| flan_hybrid | 8 | 12% | 38% | 0% | 62% |
| qwen06_hybrid | 8 | 0% | 25% | 0% | 62% |
| gold_context | 8 | 12% | 50% | 25% | 75% |
| qwen17_hybrid | 8 | 0% | 25% | 0% | 62% |

Grounding is judged against the actual supplied context; a valid citation number does not prove support. A supported abstention can be grounded but not correct. No pilot question asks for a numerical calculation, so numerical accuracy is N/A. Gold-context answers are diagnostic.

## Conclusion and next step

Keep standard hybrid as the main control. Qwen3-1.7B is worth a targeted follow-up for ranking quality, but does not yet establish a retrieval improvement: its dense difference versus FLAN has unadjusted McNemar p=0.1078; see the paired table for the other comparisons. No planned Hit@5 comparison is significant at5%, even before adjustment.

Next test a larger generation budget and a prompt that discourages invented legal citations, one change at a time. The answer pilot also requires a stronger or better-constrained answer generator and independent review before treating its scores as final answer accuracy. Freeze promising settings and evaluate on new questions. All current results are exploratory, not an independent held-out evaluation.

**Pilot finding:** The local answer generator remains unreliable even with gold-only context. For example, Q39 includes the correct alibi passage but hybrid answers miss the required application for leave; Q91 includes the case name but answers do not extract R v Donnelly; Q50's gold-context answer contradicts the explicit statement that Victoria has no standalone accident defence. These examples are diagnostic, not independent legal-expert judgments.

[Reviewed answers and rationale](answer_reviews.csv) · [Exact contexts and outputs](final_answers.csv) · [Full retrieval metrics](retrieval_metrics.csv) · [Per-query timing](latency_per_query.csv) · [Reproduction and rubric](../../person3_stronger_hyde/README.md)
