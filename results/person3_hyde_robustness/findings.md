# Cheryl — HyDE generation and parameter robustness

**Goal:** Following last week's feedback, test whether better HyDE generation improves dense retrieval and, in turn, hybrid retrieval.

Legal RAG Bench: **100 questions / 4,876 passages**. Generator: **FLAN-T5-base**. Embeddings: **MiniLM-L6-v2**, fixed throughout. Generation uses only the question.

## Last week: starting point

Same benchmark: **100 questions / 4,876 passages**, FLAN-T5-base + MiniLM. These are last week's saved primary results.

| Method | Precision@5 | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 baseline | 6.2% | 31% | 0.2295 | 0.2500 |
| Original-question dense baseline | 5.8% | 29% | 0.1610 | 0.1933 |
| Standard hybrid: BM25 + question dense | 7.2% | 36% | 0.2490 | 0.2764 |
| Reference-answer dense — diagnostic only | 14.4% | 72% | 0.5498 | 0.5929 |
| Single HyDE answer: dense | 3.0% | 15% | 0.0883 | 0.1033 |
| HyDE hybrid: BM25 + HyDE dense | 7.4% | 37% | 0.2460 | 0.2765 |

**Why this week's follow-up:** Test whether improving the 15% HyDE dense result could also improve hybrid retrieval. The 72% result uses the dataset's gold answer; it is a diagnostic, not a usable HyDE result. Generation software/settings differ between weeks, so cross-week HyDE gains are descriptive; this week's recomputed controls provide the main comparison.

[Last week's source metrics](../person3_hyde/summary_k5.csv)

## 1. What I tested this week

| Parameter / method | Settings tested |
|---|---|
| Prompt style | Short answer; evidence passage |
| Maximum generated length | 32; 96 tokens |
| Decoding | Greedy; sampling with temperature 0.7 and top-p 0.9 |
| Random-seed check | Three seeds for sampled evidence passages at 96 tokens |
| Retrieval input | One HyDE answer; question + HyDE; three-answer embedding average; three-answer ranking fusion |
| Hybrid weights | BM25 / dense: 75% / 25%, 50% / 50%, 25% / 75% |
| Candidates before fusion | Top 20; Top 100 per retrieval list; RRF constant = 60 |
| Evaluation | Precision, Recall/Hit Rate, MRR, nDCG at K = 1, 3, 5, 10; paired McNemar tests |
| Experiment size | 10 generation settings; 3,000 generated hypotheses; 284 method/configuration rows per K |

Three-answer methods use three prompt variants, so they test both prompt diversity and the number of hypotheses.

## 2. Dense retrieval results

**Hit@5:** percentage of questions with the labelled evidence in the first five results. With one labelled passage per question, Recall@5 equals Hit@5.

| Retrieval input | Hit@5 range across 10 settings | Mean Hit@5 | Highest observed Hit@5 |
|---|---:|---:|---:|
| Original question — baseline | — | — | 29% |
| Single HyDE answer | 4%–19% | 8.8% | 19% |
| Three HyDE answers: average embeddings | 7%–21% | 13.8% | 21% |
| Three HyDE answers: fuse rankings (RRF) | 6%–20% | 12.3% | 20% |
| Original question + one HyDE answer | 24%–29% | 26.8% | 29% |
| Dataset reference answer — diagnostic only | — | — | 72% |

**Finding:** No tested HyDE dense method exceeded the original-question baseline. The 72% reference-answer result uses the gold answer and cannot be deployed.

## 3. Hybrid retrieval results

Both rows use **75% BM25 / 25% dense**, candidate depth **100**, and RRF constant **60**.

| Method | Precision@5 | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| Standard hybrid: BM25 + original-question dense | 7.2% | 36% | 0.2490 | 0.2764 |
| Highest observed HyDE hybrid: BM25 + three-answer RRF | 7.8% | **39%** | 0.2260 | 0.2666 |
| Change | +0.6 pp | +3 pp | -0.0230 | -0.0098 |

Selected HyDE setup: **short-answer prompt / 32 tokens / greedy / three hypotheses**. Paired comparison: **7 gained, 4 lost**; McNemar **p = 0.5488**, Holm-adjusted **p = 1** across 280 comparisons.

**Finding:** More Top-5 hits, but worse ranking quality and no statistically significant improvement. Improving standalone HyDE dense did not consistently improve hybrid retrieval.

## Conclusion and next step

**Conclusion:** These generation and parameter changes did not establish a reliable HyDE improvement with the current models.

**Next:** Freeze a configuration, validate on new questions, then test a stronger generator.

**Limits:** The same 100 questions were previously inspected; ranges and selected maxima are exploratory. Final answer correctness was not evaluated. Saved timings are batched CPU estimates, not interactive latency.

[Full generation comparison](generation_comparison_k5.csv) · [All weights and configurations](summary_k5.csv) · [Seed checks](seed_robustness_k5.csv) · [Paired tests](paired_comparisons_k5.csv) · [Help/hurt examples](help_hurt_examples.csv) · [Reproduction instructions](../../person3_hyde/ROBUSTNESS_README.md)
