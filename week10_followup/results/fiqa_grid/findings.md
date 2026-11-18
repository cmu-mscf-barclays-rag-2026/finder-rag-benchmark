# Meeting 10 FiQA depth and budget grid

This uses the saved BEIR FiQA BM25 Top-20 ranking on 648 test questions and
57,638 answer passages. It applies the same pinned context measurement
tokenizer as the ViDoRe grid, with K = 5, 8, 10, 20 and 6k/10k/12k budgets.
FiQA passages are independent answers, so physical-page sibling expansion
does not apply. The 7,776 per-question rows are in `per_query.csv`. All 1,944
overlapping Week 8 rows have identical delivered passage counts, token counts
and Hit values.

| K | Budget | Delivered Hit | Recall | Mean passages | Mean tokens |
|---:|---:|---:|---:|---:|---:|
| 5 | 6k | .389 | .235 | 5.00 | 1,164 |
| 8 | 6k | .455 | .278 | 8.00 | 1,899 |
| 10 | 6k | .485 | .297 | 10.00 | 2,393 |
| 20 | 6k | .559 | .359 | 19.60 | 4,678 |
| 20 | 10k | .560 | .361 | 20.00 | 4,847 |
| 20 | 12k | .560 | .361 | 20.00 | 4,847 |

At K=10, all ten FiQA answer passages fit under 6k for every question; raising
the budget to 12k changes no score. The slight K=20 gain from 6k to 10k is
because 0.40 passages per question were excluded at 6k on average. FiQA's
main limit here is candidate depth, whereas the same 6k budget admits only
6.57 of the ten Qwen256 ViDoRe pages on average. This contrast is about
passage length and the tested retrieval configurations; raw Hit rates differ
in corpus and relevance definitions and are not pooled.

FiQA relevant answers can be alternatives rather than jointly necessary
facts. Recall is judged-answer retrieval, not answer sufficiency.
