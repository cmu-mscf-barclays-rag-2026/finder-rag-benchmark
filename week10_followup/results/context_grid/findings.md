# Meeting 10 context follow-up — ViDoRe finance

This re-scores the saved BM25, MiniLM and Qwen256 rankings on the same 309
English questions and 2,942 OCR pages. The full 44,496 matched traces are in
`per_query.csv`; `summary.csv` and `paired_neighbors.csv` contain aggregate and
paired results. There is no new model inference in this grid. All 5,562
overlapping 6k configurations match the saved Week 9 delivered IDs and token
counts exactly.

The grid uses seed depths **5, 8, 10, 20**, physical-page expansion **0/0,
1/0, 0/1, 1/1**, and budgets **6,000, 10,000, 12,000 tokens**. Page IDs are
deduplicated before packing. Complete pages are packed in candidate order,
skipping those that do not fit. The same pinned MiniLM tokenizer measures all
budgets. `next_ranked_seed_fits` diagnoses whether the first *unconsidered*
ranked seed would fit in the leftover budget; it does not claim relevance.

## Seed depth versus context capacity

No-sibling configurations:

| Method | Budget | K | Delivered hits / 309 | Recall | Mean pages | Median (P10–P90) | Mean excluded seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 6k | 10 | 244 | .496 | 6.03 | 6 (4–9) | 3.97 |
| BM25 | 6k | 20 | 244 | .497 | 6.35 | 6 (5–9) | 13.65 |
| BM25 | 12k | 10 | 277 | .586 | 9.77 | 10 (9–10) | .23 |
| BM25 | 12k | 20 | 279 | .608 | 12.12 | 11 (9–17) | 7.88 |
| MiniLM | 6k | 10 | 178 | .295 | 6.65 | 6 (5–9) | 3.35 |
| MiniLM | 6k | 20 | 179 | .297 | 7.29 | 7 (5–10) | 12.71 |
| MiniLM | 12k | 10 | 194 | .341 | 9.91 | 10 (10–10) | .09 |
| MiniLM | 12k | 20 | 202 | .364 | 13.44 | 13 (10–18) | 6.56 |
| Qwen256 | 6k | 10 | 208 | .381 | 6.57 | 6 (5–9) | 3.43 |
| Qwen256 | 6k | 20 | 209 | .386 | 7.26 | 7 (5–10) | 12.74 |
| Qwen256 | 12k | 10 | 230 | .442 | 9.90 | 10 (10–10) | .10 |
| Qwen256 | 12k | 20 | 240 | .472 | 13.32 | 13 (10–18) | 6.68 |

The 6k plateau is largely a packing limit: Top-20 Qwen adds only one delivered
hit over Top-10 while 12.74 of its twenty seed pages are excluded on average.
At 12k, Qwen Top-20 delivers ten more hits than Top-10, though it still
excludes 6.68 seeds on average. Increasing K and increasing budget address
different bottlenecks. With Top-10 Qwen and no neighbors, the next ranked seed
would fit in the unused budget for 13.9% of questions at 6k and 76.1% at
12k. Mean unused space rises from 373 to 2,681 tokens, so the larger budget
often exposes a seed-depth limit. As an unbounded seed-ranking diagnostic,
Qwen256's ranked Top-20 Hit is 82.5%, versus 67.6% delivered at 6k and 77.7%
delivered at 12k.

## Value of immediate neighbors

Paired against Top-10 with no expansion, using the same questions and budget:

| Method | Budget | Policy | Hits gained | Hits lost | Mean recall change | Mean seed-page change |
|---|---:|---|---:|---:|---:|---:|
| BM25 | 6k | 1/0 | 4 | 15 | −.036 | −2.06 |
| BM25 | 6k | 0/1 | 11 | 13 | −.032 | −2.07 |
| BM25 | 12k | 1/0 | 2 | 24 | −.028 | −2.76 |
| BM25 | 12k | 0/1 | 4 | 20 | −.027 | −2.81 |
| MiniLM | 6k | 1/0 | 14 | 28 | −.001 | −2.42 |
| MiniLM | 6k | 0/1 | 8 | 25 | −.029 | −2.49 |
| MiniLM | 12k | 1/0 | 18 | 11 | +.035 | −2.61 |
| MiniLM | 12k | 0/1 | 12 | 11 | +.007 | −2.67 |
| Qwen256 | 6k | 1/0 | 12 | 18 | −.006 | −2.12 |
| Qwen256 | 6k | 0/1 | 10 | 14 | −.014 | −2.20 |
| Qwen256 | 12k | 1/0 | 12 | 13 | +.016 | −2.34 |
| Qwen256 | 12k | 0/1 | 7 | 10 | +.013 | −2.57 |

Neighbor pages can raise the count of delivered pages while displacing ranked
seeds. At 12k, preceding pages help MiniLM recall more than following pages;
the Qwen256 directional difference is small. BM25 loses net hits under both
directions at both budgets. These are descriptive paired outcomes, not a
validated policy selection. The source qrels measure page relevance rather
than complete answer sufficiency; the ten-question fact pilot remains draft.

## Draft fact-coverage diagnostic

The separate `pilot_grid/` result applies the unreviewed ten-question,
twenty-fact annotations to every configuration. At Top-10 and 6k, Qwen256
without neighbors covers 52.5% of drafted facts on average and all drafted
facts for 5/10 questions. Adding one preceding page raises those pilot figures
to 68.3% and 6/10, even though the full-benchmark page Hit rate falls under
that policy. This is a useful disagreement to inspect with a human reviewer,
**not** evidence of a validated answer-sufficiency gain.

The 10k rows and all four K/policy combinations are retained in the CSV files.
The 256-token Qwen page encoding truncates 2,779 of 2,942 pages. The
[controlled 1,024-token follow-up](../qwen_window_compare/findings.md)
improves Qwen Top-10 delivered Hit at 6k from .673 to .773 on the same
questions, while the mean number of delivered pages stays near 6.6.
