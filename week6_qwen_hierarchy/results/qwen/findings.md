# Week 6: Qwen hierarchy experiment

Qwen embedding window selected on development questions: **4096 tokens**.

Evidence inclusion uses the frozen Week 5 tokenizer and 80 test questions. Context budgets are body-only; model input windows use Qwen tokens.

## Answer

Qwen substantially improves dense evidence retrieval, but the hierarchy advantage still does not survive a 2,048-token delivery budget. At Hit@10, Qwen dense child retrieval reaches **44/80**, compared with **36/80** for direct parents, **33/80** for full-parent expansion, and **42/80** for selected siblings. Unlimited context reverses the ordering: child **53/80**, parent **56/80**, and full-parent expansion **57/80**.

The selected-sibling policy recovers most of the budgeted child baseline and is less damaging than returning full parents, but it does not improve on child retrieval. Its paired comparison against children is 3 wins, 5 losses, and 72 ties. Direct parents have 8 wins and 16 losses; full-parent expansion has 5 wins and 16 losses.

The development sweep plateaus after 4k on its registered criterion (2k: 6/20, 4k: 8/20, 8k: 8/20, 16k: 8/20). The result supports 4k over 2k on these 20 development questions, while 8k and 16k add no measured benefit. Because every child fits at 2k, this sweep changes only parent representations.

Qwen dense also exceeds MiniLM dense under the shared budget: child 44/80 versus 24/80 and parent 36/80 versus 21/80. The inherited hybrid weight is weaker than Qwen dense (budgeted child 29/80 and parent 24/80), so the Week 5 fusion setting should not be treated as Qwen-optimal.

The paired tests in `paired_comparisons.csv` are exploratory, two-sided exact McNemar tests without multiple-comparison adjustment. The small test set supports the direction and size of these observed differences more strongly than broad claims about other legal corpora.

## Complete test results

| Embedding | Window | Method | Mode | Budget | Hit@5 | Hit@10 | Coverage@10 | Mean tokens | P95 tokens | Coverage / 1k tokens |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0 | bm25 | child | unbounded | 23/80 | 29/80 | 36.2% | 3823 | 4628 | 0.0978 |
| none | 0 | bm25 | child | 2048 | 22/80 | 22/80 | 28.2% | 2039 | 2048 | 0.1377 |
| none | 0 | bm25 | parent | unbounded | 31/80 | 37/80 | 46.2% | 10569 | 19027 | 0.0473 |
| none | 0 | bm25 | parent | 2048 | 21/80 | 21/80 | 28.1% | 2039 | 2048 | 0.1373 |
| none | 0 | bm25 | child_to_parent | unbounded | 26/80 | 32/80 | 40.0% | 8956 | 15774 | 0.0496 |
| none | 0 | bm25 | child_to_parent | 2048 | 21/80 | 21/80 | 26.3% | 2039 | 2048 | 0.1284 |
| none | 0 | bm25 | child_to_selected_siblings | unbounded | 25/80 | 30/80 | 37.5% | 6062 | 8544 | 0.0656 |
| none | 0 | bm25 | child_to_selected_siblings | 2048 | 23/80 | 23/80 | 28.7% | 1977 | 2044 | 0.1450 |
| MiniLM | 256 | dense | child | unbounded | 23/80 | 28/80 | 35.0% | 3015 | 4000 | 0.1168 |
| MiniLM | 256 | dense | child | 2048 | 23/80 | 24/80 | 30.0% | 2031 | 2048 | 0.1465 |
| MiniLM | 256 | dense | parent | unbounded | 27/80 | 37/80 | 46.2% | 5176 | 9612 | 0.0941 |
| MiniLM | 256 | dense | parent | 2048 | 20/80 | 21/80 | 28.7% | 2037 | 2048 | 0.1403 |
| MiniLM | 256 | dense | child_to_parent | unbounded | 32/80 | 39/80 | 48.8% | 9973 | 19833 | 0.0689 |
| MiniLM | 256 | dense | child_to_parent | 2048 | 20/80 | 20/80 | 27.1% | 2048 | 2048 | 0.1324 |
| MiniLM | 256 | dense | child_to_selected_siblings | unbounded | 30/80 | 34/80 | 42.5% | 5468 | 7904 | 0.0867 |
| MiniLM | 256 | dense | child_to_selected_siblings | 2048 | 24/80 | 24/80 | 30.0% | 1996 | 2044 | 0.1502 |
| MiniLM | 256 | hybrid | child | unbounded | 24/80 | 33/80 | 41.2% | 3719 | 4494 | 0.1126 |
| MiniLM | 256 | hybrid | child | 2048 | 23/80 | 23/80 | 30.0% | 2045 | 2048 | 0.1467 |
| MiniLM | 256 | hybrid | parent | unbounded | 32/80 | 49/80 | 61.3% | 9677 | 15217 | 0.0667 |
| MiniLM | 256 | hybrid | parent | 2048 | 22/80 | 22/80 | 27.6% | 2048 | 2048 | 0.1345 |
| MiniLM | 256 | hybrid | child_to_parent | unbounded | 29/80 | 40/80 | 50.0% | 8757 | 16028 | 0.0644 |
| MiniLM | 256 | hybrid | child_to_parent | 2048 | 20/80 | 20/80 | 25.6% | 2046 | 2048 | 0.1250 |
| MiniLM | 256 | hybrid | child_to_selected_siblings | unbounded | 26/80 | 35/80 | 43.8% | 5938 | 8628 | 0.0787 |
| MiniLM | 256 | hybrid | child_to_selected_siblings | 2048 | 23/80 | 24/80 | 30.0% | 1988 | 2047 | 0.1514 |
| Qwen3-0.6B | 4096 | dense | child | unbounded | 44/80 | 53/80 | 66.2% | 3203 | 4137 | 0.2074 |
| Qwen3-0.6B | 4096 | dense | child | 2048 | 42/80 | 44/80 | 59.3% | 2037 | 2048 | 0.2900 |
| Qwen3-0.6B | 4096 | dense | parent | unbounded | 46/80 | 56/80 | 70.0% | 6153 | 10763 | 0.1287 |
| Qwen3-0.6B | 4096 | dense | parent | 2048 | 36/80 | 36/80 | 46.3% | 2048 | 2048 | 0.2258 |
| Qwen3-0.6B | 4096 | dense | child_to_parent | unbounded | 52/80 | 57/80 | 71.2% | 8618 | 17457 | 0.0955 |
| Qwen3-0.6B | 4096 | dense | child_to_parent | 2048 | 33/80 | 33/80 | 43.4% | 2048 | 2048 | 0.2121 |
| Qwen3-0.6B | 4096 | dense | child_to_selected_siblings | unbounded | 48/80 | 54/80 | 67.5% | 5261 | 8294 | 0.1336 |
| Qwen3-0.6B | 4096 | dense | child_to_selected_siblings | 2048 | 42/80 | 42/80 | 52.5% | 1986 | 2045 | 0.2650 |
| Qwen3-0.6B | 4096 | hybrid | child | unbounded | 30/80 | 38/80 | 47.5% | 3783 | 4565 | 0.1260 |
| Qwen3-0.6B | 4096 | hybrid | child | 2048 | 29/80 | 29/80 | 38.0% | 2044 | 2048 | 0.1857 |
| Qwen3-0.6B | 4096 | hybrid | parent | unbounded | 39/80 | 43/80 | 53.8% | 9893 | 17361 | 0.0600 |
| Qwen3-0.6B | 4096 | hybrid | parent | 2048 | 24/80 | 24/80 | 30.8% | 2048 | 2048 | 0.1502 |
| Qwen3-0.6B | 4096 | hybrid | child_to_parent | unbounded | 32/80 | 42/80 | 52.5% | 8836 | 14350 | 0.0611 |
| Qwen3-0.6B | 4096 | hybrid | child_to_parent | 2048 | 19/80 | 19/80 | 25.5% | 2044 | 2048 | 0.1247 |
| Qwen3-0.6B | 4096 | hybrid | child_to_selected_siblings | unbounded | 30/80 | 39/80 | 48.8% | 6076 | 8644 | 0.0814 |
| Qwen3-0.6B | 4096 | hybrid | child_to_selected_siblings | 2048 | 23/80 | 23/80 | 28.7% | 1988 | 2046 | 0.1443 |

Coverage efficiency is the mean of per-query coverage fractions divided by returned tokens (zero for empty contexts), scaled by 1,000 in this table.

MiniLM rows reuse saved rankings; BM25 is independent of the embedding. Qwen development results are in window_sweep.csv; only the selected window is evaluated on test.

A MiniLM–Qwen difference mixes model capacity, training, tokenizer, instruction and window effects. Only the within-Qwen window sweep holds the encoder fixed. Whole-sibling selection changes delivery policy; compare it within each encoder.

The split is query-held-out, not parent-disjoint (three gold parents cross dev/test). No HyDE, answer generation, correctness judging, or latency benchmark is included.
