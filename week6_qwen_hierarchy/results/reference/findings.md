# Week 6: Qwen hierarchy experiment

**Reference-only run; Qwen results are pending.**

Evidence inclusion uses the frozen Week 5 tokenizer and 80 test questions. Context budgets are body-only; model input windows use Qwen tokens.

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

Coverage efficiency is the mean of per-query coverage fractions divided by returned tokens (zero for empty contexts), scaled by 1,000 in this table.

MiniLM rows reuse saved rankings; BM25 is independent of the embedding. No Qwen window sweep ran; window_sweep.csv has headers only.

A MiniLM–Qwen difference mixes model capacity, training, tokenizer, instruction and window effects. Only the within-Qwen window sweep holds the encoder fixed. Whole-sibling selection changes delivery policy; compare it within each encoder.

The split is query-held-out, not parent-disjoint (three gold parents cross dev/test). No HyDE, answer generation, correctness judging, or latency benchmark is included.
