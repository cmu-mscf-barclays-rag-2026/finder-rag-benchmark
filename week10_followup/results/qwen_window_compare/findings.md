# Controlled Qwen document-window comparison on ViDoRe finance

This compares the same pinned Qwen3-Embedding-0.6B encoder over the same
2,942 OCR pages and 309 English questions. The document window changes from
256 to 1,024 Qwen tokens. Query instructions, the 2,048-token query window,
search, tokenizer, context packing, and query embeddings are identical; the
saved query vectors match exactly. The 1,024-token run preserves the final
special token, and an independent CPU check of six pages and six queries
matches the saved GPU vectors (minimum cosine 0.9999998). This is a sampled
cross-device check, not a full-corpus equivalence proof. All 7,416 overlapping
no-neighbor and 1/1 delivery traces match the original Qwen1024 run's
delivered IDs and token counts exactly.

At 256 tokens, 2,779/2,942 pages are truncated. At 1,024 tokens, 1,039
pages are still truncated, so this is a longer-window comparison rather than
a full-page encoding result.

| Ranked cutoff | Hit 256 | Hit 1024 | Newly hit | Lost hit | nDCG 256 | nDCG 1024 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .414 | .505 | 50 | 22 | .367 | .457 |
| 3 | .583 | .689 | 46 | 13 | .368 | .448 |
| 5 | .657 | .754 | 41 | 11 | .371 | .457 |
| 10 | .748 | .822 | 32 | 9 | .394 | .485 |
| 20 | .825 | .893 | 28 | 7 | .421 | .519 |

After whole-page context packing with no neighbors:

| K | Budget | Delivered Hit 256 | Delivered Hit 1024 | Newly hit / lost | Mean pages 256 → 1024 |
|---:|---:|---:|---:|---:|---:|
| 10 | 6k | .673 | .773 | 39 / 8 | 6.57 → 6.61 |
| 10 | 12k | .744 | .822 | 32 / 8 | 9.90 → 9.92 |
| 20 | 6k | .676 | .777 | 39 / 8 | 7.26 → 7.20 |
| 20 | 12k | .777 | .841 | 28 / 8 | 13.32 → 13.32 |

The longer input improves which pages rank high, while the 6k budget still
limits Top-10 delivery to about 6.6 pages. On this corpus, Qwen1024 Top-10
at 6k remains below the BM25 OCR delivered Hit rate of .790, so extending
the Qwen input does not make it the strongest tested text-only retriever.
The draft ten-question fact pilot at Top-10/6k without neighbors changes
from .525 to .583 mean fact recall; those annotations are unreviewed and
are not evidence of validated answer-sufficiency improvement.

The full paired delivered changes for all K, policies, and budgets are in
`paired_delivered.csv`; the ranked comparisons and 14,832 per-question
delivery traces are in the adjacent CSVs. Kevin's Qwen3-VL OCR/image model
is a different encoder and input modality, so its scores are not a controlled
window comparison with this run.
