# Chunk-independent evidence metrics

The evaluator unions source-coordinate ranges of the **actual returned text** separately for each canonical source. It does not award credit from matching a parent/child ID alone. Adjacent or overlapping pieces can jointly cover a gold span; duplicate or overlapping retrieved text receives no extra evidence credit.

For each query:

- **All spans included:** 1 only when the union of returned ranges covers every character of every proposed required span; otherwise 0. Averaging across queries gives the full-evidence inclusion rate.
- **Complete span recall:** number of completely included spans divided by annotated span count. It is sensitive to how an annotator divides evidence into spans, although not to retrieval chunk boundaries.
- **Evidence character coverage:** covered gold characters / total gold characters, using interval unions in both numerator and denominator. Includes spaces and punctuation; never double-counts overlap.
- **Evidence token coverage:** fraction of tokens in the unioned gold slices whose complete character offsets are included. Uses the pinned MiniLM WordPiece tokenizer **only as a counting tool**. Tokenization is performed on the merged gold slices, not each returned chunk, so re-chunking does not change the denominator. Spaces have no tokens, so token coverage can equal 1 while character-exact inclusion is below 1.
- **Any span overlap:** 1 if even part of a gold span is present. A weak diagnostic, not answer sufficiency.
- **Any complete span:** 1 if at least one span is fully present. Does not mean all required evidence was returned.
- **Whole gold child included:** a separate legacy comparator requiring every character of the original gold child. It is not the same target as the fine-grained evidence annotation.
- **Returned tokens:** sum of body tokens delivered, including duplicated text if a retriever returns overlapping chunks. Metadata, separators and generation prompts are excluded.

Metrics are computed at k=1, 5 and 10, before and after a 2,048-token body budget. Prefix packing follows rank order and may truncate the final unit. It updates the actual returned end offset rather than pretending a truncated parent was fully delivered. Fixed top-k is not a fixed amount of text; use the common-budget rows and returned-token counts as well.

Macro averages give each question equal weight. These are evidence metrics, not generated-answer correctness, faithfulness, or a demonstration of legally sufficient reasoning. Unannotated alternative evidence may support the answer but receives no credit under this prototype.

`boundary_controls.json` contains synthetic delivery controls using a real annotated span. `comparison.csv` contains actual corpus-wide BM25 retrieval results. Do not present control scores as retrieval performance.
