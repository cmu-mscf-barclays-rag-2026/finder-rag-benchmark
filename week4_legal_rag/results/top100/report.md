# Top-100 candidate-recall diagnostic

Same 80 held-out questions and frozen configurations as the initial run. No retuning.

| Method | Top-5 hits | Gold at ranks 6–100 | Gold absent from Top-100 | Recall@100 |
|---|---:|---:|---:|---:|
| bm25 | 30 | 23 | 27 | 66.25% |
| dense | 23 | 38 | 19 | 76.25% |
| hybrid | 32 | 29 | 19 | 76.25% |

Hybrid pre-fusion union (up to 200 chunks): gold present for 65/80 questions. Fusion drops it from the final Top-100 for 4 questions.

## Interpretation

- Ranks 6–100: an ideal reranker could potentially recover these parent hits at Top-5 using this candidate set.
- Absent from Top-100: a reranker restricted to that list cannot recover the labeled passage. It might still rank below 100; this is not proof it is never retrievable.
- Hybrid uses its own selected BM25 settings and chunk size; its BM25 component differs from the standalone BM25 baseline.
- K counts actual chunks, not distinct parents. A gold-parent hit does not prove a chunk contains sufficient answer evidence.
- Recall@100 is a parent-hit ceiling for reranking these candidates, not predicted answer accuracy or an attainable guarantee.
- All 240 Top-20 rankings match the original run exactly by ID, with scores checked within 1e-5.
- Per-query failure categories and complete Top-100 rankings are saved alongside this report. This is a descriptive test-set diagnostic, not new model selection.
