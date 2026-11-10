# Week 9 finance findings — November 9, 2026

Completed text-retrieval experiments on the full ViDoRe V3 Finance EN corpus:
**309 English queries, 2,942 pages, six reports.** All methods search the same corpus.
No test-set tuning or claims of statistical significance. The ten-question fact pilot is
**assistant-drafted and unreviewed**, so its scores are exploratory annotation coverage.

## Full benchmark: ranked pages

| Method | Hit@5 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.7896 | 0.8964 | 0.5880 | 0.6483 | 0.5333 |
| minilm | 0.5307 | 0.6311 | 0.3408 | 0.4238 | 0.3107 |
| qwen256 | 0.6570 | 0.7476 | 0.4447 | 0.5183 | 0.3945 |

## Equal 6,000-token budgets

| Method | Seed depth | Siblings before/after | Delivered Hit | Delivered Recall | Mean tokens |
|---|---:|---|---:|---:|---:|
| bm25 | 10 | 0/0 | 0.7896 | 0.4962 | 5654 |
| bm25 | 20 | 0/0 | 0.7896 | 0.4969 | 5779 |
| bm25 | 10 | 1/1 | 0.7573 | 0.4522 | 5823 |
| bm25 | 10 | 2/1 | 0.7346 | 0.4357 | 5853 |
| minilm | 10 | 0/0 | 0.5761 | 0.2951 | 5593 |
| minilm | 20 | 0/0 | 0.5793 | 0.2968 | 5791 |
| minilm | 10 | 1/1 | 0.5243 | 0.2825 | 5851 |
| minilm | 10 | 2/1 | 0.5243 | 0.2893 | 5874 |
| qwen256 | 10 | 0/0 | 0.6731 | 0.3809 | 5627 |
| qwen256 | 20 | 0/0 | 0.6764 | 0.3855 | 5821 |
| qwen256 | 10 | 1/1 | 0.6440 | 0.3658 | 5858 |
| qwen256 | 10 | 2/1 | 0.6214 | 0.3479 | 5890 |

All policies use whole-page skip-to-fit packing. Siblings are immediate PDF neighbors
in the same document, deduplicated before token accounting. Ranked metrics describe
seed retrieval; delivered metrics describe the entire context after expansion/packing.

## Ten-question draft fact pilot: Top10, no siblings, 4,000 tokens

| Method | Page Hit | Fact recall | All required facts covered | Facts / 1k tokens |
|---|---:|---:|---:|---:|
| bm25 | 0.7000 | 0.5833 | 0.5000 | 0.3458 |
| minilm | 0.8000 | 0.5583 | 0.5000 | 0.3379 |
| qwen256 | 0.7000 | 0.5250 | 0.5000 | 0.2870 |

These are rubric-dependent scores, not validated answerability or answer accuracy.
The pilot was selected after examining questions and sources; it is not a held-out evaluation.
Unannotated alternative evidence can cause false negatives. See the
[annotation protocol and issues](../annotations/PROTOCOL.md) before interpreting the numbers.

## Cases where relevance and fact coverage disagree

- bm25, query 10: sufficient despite incomplete page recall; page recall 0.500, fact recall 1.000.
- bm25, query 14: sufficient despite incomplete page recall; page recall 0.500, fact recall 1.000.
- bm25, query 2: hit without all annotated facts; page recall 0.500, fact recall 0.333.
- bm25, query 22: hit without all annotated facts; page recall 0.200, fact recall 0.500.
- minilm, query 10: sufficient despite incomplete page recall; page recall 0.500, fact recall 1.000.
- minilm, query 13: hit without all annotated facts; page recall 1.000, fact recall 0.000.
- minilm, query 14: sufficient despite incomplete page recall; page recall 0.250, fact recall 1.000.
- minilm, query 2: hit without all annotated facts; page recall 0.250, fact recall 0.333.
- qwen256, query 13: hit without all annotated facts; page recall 1.000, fact recall 0.000.
- qwen256, query 14: sufficient despite incomplete page recall; page recall 0.250, fact recall 1.000.
- qwen256, query 22: hit without all annotated facts; page recall 0.400, fact recall 0.250.
- qwen256, query 24: sufficient despite incomplete page recall; page recall 0.200, fact recall 1.000.

Full cases: [metric_disagreements.csv](metric_disagreements.csv).

## Cost and measurement limits

| Method | Local build / document encoding seconds | Query encoding seconds | Mean search ms |
|---|---:|---:|---:|
| bm25 | 1.09 | 0.00 | 3.746 |
| minilm | 219.07 | 0.52 | 0.258 |
| qwen256 | 1504.64 | 29.83 | 0.339 |

- Timings are local observations, affected by cache state and concurrent work; they are not a controlled hardware speed ranking.
- Qwen timing excludes the discarded invalid run and earlier interrupted work. The manifest records the corrected run and its cache hits; it is not the total development cost.
- BM25 indexes full OCR pages. The dense models encode only their first 256 tokens: MiniLM truncates 2,764 of 2,942 pages (93.9%). A sparse-versus-dense difference therefore mixes retrieval method with visible input length. Do not generalize it to full-page dense retrieval.
- Search timing excludes query embedding, model load, context packing and answer generation. Dense search uses one BLAS thread.
- MiniLM and Qwen256 both use 256-token document encoder windows. Their tokenizers, pooling and query instructions differ, so 256 tokens need not cover identical characters. The 4,096-token Qwen option was not completed or scored.
- The delivery ruler is the same pinned MiniLM tokenizer for every method and includes no reader prompt overhead.
- OCR can lose table/layout details. This run does not measure visual or combined retrieval.
- No HNSW, external embedding API, answer model, paid API usage, or human validation result is claimed.
- Published relevance labels and questions are retained for the full run, including flagged entity/answer-leakage issues. See the protocol for examples.

## Qwen preprocessing validation

An initial Qwen run was invalid: slicing a tokenized page removed its final end-of-text
token when truncation occurred, while short queries retained it. Week 9 now delegates
truncation to the tokenizer, asserts preservation of the actual special-token suffix,
and uses a distinct cache identity. The corrected six-input MPS smoke test agreed with
independent single-input CPU encodings above cosine 0.9999999. The invalid run is isolated
under `invalid_qwen256/` and excluded from every comparison above.

See [the failed diagnostic](invalid_qwen_cpu_check.json) and
[the corrected smoke check](qwen_smoke_check.json). Each reported Qwen run also has
a `<method>_cpu_check.json` tied to its exact saved vector file.

## Next experiment

Have a second annotator review the ten-question pilot against PDFs, expand alternative support,
and adjudicate the scope of conditional/rate questions. Freeze that version before comparing
sufficiency on new questions. Visual retrieval and HNSW remain separately scoped follow-ups.
