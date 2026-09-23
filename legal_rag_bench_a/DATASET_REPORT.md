# Dataset audit

Inspected: 22 September 2026. Dataset revision: `db0b31dc6d195ce9916897e1ac5e4e6209736c8a`.

## Source and distributed units

The corpus comes from the Victorian Criminal Charge Book. The authors converted Word sections into Markdown, preserved heading structure, and used semantic chunking where necessary. The paper describes a 512-token bound using the Kanon tokenizer. This audit does not reproduce that tokenizer count; its measured lengths use whitespace words and Unicode characters.

The released files contain passages, rather than the original Word documents. Grouping IDs before the `-cN-sN` suffix yields 724 section groups. Their lengths sum passage text only, exclude footnotes, and do not establish exact original document lengths or a complete reconstruction.

## Actual fields

| File | Field | Observed type | Interpretation |
|---|---|---|---|
| corpus.jsonl | id | string | Stable passage ID, e.g. `1.2-c2-s2` |
| corpus.jsonl | title | string | Section/subsection title |
| corpus.jsonl | text | string | Markdown passage body |
| corpus.jsonl | footnotes | string or null | Separate footnote content |
| qa.jsonl | id | integer | Query ID; normalized to string by our loader |
| qa.jsonl | question | string | Expert-written question/scenario |
| qa.jsonl | answer | string | Reference answer for later generation evaluation |
| qa.jsonl | relevant_passage_id | string | One labeled supporting passage |

The data card's simplified schema omits title/footnotes and describes question IDs as strings. Code follows the inspected records. The normalized `gold_ids` list contains the single original gold ID, allowing a common evaluator interface without inventing extra evidence.

## Measured results

| Statistic | Value |
|---|---:|
| Corpus passages | 4,876 |
| Questions | 100 |
| Unique gold passages | 95 |
| Single-gold queries | 100 (100%) |
| Multi-gold queries | 0 (0%) |
| ID-derived section groups | 724 |
| Passages with nonempty footnotes | 1,907 |
| Repeated exact-text rows beyond the first occurrence | 218 |

Repeated text is preserved because passage IDs carry location and evaluation identity. Do not deduplicate it as in the earlier FinDER experiment without explicitly remapping labels.

| Length in whitespace words | Minimum | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Passage body | 2 | 209 | 382 | 452 |
| ID-derived section, summed bodies | 77 | 790.5 | 4,284.35 | 11,424 |
| Question | 4 | 46.5 | 91 | 162 |
| Reference answer | 14 | 49.5 | 80.2 | 118 |

The corpus contains 1,019,417 passage-body words. Median passage length is 1,216 characters. Extremely short passages can be headings or update notices; preserve the released corpus for the initial comparable baseline.

## What the benchmark can establish

Questions were deliberately written to differ lexically from their supporting passage. This motivates BM25-versus-dense-versus-hybrid comparisons, but does not guarantee that dense retrieval will win. Observe that outcome experimentally.

Each query has one gold ID, and the authors aimed to make that passage sufficient. Thus hit rate, recall, labeled-evidence coverage, and all-gold inclusion coincide for these labels. Multiple legal issues in a question do not establish a requirement for multiple passages. Other useful passages may exist but are not exhaustively annotated.

The original study evaluates retrieval at top 5 and separately judges answer correctness and groundedness. Our standard ranking metrics extend the retrieval-only analysis; they do not reproduce the study's model comparisons or LLM judging.

## Implications for the team

- **Initial retrieval comparison:** keep the released passage IDs and text unchanged across methods. Use the same query subset and k values.
- **Text policy:** initial runs index only `text`. Explicitly report any title/footnote augmentation; repeated footnotes can increase context length and duplication.
- **Chunking:** the corpus is already chunked. Additional splitting requires child IDs and offsets. Mapping a child to a gold parent measures parent discovery, not whether the child contains the necessary answer span. Merging passages changes the retrieval/context budget. Keep both outside the initial baseline or report separately.
- **Embedding limits:** 512 Kanon tokens are not necessarily 512 tokens for another model. Check truncation with the chosen embedding tokenizer rather than assuming all passage content fits.
- **Graph experiment:** source-derived section membership and explicit citations can provide graph edges. Do not build edges using query answers or gold relevance mappings. Passage-label metrics alone cannot establish that graph retrieval combines several necessary facts.
- **Completeness extension:** a future evaluation needs expert-validated sets of jointly required evidence, possibly with alternative sufficient sets. This dataset does not provide those annotations. Do not automatically label neighboring or cited passages as required evidence.
- **Financial relevance:** this is an auxiliary semantic retrieval test in one legal jurisdiction. Financial-domain conclusions require financial documents and queries as well.

## Evidence and reproduction

The numerical findings above are computed from the pinned files; see `results/dataset_profile.json`, the length tables, and `results/profile_manifest.json`. The benchmark-design claims are described in [the paper](https://arxiv.org/html/2603.01710v1) and [the dataset card](https://huggingface.co/datasets/isaacus/legal-rag-bench). Default top-k and the original pipeline are documented in [the authors' code repository](https://github.com/isaacus-dev/legal-rag-bench).
