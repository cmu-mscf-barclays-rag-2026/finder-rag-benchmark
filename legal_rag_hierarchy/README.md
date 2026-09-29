# Legal RAG Bench: Hierarchy Construction and Analysis

Part 1 prepares reproducible child and merged-parent corpora for Part 2's retrieval experiments. Open **Hierarchy_Analysis.ipynb** for the findings, real examples, and token-size tables.

## Findings

- 4,876 original children form **2,785 c-parent groups** within 724 broader section-ID groups.
- Parent construction removes only the final `-sN` suffix. `1.2-c2-s1` and `1.2-c2-s2` belong to `1.2-c2`, not to a parent containing all of section `1.2`.
- All groups have consistent titles and contiguous child numbers. Source order already agrees with numeric child order.
- Across 2,091 adjacent child boundaries, no normalized suffix/prefix word overlap was found. Text is preserved without deduplication. This does not rule out semantic repetition elsewhere.
- 1,092 adjacent pairs repeat the same nonempty footnote metadata. Footnotes are retained as metadata but excluded from both retrieval texts, preventing asymmetric context expansion.
- 2,008 parents contain only one child; merging changes the remaining 777 groups.

| MiniLM content tokens, without truncation | Child | Parent |
|---|---:|---:|
| Minimum | 3 | 17 |
| Mean | 288.90 | 505.82 |
| Median | 286 | 299 |
| 95th percentile | 513 | 1,651.80 |
| Maximum | 576 | 12,632 |
| Units exceeding the model's 256-token limit, including special tokens | 2,684 (55.05%) | 1,556 (55.87%) |

Counts use the pinned `all-MiniLM-L6-v2` WordPiece tokenizer as a reference consistent with the earlier project. They are not Kanon-token counts and do not assume Part 2 must use MiniLM. Longer-context models require their own tokenizer/limit analysis. A dense index using standard MiniLM truncates many units, including original children. BM25 has no corresponding encoder limit.

## Two ready-to-use corpora

`data/children.jsonl` and `data/parents.jsonl` are already built. Both use passage bodies only, preserve Unicode/Markdown, and retain headings already in the body. Parent text is the ordered child bodies joined with exactly two newlines. Titles and footnotes are metadata; they are not appended to either retrieval text.

Every child retains its original ID and `parent_id`. Every parent has ordered `child_ids` and character offsets in `child_spans`. All 4,876 spans have been checked against the original text. Offsets are Python Unicode-character offsets, not byte offsets.

These are **ID-derived reconstructions**, not recovered original Word documents. The source explains hierarchical splitting and further semantic chunking, but does not publish a formal definition of each ID component. Our grouping interpretation is supported by all-record consistency checks and documented text examples. See HIERARCHY_REPORT.md.

## Run and reproduce

From this folder, using Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m hierarchy.build
python -m unittest discover -s tests -v
```

The builder downloads the pinned source and tokenizer into `.cache/`, then regenerates data and analysis. Existing Part A source files can be reused:

```bash
python -m hierarchy.build --source-dir /path/to/pinned/source/files
```

The directory must contain the original `corpus.jsonl` and `qa.jsonl`. Source SHA-256 checks prevent accidentally mixing revisions. The builder fails on missing child sequences, duplicate IDs, or conflicting parent titles rather than silently producing uncertain text.

## Part 2 handoff

See **PART2_HANDOFF.md** for exact examples and metric definitions. Three modes share the same corpus mappings and queries:

1. **child:** rank and return original children.
2. **parent:** index and rank merged parents directly.
3. **child_to_parent:** rank children, take the first k child hits, expand to their parents, and deduplicate parents in first-hit order. Do not backfill to k unique parents.

The existing Part A 20-development/80-test split is preserved exactly. Three parent IDs contain gold children from both splits; this is a query-held-out comparison, not a parent-disjoint evaluation. All corpus units are searchable. Do not tune on the held-out test results.

Evaluate at k = 1, 5, and 10, and report both unbounded context and a common content-token budget. Parent-hit improvement can reflect returning more text; use original-child evidence coverage after truncation and context size to interpret it. The context helpers preserve provenance so Part 2 need not guess whether the gold text survived.

This package does **not** claim to reproduce the reported 44 → 56 improvement. That requires the original predictions, metric definition, query split, text policy, retrieval parameters, and context budget.

## Files

| File/folder | Purpose |
|---|---|
| `Hierarchy_Analysis.ipynb` | Executed presentation notebook |
| `hierarchy/core.py` | Parse IDs, order children, audit overlap, merge text and preserve offsets |
| `hierarchy/build.py` | Build both corpora, map labels, count tokens, save manifests |
| `hierarchy/handoff.py` | Assemble context and measure original-child evidence after truncation |
| `hierarchy/evaluate.py` | Command-line evaluation of Part 2's saved rankings |
| `hierarchy/io.py` | Pinned source/tokenizer I/O and file helpers |
| `data/` | Two processed corpora, mapped questions, and child/parent mappings |
| `results/` | Token distributions, boundary checks, examples, validation, and manifests |
| `tests/` | Hand-calculated hierarchy and context-budget tests |

## GitHub upload

Upload this entire **legal_rag_hierarchy** folder at the team repository root, preferably on a branch such as `legal-rag-hierarchy`. It can sit beside `legal_rag_bench_a`; no existing team file needs replacing. The included derived corpora carry the source dataset's noncommercial restrictions; see DATA_LICENSE.md before redistribution or sponsor use. Caches and virtual environments are excluded from the ZIP.

Sources: [dataset](https://huggingface.co/datasets/isaacus/legal-rag-bench), [paper section 3](https://arxiv.org/html/2603.01710v1#S3), [MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
