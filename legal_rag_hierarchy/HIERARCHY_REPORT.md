# Hierarchy interpretation and audit

## What is known and what is inferred

The authors describe hierarchy-aware conversion of Criminal Charge Book Word sections into Markdown, followed by semantic chunking where needed. They do not provide a formal public decoding specification for `c` and `s` in the inspected card/paper. We therefore use operational names: section prefix, c-parent, and s-child. The package does not assert that `c` literally means chapter or `s` literally means sentence.

`1.1-c1-s1` maps to c-parent `1.1-c1` and broader section `1.1`. Removing both suffixes, as in the earlier descriptive section-length analysis, would instead merge every c-parent in that section. This package deliberately keeps c-parent boundaries.

## All-record checks

All 4,876 IDs parse. They form 2,785 c-parents and 724 section prefixes. Each c-parent has one consistent title and a child sequence starting at 1 with no gaps. No groups require reordering from source order, but the builder always sorts numerically so future file order cannot alter the result. Children numbered 10 or higher are not sorted lexicographically.

Every merged parent preserves every child body exactly, separated by two newlines. All child offsets round-trip. No headings, short update notices, duplicate text, or overlapping text are removed. Footnote variants retain their source child IDs as metadata. Retrieval text excludes those footnote fields in both variants.

The audit checks 2,091 adjacent boundaries for exact character suffix/prefix overlap and case-normalized whitespace-word suffix/prefix overlap. None has word overlap; no boundary meets the review threshold of 30 exact characters or five words. This is a mechanical boundary check, not proof of no paraphrased or nonadjacent repetition. The full audit is in `results/boundary_overlap.csv`.

## Three inspected examples

1. **1.1-c1, Introductory Remarks:** two children. The first includes the opening heading and numbered introductory material; the second resumes at item 4. This supports ordered concatenation.
2. **1.2-c2, Excusing Jurors:** four children. The first is a heading, the next two carry numbered content, and the last is an update notice. Keeping all four preserves the source structure. Retrieving only the heading is not the same as retrieving the gold substantive child, but expanding it can recover that child.
3. **4.12-c7, identification-evidence warning:** 39 children, the largest group by child count. Later children introduce nested headings, so a c-parent is not necessarily one short topic paragraph. Its large size makes encoder/context truncation important.

These examples support the ID-based grouping, but are not verification against the original Word files. We describe the outputs as reconstructed groups of released passages.

## Token and context interpretation

MiniLM content-token counts average 288.90 for children and 505.82 for parents. The largest parent is 12,632 tokens. Counts are measured without truncation, with special-token-inclusive lengths separately tested against the model's configured 256-token limit. The 512-token bound described by the dataset authors uses a different tokenizer and does not contradict these measurements.

Direct parent embedding with MiniLM cannot represent an entire long parent under the model's default limit. Child retrieval followed by parent expansion avoids embedding the whole parent, but still increases the text returned and does not eliminate truncation within long child inputs. Part 2 should report the embedding truncation policy separately from the final context budget.

## Evaluation interpretation

Original gold child IDs are retained. Their parent IDs are derived by the same mapping used to construct the corpus. No new relevant passages are invented. Parent membership and inclusion of the complete gold text after truncation are separately measurable.

Gold-body token coverage concerns the proportion of annotated passage text retained. It does not measure the proportion of legal reasoning steps or answer facts covered. The source still has only one annotated child per query.
