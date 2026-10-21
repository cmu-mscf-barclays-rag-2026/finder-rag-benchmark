# Exact supporting-span annotation protocol

## What is being annotated

Annotate source evidence sufficient to support the supplied reference answer, not the answer string itself. An answer may be a paraphrase. Preserve conditions, exceptions, negation, dates, offence names and requested citations. Avoid selecting a whole paragraph merely because it is already a gold chunk; avoid isolated keywords that omit the proposition.

The pilot uses all 20 inherited development questions, chosen without examining the new retrieval outcomes. The 80 test questions are not evaluated or annotated here. Development results are exploratory, not a new held-out benchmark.

## Coordinate contract

`source_document` identifies a reconstructed parent from the earlier hierarchy package, **not an original PDF or a whole original legal document**. Each annotation uses zero-based, half-open `[start_char, end_char)` Python Unicode character offsets in the exact parent `text`. It stores the source SHA-256 and the verbatim text slice. Do not normalize whitespace, Markdown, Unicode or line endings. Offsets are not bytes, browser UTF-16 offsets or token indices.

Child and grandchild chunks carry these same coordinates. Rechunking does not change the gold evidence definition. Changing source reconstruction does require remapping and revalidation. Footnotes remain source metadata and are not indexed or annotated in this pilot.

## Review process

1. Read the full question and reference answer, then the original gold child and its surrounding parent.
2. Check the proposed verbatim span(s) in `REVIEW_SHEET.md`. Check the rationale and flagged ambiguity.
3. Verify each clause supports an explicit part of the requested answer. Retain necessary qualifications. Do not invent additional required passages from the single-gold label.
4. Check whether another source occurrence could also be sufficient. The pilot currently scores one chosen evidence set; it does not exhaustively annotate acceptable alternatives.
5. If needed, edit `data/annotations.json`: update the exact offsets and text together, leaving the source hash unchanged. Run `python span_bench.py validate`. Structural validation is necessary but is not semantic approval.
6. A real reviewer should record their name, review date and decision in the annotation record. Set `review_status` to `human_approved` only after that person has checked it. Record unresolved cases as `needs_adjudication`. For disputed cases, use a second reviewer and record the resolution.
7. Rerun evaluation after annotation edits and regenerate the report. Archive annotation versions/hashes with results. Do not improve labels selectively after observing which method wins.

## Current status

20 questions and 27 spans were selected and read by the assistant, and checked against exact source slices by code. **Zero questions have independent human approval.** All included retrieval results use draft annotations. The `--approved-only` scoring option refuses to run when no approved questions exist.

Some questions have several supporting clauses inside their one official gold passage. This is not validation of genuinely multi-document reasoning. The evaluator supports multiple source documents mechanically, but this pilot has no independently verified multi-document questions.

Priority review: Q48's `could get hurt` wording versus the source's probability requirement; Q25's classification; Q58's historical applicability. Q89 and Q90 share evidence, so the 20 observations are not fully independent. Read this as benchmark annotation, not current legal advice.
