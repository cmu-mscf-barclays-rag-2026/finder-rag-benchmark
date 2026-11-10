# Question-aware evidence prototype

Offline Legal20 experiment using saved Question, Focused-mix and Short-mix rankings. A frozen AI-assisted annotation draft identifies rules needed for each question and links them to exact source quotations and character offsets. The evaluator reports quotation availability under identical ranked-context token budgets, with AND/OR logic for complementary facts and alternative routes.

This is an engineering prototype. It is **not** a human-validated benchmark, legal-correctness evaluation, semantic fact-recall measurement, or reader-accuracy experiment. Human review remains outstanding. Query92 is retained but excluded from aggregates because its released answer contradicts the question; aggregate n=19.

Run from the repository root:

```bash
.venv/bin/python person3_meeting9_evidence/run.py
.venv/bin/python person3_meeting9_evidence/verify.py
```

The runner needs the original local Legal corpus, saved `person3_prompt_topk` rankings and pinned MiniLM tokenizer cache. `verify.py` checks cached artifacts without executing a tokenizer/model/API. Frozen files cannot silently change. No API, new embeddings, or historical full-run generation is executed.

Results: `results/person3_meeting9_evidence/legal20_v1/`.
Readable report and chart: `../outputs/meeting9_20261109/evidence/`.
The `annotation_draft.py` source contains the source-grounded draft and issue notes; `human_review_template.csv` is intentionally pending, not completed human annotation.
