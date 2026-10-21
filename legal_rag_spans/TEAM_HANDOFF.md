# Using the shared evaluator

The package is independent of the old Qwen folder at runtime. It preserves the earlier canonical parents and mappings. Use its own pinned `data/sources.jsonl.gz` text when making new chunks; the evaluator rejects changed source hashes or inaccurate slices.

```python
import span_bench as s

documents, annotations, tokenizer = s.load()
units = s.make_units(documents, 'grandchild', tokenizer, size=128, overlap=32)
# Index unit['text'] with BM25, dense, hybrid or graph retrieval.
# Keep the unit's source_document, source_sha256, start_char and end_char.

# Replace this empty result with your method's actual ranked source fragments.
ranked = []
returned, token_count = s.apply_budget(ranked[:5], tokenizer, 2048)
scores = s.evaluate(annotations[0], returned, documents, tokenizer)
```

To evaluate another method, save a JSON object mapping every pilot question ID to a ranked list of fragments. A fragment has `source_document`, `source_sha256`, `start_char`, `end_char` and `text`; `id` is optional for scoring. Use `s.fragment(document, start, end, unit_id)` to construct one from canonical text. `results/rankings.json` records baseline coordinates without text to avoid duplicating the corpus; hydrate text from those coordinates before using the external scoring CLI.

```bash
python span_bench.py score --predictions your_ranked_fragments.json --k 5 --budget 2048 --output your_scores.csv
```

After genuine human review, add `--approved-only`. That option currently fails intentionally because no labels have human approval. Missing query predictions fail rather than silently shrinking the evaluation denominator; represent no results with `[]`.

If your pipeline returns already budgeted fragments, call `evaluate()` directly on those fragments instead of applying a second budget. Graph summaries, generated paraphrases, or rewritten contexts need traceable source evidence mappings; do not invent offsets for them. A text-search match in a different document is not automatically the same annotated source occurrence.

Compare methods only on the same question IDs, annotation version, canonical corpus, k, budget and delivery policy. This 20-question draft pilot is not directly comparable to earlier 80-question held-out tables. No Qwen, HyDE, graph or LLM generation rerun is required for Kevin's evaluator deliverable.
