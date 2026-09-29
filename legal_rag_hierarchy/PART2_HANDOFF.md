# Part 2: comparing child, parent, and parent-expansion retrieval

## Load the data

```python
from hierarchy.io import ROOT, read_jsonl, load_tokenizer
from hierarchy.handoff import assemble_context, evaluate_run

children = {r['id']: r for r in read_jsonl(ROOT / 'data/children.jsonl')}
parents = {r['id']: r for r in read_jsonl(ROOT / 'data/parents.jsonl')}
questions = read_jsonl(ROOT / 'data/queries.jsonl')
test_questions = [q for q in questions if q['split'] == 'test']
tokenizer, tokenizer_metadata = load_tokenizer()

# For a child index:
child_ids = list(children)
child_texts = [children[pid]['text'] for pid in child_ids]
# For a parent index:
parent_ids = list(parents)
parent_texts = [parents[pid]['text'] for pid in parent_ids]
```

Use only `question` as the retrieval input. `answer`, `gold_ids`, and gold parent/child mappings are evaluation labels. Preserve list alignment between texts and original IDs when indexing; do not return integer row positions.

`gold_ids` is an alias for original gold child IDs, so the Part A evaluator still works for child rankings. Parent rankings must use mapped parent labels or the hierarchy evaluator; never directly compare parent IDs against child IDs.

## Retrieval experiments

Run BM25 on both corpora with the same tokenizer and BM25 parameters. For hybrid, construct both BM25 and dense indexes over the same unit type, apply the same embedding model and fusion rule/weights, and record any dense input truncation. A parent BM25 index paired with a child dense index is a different experiment and requires an explicit fusion mapping.

For expansion, reuse exactly the child predictions from the child experiment. At k child seeds, expand to their parents and deduplicate in first-hit order. Two child hits can produce one parent. This package does not fetch extra children to fill k unique parents; that would change the seed budget.

Save rankings with one row per selected query:

```json
{"query_id":"1","passage_ids":["1.2-c2-s2","1.1-c1-s1"]}
```

For direct parent retrieval, use parent IDs such as `1.2-c2`. This example shows a schema, not actual retrieval output.

```bash
python -m hierarchy.evaluate --predictions child_predictions.jsonl --method bm25 --mode child --split test --out results/local_bm25_child
python -m hierarchy.evaluate --predictions parent_predictions.jsonl --method bm25 --mode parent --split test --out results/local_bm25_parent
python -m hierarchy.evaluate --predictions child_predictions.jsonl --method bm25 --mode child_to_parent --split test --out results/local_bm25_expanded
```

Repeat with `--token-budget 2048` for a common delivered-context budget, then with hybrid predictions. This budget uses the supplied MiniLM tokenizer as a common measurement unit; it is not a generation model's context-window promise. It excludes prompt instructions, the query, and special tokens. If generation uses another tokenizer, measure its full formatted prompt separately.

## What each metric means

| Field | Meaning |
|---|---|
| `ranked_unit_hit_at_k` | Child/expansion: original gold child in the top k seeds. Parent: gold parent in top k parent hits. These rank different units. |
| `ranked_unit_recall_at_k` | Fraction of corresponding gold units in the top k seeds |
| `gold_child_in_selected_units` | A selected body contains a gold child before context truncation |
| `gold_child_hit_in_context` | At least one complete original gold child survives in returned text |
| `gold_child_coverage_in_context` | Fraction of original gold children fully retained |
| `all_gold_children_in_context` | All original gold children fully retained |
| `gold_body_token_coverage` | Fraction of tokens in the annotated child bodies whose character spans survive; not answer-fact completeness |
| `returned_units` | Number of bodies after parent deduplication |
| `returned_content_tokens` | Actual token count of assembled/truncated body context |
| `context_truncated` | Whether the token budget removed any selected text |

All aggregates are macro means over queries. The token-coverage denominator is the annotated passage body, because the benchmark supplies no minimal answer-span labels. A partially retained gold passage may or may not contain the answer; the package does not infer that.

The function returns the exact assembled text and provenance. If later prompt formatting, another truncation step, or selective excerpting changes that text, coverage must be recalculated for the actual delivered context. This module measures evidence inclusion, not answer correctness or groundedness.

## Recommended comparison table

Include method, retrieval mode, k, query split, token budget, ranked-unit hit, gold-child hit after context construction, gold-body token coverage, mean/p95 context tokens, and latency. Per-query output supports additional context-size summaries. Run timing using Part A's shared utility around the complete retrieval plus expansion/lookup path, and record whether query encoding is included.

For the 44 → 56 claim, first identify what 44 and 56 count, the denominator, k, whether parent membership counts as a hit, and whether the budget is fixed. Reproduce the original setup before changing parameters. A larger parent can recover the gold child through a sibling hit without improving the original child ranking.

## Split caveat

The existing 20/80 split is preserved for comparability. Gold parent IDs `7.1.3-c1`, `7.4.9-c5`, and `7.6.1-c3` occur on both sides. Parameter tuning therefore is not parent-disjoint. If the team chooses a new parent-group split, all methods must rerun under a new split ID; do not silently mix old and new results.
