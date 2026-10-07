# Week 6 — Florence Liu: Qwen hierarchy and context budgets

**Question:** Does hierarchical retrieval improve with Qwen3-Embedding-0.6B,
and can its benefit survive a realistic context budget?

This continues [Week 5](../week5_hierarchy/README.md). MiniLM represented inputs
using at most 256 tokens. Hybrid parent retrieval achieved 49/80 Hit@10 versus
33/80 for children with unlimited context, but only 22/80 versus 23/80 under a
2,048-token prefix cutoff. Week 6 separates document encoding from context delivery.

**Result:** Qwen makes dense retrieval much stronger, but it does not preserve a
hierarchy advantage under the 2,048-token delivery budget. At the development-selected
4k embedding window, dense child retrieval achieves 44/80 Hit@10, compared with
36/80 for direct parents, 33/80 for child → parent expansion, and 42/80 for selected
siblings. With unlimited delivery, direct parents and expansion rise to 56/80 and
57/80 versus 53/80 for children. See the [full findings](results/qwen/findings.md).

The embedding-window sweep reaches 6/20 development Hit@10 at 2k and 8/20 at 4k,
8k, and 16k under hybrid direct-parent retrieval with the 2,048-token delivery
budget. The registered tie-break therefore selects 4k. Qwen dense child retrieval
also improves sharply over MiniLM dense under the shared delivery budget, from
24/80 to 44/80 Hit@10. The frozen Week 5 hybrid weighting underperforms Qwen dense,
so it should not be read as a Qwen-tuned fusion result.

The [reference results](results/reference/findings.md) reproduce all 36 comparable
Week 5 rows. The new fixed sibling policy achieves 24/80 hybrid Hit@10 under
2,048 tokens, versus 20/80 for full-parent expansion and 23/80 for children.
These are MiniLM reference results, not Qwen findings; the one-hit advantage over
children is descriptive and does not establish a reliable gain.

## Frozen protocol

- Kevin's checksum-verified hierarchy: 4,876 children, 2,785 parents, 20 development
  and 80 test questions. Original questions only; no HyDE or reference-answer queries.
- BM25 k1=1.2 and b=0.75; weighted RRF c=60, BM25 weight=0.75, candidate depth=100.
  These are the saved Week 5 choices, not retuned for Qwen.
- Dense dot-product retrieval over normalized vectors; ties by document ID.
- Qwen document windows: 2,048 / 4,096 / 8,192 / 16,384 tokens; optional 32,768.
  The original question gets a single fixed retrieval instruction and a fixed
  2,048-token query window. The runner rejects truncated questions.
- Child, direct parent, child → parent, and child → selected siblings, each with
  unlimited and 2,048-token delivered context. Embedding window and delivery budget
  are independent. BM25 has no embedding window.
- Sweep Qwen windows on **development only**. Select on hybrid direct-parent
  budgeted Hit@10, then coverage, mean coverage efficiency, then smaller window.
  Write `selected.json` before encoding/ranking Qwen test queries. Evaluate all
  methods/modes at that one selected window on the test split.
- Reuse saved MiniLM and BM25 test rankings. Recompute their context metrics with
  the same evaluator, including the new sibling policy. Verify newly computed
  BM25 test rankings exactly match the saved reference during a Qwen run.

The [configuration](config/experiment.json) pins the Qwen revision and measurement
tokenizer. [Qwen's official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
documents its 32k context, last-token pooling, query instructions and Transformers
version requirement. The implementation uses `AutoModel`, last nonpadding token
pooling with left padding, and L2 normalization. It does not use a chat template.
Documents receive no instruction. Right truncation includes any special tokens.

## Context policy and metrics

The original three modes use the unchanged Week 5 context assembler: concatenate
ranked units, then apply an exact token-budgeted prefix cutoff. Expansion counts
K child seeds, deduplicates their parents in first-hit order, and does not backfill.

Selected siblings visit each of the first K seeds followed by its immediately
previous and next sibling in parent span order. Candidates are deduplicated in
first-visit order. A candidate is added whole if the concatenated context fits;
oversized candidates are skipped and subsequent candidates are considered.
This policy can skip a seed and use space for its neighbors. It never consults
gold labels, adds farther neighbors, or retrieves additional seeds. K therefore
counts seeds, not the final number of returned chunks. Unlimited sibling delivery
still includes only those local candidates, not entire parents.

**Every model uses the pinned, untruncated Week 5 MiniLM tokenizer for context
budgets and gold-token coverage.** Qwen's own tokenizer measures encoder windows
and truncation. This preserves an identical evaluation ruler across models;
2,048 here is a common body-text proxy, not an unspecified generator's exact prompt
limit. Separators count toward context size; instructions and prompt wrappers do not.

- Hit@5 / Hit@10: at least one complete original gold child is in delivered text.
- Gold-text coverage: fraction of original gold-body tokens visible in returned
  text, including partial evidence under prefix clipping.
- Mean / P95 context tokens: delivered body text under the fixed tokenizer.
- Coverage efficiency: per-query coverage fraction divided by context tokens;
  zero for empty context. Aggregate with the arithmetic mean of per-query ratios,
  not the ratio of aggregate means. Tables scale this by 1,000 for readability.

A MiniLM–Qwen difference does **not** isolate window size: training, model capacity,
tokenization and query instruction also change. The within-Qwen sweep tests window
size with those factors held fixed. If the token audit shows all parents fit at a
window, larger windows provide identical model inputs and cannot test dilution
beyond the corpus's actual length. The existing split is query-held-out, not
parent-disjoint (three gold parents occur in both splits). No generated-answer or
latency claims are made.

The pinned Qwen tokenizer audit finds a maximum of 611 tokens per child and
14,463 per parent (including special tokens). Of 2,785 parents, 109 exceed 2k,
23 exceed 4k, two exceed 8k, and none exceed 16k. Consequently the optional 32k
setting cannot add document content in this particular hierarchy.

## Run

Use a separate environment so Week 5 remains reproducible:

```sh
python3 -m venv .venv-week6
.venv-week6/bin/python -m pip install -r week6_qwen_hierarchy/requirements.txt
.venv-week6/bin/python -m week6_qwen_hierarchy.qwen_hierarchy \
  --output week6_qwen_hierarchy/results/qwen
```

`requirements-lock.txt` records the exact runtime package versions used locally;
use it instead of `requirements.txt` when recreating that environment on a
compatible platform. The tests also run in the existing Week 5 environment.

The first full run downloads about 1.2 GB of Qwen weights. CPU float32 is the
portable default. For an available CUDA accelerator, add `--device cuda --dtype
bfloat16`; use `--device mps` for supported Apple acceleration. Add `--include-32k`
only when feasible. `--batch-size` and `--batch-tokens` bound padded batches;
a single long input may exceed the batch-token target. Reduce batch size for
memory pressure. No silent window reduction or dataset subsampling occurs.

Embeddings are cached by the exact token IDs seen by the encoder, model revision,
pooling, precision, device and package versions. Identical inputs reuse vectors
across representations/windows. Each completed batch is saved atomically. After
interruption, rerun into a **new output directory**, reusing the cache. A nonempty
output is never overwritten. A run's manifest stays `running` until all outputs
are written, so partial results cannot be mistaken for a completed experiment.

To reproduce the reference without downloading Qwen:

```sh
.venv/bin/python -m week6_qwen_hierarchy.qwen_hierarchy \
  --reference-only --offline --output week6_qwen_hierarchy/results/reference_repeat
.venv/bin/python -m unittest discover -s week6_qwen_hierarchy/tests -v
```

Full outputs include `window_sweep.csv` (development), `hierarchy_comparison.csv`
(test), `per_query.csv` (including returned IDs), `examples.csv` (sibling wins and
losses versus full-parent expansion), `paired_comparisons.csv` (paired Hit@10
wins/losses and exploratory exact McNemar tests), `index_audit.csv`, `findings.md`,
saved rankings, split, effective config, selected window, and provenance manifest.
Reference-only output has an empty `window_sweep.csv` with headers because no
Qwen sweep ran. Generated numbers are never placeholders for unrun Qwen settings.

HyDE and Graph RAG remain outside this experiment. Once a Qwen configuration is
established, Cheryl's query method can be tested as a separate follow-up factor.
