# Week 5 — Florence Liu: hierarchical retrieval protocol

Person 1 (Kevin) builds and audits the hierarchy. Person 2 compares retrieval on
that hierarchy. Person 3 owns reference-answer retrieval and question-only HyDE.
Answer generation and GraphRAG are outside this experiment.

## Kevin's handoff

Fetched `origin/finder-threshold-mmr` at
`ed72cf8473dea5df38e56ee1f62ece325bebb788` and imported its complete
`legal_rag_hierarchy/` directory without changing its files. That snapshot now lives
at `week5_hierarchy/kevin_hierarchy/` on Florence's `florenceliu` branch.
Kevin's branch is unchanged; this is a directory import, not a branch merge.

Read [PART2_HANDOFF.md](kevin_hierarchy/PART2_HANDOFF.md) first.
The supplied `children.jsonl` has 4,876 passages; `parents.jsonl` has 2,785 groups.
Remove only the final `-sN` suffix: `1.1-c1-s1` belongs to `1.1-c1`.
Do not merge all of section `1.1`. Parent bodies preserve numerically ordered
children separated by two newlines, including the original headings.

## Experiment and deliverables

The runner compares these modes using the same questions and retrieval settings:

1. **Child:** retrieve and deliver original children.
2. **Parent:** retrieve and deliver merged parent bodies.
3. **Child → parent:** reuse the exact child rankings, expand the first K seeds,
   deduplicate parents in first-hit order, and do not backfill to K unique parents.

BM25 uses Florence's existing tokenizer/scorer. Dense retrieval uses the pinned
MiniLM revision, normalized vectors and exact cosine search. Both indexes in a
hybrid run use the same unit type. Weighted reciprocal rank fusion uses 100
candidates per component and constant 60. Retrieval input is only the question.

Select one shared BM25 configuration from k1={0.8,1.2,1.6}, b={0.25,0.75,1.0}
using the mean child/parent development nDCG@5, then MRR@5, then Hit@5, then grid
order. This fixes parameters across representations. Select a shared hybrid
BM25 weight from {0.25,0.5,0.75} using the same rule and selected BM25 settings.
Save settings before test retrieval. Report all three weights as sensitivity
results; do not select a weight from test performance.

Report K={1,5,10} with both unbounded context and a 2,048-content-token ceiling.
Save ranked-unit hits separately from complete gold-child hits in delivered
context, gold-body token coverage, and mean/p95/min/max returned tokens.
Expansion can include the gold through a sibling without improving child ranks.
Partial gold text under a budget earns partial coverage but no complete-child hit.
These measurements do not establish answer correctness.

MiniLM's 256-token input truncation and the final context ceiling are distinct.
The encoder sees only the beginning of long children/parents. BM25 sees full
bodies. Record truncation in `index_audit.csv`; do not describe the dense parent
vector as representing every word of a long parent.

## Split mismatch discovered during integration

Kevin's `legal-rag-a-sha256-gold-group-20pct-v1` split contains 20 development
and 80 test questions. Florence's initial split also has 20/80, but only **63 test
questions** and **3 development questions** overlap. Preserve Kevin's split for
the new comparison. Do not combine these scores with the previous 30/80 BM25 or
32/80 hybrid figures as if they were evaluated on the same questions.

Kevin's gold parents `7.1.3-c1`, `7.4.9-c5`, and `7.6.1-c3` span dev and test.
The test questions have also been used in earlier team experiments. This is an
exploratory within-benchmark comparison, not a fresh unseen or parent-disjoint test.

## Status of the 44 → 56 claim

The September 23 agenda records a reported Top-10 improvement but leaves its metric,
denominator, split and merging setup unconfirmed. Kevin explicitly says his handoff
does not reproduce it. A search of Legal RAG code, reports and notebooks at the
fetched branch tips found no saved child-versus-parent run establishing those counts:

| Branch | Inspected commit |
|---|---|
| main | `60e22956f3e82b3602d05c4a5f4c5a0cb21c0044` |
| feature/person2-bm25 | `ce54a012f1c9112665b8d9ae42d7a73903c4052e` |
| feature/person3-hybrid | `a25b063270ecd57151f01faaa49ff3aad2b41b1d` |
| finder-threshold-mmr | `ed72cf8473dea5df38e56ee1f62ece325bebb788` |
| yuchenlu | `6c88bbc2e060b1543eaee6fe1c91c4ea6b59d5c1` |

Cheryl's saved full-100-question Top-10 results are 37 BM25, 37 dense, 46 fixed
hybrid and 45 out-of-fold hybrid hits; they are not the claimed parent comparison.
Yuchen's saved full-100-question dense/graph Top-10 hits are 38/36, also a different
experiment. The new experiment can test the hierarchy hypothesis but cannot
authenticate an undocumented historical result merely by finding similar numbers.

To close the historical reproduction item, obtain the original child and merged
rankings or runnable script plus split, model, text policy, grouping, parameters,
hit definition, and context budget. Until then report **unverified**, not disproven.

## Reproduce

From the repository root, with Python 3.12:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r week5_hierarchy/requirements-lock.txt
PYTHONPATH=week5_hierarchy/kevin_hierarchy .venv/bin/python -m unittest discover -s week5_hierarchy/kevin_hierarchy/tests -v
.venv/bin/python -m week5_hierarchy.hierarchy_experiment --output week5_hierarchy/results/comparison
.venv/bin/python -m week5_hierarchy.validate_hierarchy --results week5_hierarchy/results/comparison
```

Use a new output directory for each run. After the model is cached, add `--offline`.
The runner accepts `--model-cache` and `--embedding-cache`. It verifies Kevin's six
processed-data hashes, normalizing Windows manifest paths without modifying the
upstream manifest, and checks every parent/child character span.

Bring `report.md` to the meeting; retain `summary.csv`, `per_query.csv`, saved
top-100 rankings, `dev_sweep.csv`, `selected.json`, `index_audit.csv`, `split.json`,
and `manifest.json` so teammates can audit or reuse the run. Latency is not measured
by this quality/context-size experiment and must not be inferred from its runtime.
The validator independently checks unbounded evidence membership, recomputes all
aggregate means, verifies selection from the saved development sweep, and writes
`validation.json`, paired win/loss counts, and concrete recovery/truncation examples.
