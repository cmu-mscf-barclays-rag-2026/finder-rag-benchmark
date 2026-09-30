# Week 5 — Florence Liu: meeting brief

**Merged parents improve Top-10 gold-evidence hits when we allow more context.
That advantage disappears under our common 2,048-token prefix cutoff.** The
unlimited-context hit increase needs to be interpreted alongside context size.

## Controlled comparison

Kevin's 4,876 children and 2,785 c-parents; his 20-development/80-test split.
All modes share development-selected BM25 k1=1.2, b=0.75. Hybrid uses MiniLM and
weighted RRF with 75% BM25 / 25% dense, selected on development questions.
K counts retrieved units, or child seeds before expansion. A hit here means the
complete annotated child is present in the delivered context.

| Method | Mode | Unlimited Hit@10 | Mean unlimited tokens | 2,048-token Hit@10 | Budgeted gold-token coverage@10 |
|---|---|---:|---:|---:|---:|
| BM25 | Child | 29/80 | 3,823 | 22/80 | 28.2% |
| BM25 | Parent | 37/80 | 10,569 | 21/80 | 28.1% |
| BM25 | Child → parent | 32/80 | 8,956 | 21/80 | 26.3% |
| Hybrid | Child | 33/80 | 3,719 | 23/80 | 30.0% |
| Hybrid | Parent | 49/80 | 9,677 | 22/80 | 27.6% |
| Hybrid | Child → parent | 40/80 | 8,757 | 20/80 | 25.6% |

The unlimited hybrid parent gain is 16 questions, or 20 percentage points, while
returning about 2.6 times as much text. Paired comparison: 17 recoveries and one
regression. Under the cap, parents recover five child misses but lose six child
hits. These small budgeted differences do not establish a robust method ranking.

At K=1, budgeted parent retrieval still improves complete-gold hits: hybrid
15/80 versus child 12/80, and BM25 14/80 versus child 11/80. The finding is
specifically about the Top-10 advantage and this prefix-based context policy;
it does not show that every use of hierarchy is ineffective.

Expansion keeps exactly the same child ranking and deduplicates parents without
backfilling. Its unlimited gain comes from sibling evidence: +3 BM25 and +7 hybrid
questions at K=10. Larger contexts can push useful evidence past the cap.

## Examples to show

- **Q9, witness privilege:** hybrid ranks sibling `2.5-c1-s2` first but misses the
  gold child `2.5-c1-s1` in its Top-10. Expansion returns parent `2.5-c1`, includes
  the gold child, and keeps it within 2,048 tokens. Direct parent retrieval also
  ranks this parent first. This shows a concrete benefit of the hierarchy mapping.
- **Q37, mushroom-poisoning evidence:** hybrid child retrieval places gold
  `4.18-c2-s1` third and retains it under the cap. Direct parent retrieval ranks
  `4.18-c2` seventh; its full Top-10 context is 17,712 tokens. Both direct parents
  and child-to-parent expansion contain the gold without a cap but lose it under
  2,048 tokens. A retrieved parent hit does not guarantee delivered gold evidence.

## Equal versus unequal hybrid weights

These are prespecified sensitivity results on the same test questions. The 75%
weight was selected before test evaluation using unbudgeted development ranking
metrics, not budgeted test performance.

| BM25 weight | Parent unlimited Hit@10 | Parent budgeted Hit@10 |
|---|---:|---:|
| 25% | 41/80 | 23/80 |
| 50% | 46/80 | 22/80 |
| 75% — selected | 49/80 | 22/80 |

## What to say about 44 → 56

“We could not authenticate the original 44 → 56 result from the fetched branch
artifacts. Under the documented matched setup, we measured 29 → 37 for BM25 and
33 → 49 for hybrid at Top-10, out of 80 questions. The original claim remains
unverified until its rankings or script and evaluation settings are available.”

Do not call this a failed reproduction: the original setup is unknown. Do not
change settings on test to try to recover the reported numbers. The branch audit
and required provenance are in [the protocol](../../PROTOCOL.md).

## Limits and next decision

The handoff split shares only 63 test questions with Florence's previous run;
three gold parent groups span development and test. Prior team experiments have
already used these benchmark questions. These are exploratory comparisons, not
a fresh parent-disjoint evaluation. MiniLM truncates 2,684 child and 1,556 parent
encoder inputs at 256 tokens. Coverage concerns annotated text, not legal answer
correctness; no generation or answer judging was performed.

For the next experiment, develop evidence selection within retrieved parents:
keep the seed child and add nearby siblings within a declared token budget.
Choose that policy on development questions and compare against the same child
baseline. A longer-context encoder is another distinct experiment because the
present parent embeddings can omit most of a long parent.

## Artifacts and checks

[Full Hit@1/5/10 table](report.md), [all metrics](summary.csv),
[paired comparisons](paired_comparisons.csv), [examples](examples.csv),
[frozen settings](selected.json), [index sizes](index_audit.csv),
[validation](validation.json).

Validation passed for ten Top-100 ranking files, 7,200 query-level rows and 90
aggregate rows. It checks input/code integrity, ranking IDs, RRF recomputation,
unbounded gold membership, context-budget invariants, aggregate metrics and
development selection. Kevin's ten hierarchy tests and the seven existing Legal
RAG tests also passed. Latency was not measured.
