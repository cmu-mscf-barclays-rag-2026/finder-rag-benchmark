# Retrieval evaluation contract

For query q, G is the set of annotated gold passage IDs and R is the ordered first k returned IDs. H is the number of IDs in both. All aggregate scores are arithmetic means of per-query scores.

| Metric | Per-query calculation | Meaning |
|---|---|---|
| Precision@k | H / k | Relevant share of the k available slots |
| Precision among returned | H / len(R), or 0 when empty | Relevant share of actual results |
| Recall@k | H / len(G) | Fraction of annotated evidence found |
| Evidence coverage@k | Same as Recall@k | Explicit evidence-oriented name; not an independent metric |
| All Evidence Hit@k | 1 if G is contained in R, else 0 | Whether every annotated gold passage was returned |
| Hit rate@k | 1 if H > 0, else 0 | Whether any gold passage was returned |
| MRR@k | 1 / first relevant rank, or 0 | Position of the first relevant result |
| nDCG@k | Binary DCG / ideal binary DCG | Ranking quality across relevant results |
| Mean returned | len(R), averaged | Evidence count supplied |
| Empty rate | 1 when R is empty | Frequency of no result |

DCG = sum(relevance_at_rank_i / log2(i+1)), with ranks starting at one. Ideal DCG places min(k, len(G)) relevant IDs first. Binary relevance is used because this dataset has no graded relevance judgments.

## Completeness example

Notebook display names follow the team table. `all_gold_included_at_k` is retained
as the internal/CSV field for All Evidence Hit@k, preserving existing code compatibility.
The screenshot's "All Evidence Recall" refers to the same binary criterion; we use
"Hit" to distinguish it from fractional recall. Evidence Coverage@k is explicitly displayed.

For required gold IDs `{A, B}` and top-three results `[X, A, Y]`, Precision@3 = 1/3, Recall@3 = Coverage@3 = 1/2, Hit@3 = 1, All-gold inclusion@3 = 0, and MRR@3 = 1/2. Finding one passage is a hit, but does not recover all annotated evidence.

In released Legal RAG Bench, len(G) = 1 for every query. Consequently Recall@k = Coverage@k = Hit@k = All-gold inclusion@k, and Precision@k = Hit@k / k. These equalities reflect the labels, not a discovered relationship between independent measures. Rank-sensitive metrics still distinguish whether a hit is first or later.

All-gold inclusion measures **annotated passage inclusion**, not proven answer completeness. If an annotation contains alternative sufficient passages rather than jointly required passages, this metric is too strict. If unannotated passages are useful, exact-ID evaluation can undercount useful retrieval. The current evaluator does not claim span-level or factual completeness.

## Input and comparability rules

- Predictions contain string query IDs and ordered original string passage IDs.
- Every selected query must appear once; `[]` explicitly records an empty retrieval.
- Duplicate passage IDs, foreign passage IDs, missing queries, extra queries, and empty gold labels are rejected.
- Short rankings keep the fixed-k precision denominator; an empty ranking scores zero.
- Save the same split, source revision, corpus/text policy, final k, and ranking depth for each method.
- For fair context comparisons, also consider tokens delivered to generation. Top-k alone is insufficient if chunk lengths differ.
- Do not combine Legal RAG Bench metrics with FinDER rows in the earlier aggregator. They are different benchmark populations and schemas.

## Generation and timing

Reference answers are reserved for generation evaluation. Correctness and groundedness need separate answer/context judgments; retrieval recall cannot substitute for them. No generation scores are produced here.

The new `benchmark_retrievers` utility measures actual retrieval calls with a common
query set, warm-ups, seeded randomized interleaving, and repeated measurements. It
returns raw per-query timings, mean/median/p95 summaries, and first-repeat rankings
for the evaluator. Empty results remain in the measurement; failures abort the run.
For stochastic retrievers, metrics describe first-repeat rankings while timing covers
all repetitions. Set model/cache behavior consistently and record it.

GPU methods can supply synchronization hooks to include completion of asynchronous
work. Adapters must include query encoding, search, fusion/refinement, and evidence
lookup; constructing indexes and generating answers happen outside the timer.
Supply original passage IDs, at most k, without duplicates. The evaluator then checks
corpus membership. Report device and versions; do not compare these single-query
numbers with batched throughput. Reading a saved ranking alone is not retrieval timing.
