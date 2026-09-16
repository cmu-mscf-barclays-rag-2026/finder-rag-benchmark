# Shared evaluation protocol / 团队统一评估约定

## Person 2's scope

**B: BM25 + MRR/nDCG + keyword-vs-dense analysis.** The additional common metrics
are provided so every team's run can be scored consistently. Person 3 contributes
the final generation/correctness implementation; Person 4 can extend timing and
error analysis. This folder supplies the shared interface and explicit contracts.

## Fixed experimental conditions

1. Use the exact same prepared data fingerprint, query IDs and qrels across methods.
2. Build one corpus from all rows before selecting evaluation questions. Limiting
   the query set must not remove distracting documents or gold references.
3. The searchable corpus contains only normalized reference text. Deduplicate
   identical passages globally, keeping a many-query-to-one-reference mapping.
4. IDs hash normalized, case-preserving passage text. Chunk IDs add character
   start/end offsets. Both corpus membership and qrels are fixed before retrieval.
5. Keep local dev and test query partitions distinct. Tune on dev; evaluate frozen
   configurations on test. The published dataset has one split named `train`.
6. Repeat the same cutoffs (3/5/10) for each method. Each cutoff is actually retrieved
   and timed independently; latency at k=10 is not relabeled as latency at k=3.
7. No query expansion, stemming, stopword filtering or company filtering in the
   default BM25 method. Query terms are counted once; ties sort by document ID.
8. FinDER's reference pool omits full-filing distractors. Report that scope in every
   result. Rankings from full `10-k.zip` retrieval require a separately agreed
   corpus and evidence-to-chunk labeling protocol.

## Metric definitions

For query q, let `R_q` be documents with qrels grade > 0. At cutoff k:

| Metric | Definition |
| --- | --- |
| Precision@k | Relevant documents among first k / k. Missing slots are nonrelevant. |
| Recall@k | Relevant documents among first k / total documents in `R_q`. |
| RR@k | `1 / first relevant rank` if found within k; otherwise 0. Ranks start at 1. |
| MRR@k | Arithmetic mean of RR@k over all evaluated queries, including misses. |
| DCG@k | `sum((2**grade_i - 1) / log2(i + 1))`, for ranks i from 1 through k. |
| nDCG@k | DCG@k / ideal DCG@k. Ideal ranking uses **all qrels**, including gold evidence not retrieved. |
| Evidence coverage@k | Union of retrieved gold-reference character spans / total normalized gold-reference characters. Overlapping spans count once. |
| Answer accuracy (EM) | Conservative Unicode/case/whitespace-normalized exact match; signs, decimals and percentages remain significant. |
| Answer token F1 | Lexical multiset overlap, retaining signed/decimal/percentage number tokens. Not semantic correctness. |
| Answer correctness | Person 3's named/versioned callback in [0, 1]. Unavailable until supplied. |
| Search latency | Elapsed `perf_counter()` time around `retriever.search`, in milliseconds. |

All aggregate ranking scores are macro-averages: each question has equal weight.
FinDER qrels here are binary; the nDCG implementation also accepts nonnegative
graded relevance. Numeric gains are scaled internally for stability, preserving
the DCG/IDCG ratio. Empty rankings score zero. No-positive-qrels cases score zero
in the standalone metrics; data preparation rejects queries without evidence.
Duplicate ranked document IDs are errors, preventing inflated scores. The
standalone `mrr_at_k()` counts missing query results as zero; the shared evaluator
requires an explicit row, with an empty ranking if necessary, for each declared query.

Example: ranks **1, 5, missed** yield MRR@5 = `(1 + 0.2 + 0) / 3 = 0.4`.

### Why nDCG's denominator matters

If a question has two gold passages but retrieves only one at rank 1, nDCG@5 is
`1 / (1 + 1/log2(3)) ≈ 0.6131`, not 1. The unreturned gold evidence still belongs
in the ideal ranking.

### Chunking caveat

Whole-reference mode directly ranks exact annotated evidence passages. In chunk
mode, every chunk of a gold passage inherits grade 1. That measures source
membership, not whether each child contains a complete supporting fact. Smaller
chunks can multiply the number of positive labels. Comparing chunked runs requires
reporting this proxy and evidence character coverage together. Neither exact ID
matching nor character coverage measures semantic claim support.

### Incomplete judgments

Only annotated references are labeled relevant. Other retrieved passages can
contain useful evidence but still receive grade 0. Reports include examples for
manual inspection. Do not change qrels per method after seeing results.

## Teammate run JSON contract

`run.json` carries this shape; obtain a valid fingerprint/query list using
`load_bundle()` and `select_queries()` rather than hand-typing identifiers:

```json
{
  "schema_version": 1,
  "method": "dense",
  "bundle_fingerprint": "the-shared-manifest-fingerprint",
  "partition": "test",
  "query_ids": ["query-id"],
  "top_k": 5,
  "config": {"model": "your-model-and-revision"},
  "latency_scope": "retriever.search_only",
  "timing_protocol": "sequential_one_measurement_per_query",
  "results": [
    {
      "query_id": "query-id",
      "doc_ids": ["shared-corpus-doc-id"],
      "scores": [0.73],
      "latency_ms": 4.2,
      "answer": null
    }
  ]
}
```

`scores`, `latency_ms` and `answer` may be absent; rankings, known unique query IDs
and the fingerprint are required. Scores must be finite and correspond to the
ranked document IDs. Use ranking order as returned, even if models' raw score
scales differ. Exactly one result is required for each declared query; an empty
ranking is legitimate and receives zero ranking scores.

`compare` rejects mismatched data/query fingerprints, cutoffs and partitions. It
does not equate answer scores from different query coverage or differently timed
systems. When using a custom correctness scorer, evaluate each run with that
same callback and version through Python and then assemble the resulting summaries.

## Answer and timing handoff

- Generate each method's answers from its own contexts at that cutoff. Keep the
  generator, prompt, sampling and judge configuration fixed and record them in
  `run['config']`. Gold answers must not enter retrieval or generation prompts.
- Gold answers that are missing/blank are excluded from answer scoring only.
  Retrieval metrics still include those queries. An empty generated answer is a
  real prediction and is scored; absent answers remain unmeasured.
- Report answer scored/eligible counts and compare the same scored query IDs.
- Search latency includes query preprocessing and ranking, excludes index build,
  downloads and generation. Index build is recorded separately. A single timing
  per query is exploratory; Person 4 should specify warmup/repeats, hardware and
  aggregate statistics before making performance claims.

## 合作时可以这样说明

我们统一使用 FinDER，并共享同一个 corpus、query split、qrels 和 evaluation
framework。Person 2 实现 BM25、MRR 和 nDCG；所有 retrieval methods 都调用同一个
`evaluate_rag()`。Answer correctness 由 Person 3 接入统一接口，未测量时显示 N/A。
本实验检索的是所有标注 reference 的去重集合，不能当作完整 10-K corpus 的结果。
