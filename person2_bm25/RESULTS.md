# Florence / Person 2 results

## Current shared-split submission

The current team comparison uses 1,128 development and 4,575 test questions under
`sha1-mod5-dev-v1`. The unchanged BM25 implementation was retuned on development
queries before any test evaluation; the selected settings are **k1=1.6, b=1.0**.

| Method | Test questions | P@5 | R@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default BM25 (1.2 / 0.75) | 4575 | 0.0635 | 0.3022 | 0.2297 | 0.2449 |
| Dev-selected BM25 (1.6 / 1.0) | 4575 | 0.0655 | 0.3112 | 0.2340 | 0.2497 |

See [the complete common-split report](../results/b_bm25_common/report.md),
[dev sweep](../results/b_bm25_common/dev_sweep.csv),
[verification](../results/b_bm25_common/validation.json), and
[team CSV](../team_metrics/b_bm25.csv). All four cutoffs were independently
recomputed with the existing shared metric function from saved top-10 rankings;
the maximum absolute difference was zero.

The older seed-42 results below are retained for provenance and must not be mixed
with the current shared-split table. Their selected parameters need not agree
because the development query sets differ. Answer generation remains unmeasured.

---

# Historical seed-42 results

These are measured local experiments, not example scores. The synthetic demo is excluded.

## Dataset and protocol

- Dataset: [Linq-AI-Research/FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER).
- Pinned revision: `c4c1b6454aef7f0bb1c37235c7f52ce644642da0`; Parquet SHA-256 verified before loading.
- All 5,703 queries supply 5,830 unique normalized reference passages.
- Local seed-42 partition: 1,140 development queries and 4,563 test queries.
- Whole-reference mode, no chunking; binary relevance from exact reference membership.
- Every run indexes the same complete reference pool. No queries or benchmark answers are indexed.
- Data fingerprint: `a3889050fc1af61491d79cbefb8c9b53ac89e32f2ad564312ad8963a6e983e11`.
- Test-query fingerprint: `3401dba8687cb3ab6250bab508f1208ddcdae08fba913db2a07fdceab1bd00ff`.
- Runtime: Python 3.9.4; macOS-26.5.2-x86_64-i386-64bit (x86_64).

**Scope:** this is a pooled annotated-reference benchmark. The full `10-k.zip` corpus is not indexed. These scores are not comparable to the paper’s full-filing/RAGAS results or a differently prepared team corpus.

## Default BM25: k1 = 1.2, b = 0.75

| Method | k | Queries | P@k | R@k | MRR@k | nDCG@k | Evidence coverage@k | Answer EM | Answer correctness | Mean search ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 3 | 4563 | 0.0906 | 0.2609 | 0.2204 | 0.2277 | 0.2618 | N/A | N/A | 7.5762 |
| bm25 | 5 | 4563 | 0.0636 | 0.3029 | 0.2300 | 0.2453 | 0.3042 | N/A | N/A | 7.5415 |
| bm25 | 10 | 4563 | 0.0378 | 0.3588 | 0.2376 | 0.2636 | 0.3601 | N/A | N/A | 7.8597 |

Scores are in [0, 1]; higher ranking scores are better. Latency is one search measurement per query, excluding index build, downloads and generation. Timing differences are exploratory.

Increasing k from 3 to 10 retrieves more gold evidence, while Precision@k falls because its denominator grows. MRR improves less than recall: additional passages often appear lower in the ranking.

## Parameter selection on development queries

Nine configurations were tested on all 1,140 development queries at k=5. Selection maximized dev nDCG@5, with MRR@5 breaking ties.

| k1 | b | Dev MRR@5 | Dev nDCG@5 |
| --- | --- | --- | --- |
| 0.8 | 0.25 | 0.1861 | 0.2001 |
| 0.8 | 0.75 | 0.2236 | 0.2365 |
| 0.8 | 1.0 | 0.2295 | 0.2426 |
| 1.2 | 0.25 | 0.1837 | 0.1985 |
| 1.2 | 0.75 | 0.2250 | 0.2385 |
| 1.2 | 1.0 | 0.2307 | 0.2437 |
| 1.6 | 0.25 | 0.1854 | 0.1987 |
| 1.6 | 0.75 | 0.2240 | 0.2374 |
| 1.6 | 1.0 | 0.2287 | 0.2424 |

Selected **k1=1.2, b=1.0** on dev; dev nDCG@5 = **0.2437**.

## Frozen configuration evaluated on test

| Method | k | Queries | P@k | R@k | MRR@k | nDCG@k | Evidence coverage@k | Answer EM | Answer correctness | Mean search ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 5 | 4563 | 0.0636 | 0.3029 | 0.2300 | 0.2453 | 0.3042 | N/A | N/A | 7.5415 |
| bm25_tuned | 5 | 4563 | 0.0656 | 0.3111 | 0.2361 | 0.2516 | 0.3117 | N/A | N/A | 8.1508 |

Tuning changes test MRR@5 by +0.0061 and nDCG@5 by +0.0063. This is a descriptive difference; no statistical significance claim is made.

The comparison command recomputed both methods with the same evaluator and verified identical data/query fingerprints and cutoff. See [paired comparison](results/default_vs_tuned/comparison.md).

## What remains for team integration

- Dense, hybrid and MMR rankings must come from their owners using this exact prepared bundle. No scores for those methods have been invented.
- Generated answers and Person 3’s shared correctness evaluator have not been run. All answer-quality values are N/A. Six test queries lack nonempty gold answers, so 4,557 test queries are eligible for future answer scoring; all 4,563 remain in retrieval scoring.
- The [analysis](ANALYSIS.md) includes measured BM25 error examples and a paired sparse-vs-dense workflow.

## Verification

- 21 offline tests passed: hand-computed BM25, MRR and graded nDCG; misses; empty rankings; duplicates; precision denominator; stable IDs; partitions; overlap coverage; answer hooks; and run validation.
- The walkthrough notebook’s Python cells executed successfully on the synthetic offline fixture.
- Real-data preparation, default top-k runs, nine dev configurations, tuned test run and compatible-run comparison completed.

## Reproduce

```bash
python -m finder_bm25 prepare
python -m finder_bm25 run --top-k 3 5 10
python -m finder_bm25 sweep --top-k 5
python -m finder_bm25 run --k1 1.2 --b 1.0 --top-k 5 --method bm25_tuned --output results/bm25_tuned
python -m finder_bm25 compare --runs results/bm25/k5/run.json results/bm25_tuned/k5/run.json --output results/default_vs_tuned
python -m unittest discover -s tests -v
```

Activate the environment and run these commands from `person2_bm25/`. The [README](README.md) includes macOS/Linux and Windows setup.
