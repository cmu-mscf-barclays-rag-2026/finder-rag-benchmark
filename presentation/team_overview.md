# Four-person contribution overview

This snapshot summarizes all four uploaded workstreams. The code remains on the
owners' development branches; shared result snapshots are integrated on `main`.
Source commits are pinned in [provenance.json](../team_metrics/provenance.json).
No retrieval or generation experiment was rerun for this update.

## Contributions

| Owner | Work delivered | Branch | Detailed source |
| --- | --- | --- | --- |
| A / Yuchen | Dense MiniLM, chunk-size/overlap sweep, source-passage pooling, held-out evaluation, app and tests | [`yuchenlu`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/yuchenlu) | [Task A report](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/blob/bd5e7c35443c6eaa5a1f24ada240b2c5ffdb3eb1/results/task_a_report.md) |
| B / Person 2 | BM25 inverted index, MRR/nDCG, common evaluator, tuning, failure analysis, notebook, 21 offline tests | [`feature/person2-bm25`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person2-bm25) | [Verified results](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/blob/c27c30d11f804696c6d6969e6b7514118a15fb02/person2_bm25/RESULTS.md) |
| C / Cheryl | BM25/dense controls, weighted hybrid RRF, dev tuning, pilot answer generation, shared benchmark tools | [`feature/person3-hybrid`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person3-hybrid) | [Part C write-up](../docs/PART_C_HYBRID.md) |
| D / Kevin | Threshold/MMR tuning, paired evidence-error analysis, repeated CPU latency, held-out evaluation | [`finder-threshold-mmr`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/finder-threshold-mmr) | [Refinement report](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/blob/4013a25b33a7dc62845359941f062870c822f91a/D_RESULTS.md) |

## Common split: A, C, D

All three use 5,830 normalized reference passages and the SHA-1/modulo-5 query
split: 1,128 development and 4,575 test questions. A embeds chunks and max-pools
scores to distinct source passages, retaining passage-level relevance labels.
A's dataset-content fingerprint was recomputed from the shared Parquet and matched
its saved run manifest. C/D use the same shared implementation and configuration.

| Owner | Selected method | Precision@5 | Recall@5 | MRR@5 | nDCG@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Dense MiniLM, chunk 500 / overlap 75 | 0.04791 | 0.22840 | N/A | N/A |
| C | Hybrid RRF, dense weight 0.25 | 0.06435 | 0.30561 | 0.22051 | 0.23862 |
| D | Threshold 0.3 | 0.04503 | 0.21169 | 0.15833 | 0.16885 |
| D | MMR lambda 1.0 / fetch 10 | 0.04507 | 0.21177 | 0.15833 | 0.16889 |

Among these submitted configurations, C has the highest Recall@5. This is not an
all-four ranking, and A's MRR/nDCG were not reported. The selected D MMR disables
the diversity penalty; neither selected refinement improves the D dense control.
See the [generated shared comparison](presentation_summary.md) for exact CSV-backed
values and [comparison_k5.csv](comparison_k5.csv) for the machine-readable table.

## Separate split: B / BM25

B uses a seed-42 partition of 1,140 development and 4,563 test queries. These are
valid within-B results, but cannot be ranked against the A/C/D test results.
Values below are quoted at the precision reported by the owner.

| B method | Test queries | Precision@5 | Recall@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 k1=1.2, b=0.75 | 4563 | 0.0636 | 0.3029 | 0.2300 | 0.2453 |
| Tuned BM25 k1=1.2, b=1.0 | 4563 | 0.0656 | 0.3111 | 0.2361 | 0.2516 |

B selected parameters on its development split by nDCG@5, with MRR breaking ties.
For a final four-person table, B needs preparation and tuning on the shared hash
split followed by evaluation on its held-out test questions. Do not copy these
scores into the shared table under a different split label. The BM25 control in
C's experiment is C's control, not a replacement for B's independent contribution.

## Timing and answers

A's reported latency is a CUDA measurement; C/D's exported latencies are CPU batch
measurements, and B uses single-search timings. They do not establish a speed
ranking. D separately measured all five configurations on one CPU with warmups and
repeated, interleaved queries; see its report for that controlled latency study.

C's FLAN-T5-small answer experiment is a 50-question pilot. A/B/D have no final
shared answer scores. [answer_comparison.csv](answer_comparison.csv) therefore
remains header-only. A final comparison needs one generator, prompt, question
sample, context policy, and scoring protocol.

## Remaining integration work

1. Align B to the common corpus/query split, rerun development selection and test,
   and export `team_metrics/b_bm25.csv`.
2. Supply A's MRR/nDCG if those metrics are required across every method; keep
   missing values blank until measured.
3. Run the team's common answer-generation and scoring protocol.
4. Review code integration separately. A has independent Git history; B/C/D
   share the initial baseline. The overview refresh does not merge method code
   or change the owners' branches.
