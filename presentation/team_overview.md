# Four-person contribution overview

This snapshot summarizes all four uploaded workstreams. Florence's BM25 code and shared-split rerun are integrated on `main`; the other
owners' additional code remains on their development branches.
Source commits are pinned in [provenance.json](../team_metrics/provenance.json).
Florence's BM25 was retuned and evaluated on the common split, and the selected
A/B/C/D methods were remeasured under a common latency protocol. A/C/D full-test
quality scores remain their original saved results; generation was not rerun.

## Contributions

| Owner | Work delivered | Branch | Detailed source |
| --- | --- | --- | --- |
| A / Yuchen | Dense MiniLM, chunk-size/overlap sweep, source-passage pooling, held-out evaluation, app and tests | [`yuchenlu`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/yuchenlu) | [Task A report](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/blob/bd5e7c35443c6eaa5a1f24ada240b2c5ffdb3eb1/results/task_a_report.md) |
| B / Florence / Person 2 | BM25 inverted index, MRR/nDCG, common-split rerun, tuning, failure analysis, notebook, 25 offline tests | [`feature/person2-bm25`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person2-bm25) | [Verified results](../results/b_bm25_common/report.md) |
| C / Cheryl | BM25/dense controls, weighted hybrid RRF, dev tuning, pilot answer generation, shared benchmark tools | [`feature/person3-hybrid`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person3-hybrid) | [Part C write-up](../docs/PART_C_HYBRID.md) |
| D / Kevin | Threshold/MMR tuning, paired evidence-error analysis, repeated CPU latency, held-out evaluation | [`finder-threshold-mmr`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/finder-threshold-mmr) | [Refinement report](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/blob/4013a25b33a7dc62845359941f062870c822f91a/D_RESULTS.md) |

## Common split: A, B, C, D

All four use 5,830 normalized reference passages and the SHA-1/modulo-5 query
split: 1,128 development and 4,575 test questions. A embeds chunks and max-pools
scores to distinct source passages, retaining passage-level relevance labels.
B verifies every normalized passage and per-query relevance set against the
shared Parquet and uses the exact same hash split. A's dataset-content fingerprint was recomputed from the shared Parquet and matched
its saved run manifest. C/D use the same shared implementation and configuration.

| Owner | Selected method | Precision@5 | Recall@5 | MRR@5 | nDCG@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Dense MiniLM, chunk 500 / overlap 75 | 0.04791 | 0.22840 | N/A | N/A |
| B / Florence | BM25 k1=1.6, b=1.0 | 0.06553 | 0.31118 | 0.23397 | 0.24973 |
| C | Hybrid RRF, dense weight 0.25 | 0.06435 | 0.30561 | 0.22051 | 0.23862 |
| D | Threshold 0.3 | 0.04503 | 0.21169 | 0.15833 | 0.16885 |
| D | MMR lambda 1.0 / fetch 10 | 0.04507 | 0.21177 | 0.15833 | 0.16889 |

Among these submitted configurations, Florence's dev-selected BM25 has the highest
Recall@5. This descriptive ranking now covers all four workstreams on the same
test questions; it is not a statistical significance claim. A's MRR/nDCG remain
unreported. The selected D MMR disables
the diversity penalty; neither selected refinement improves the D dense control.
See the [generated shared comparison](presentation_summary.md) for exact CSV-backed
values and [comparison_k5.csv](comparison_k5.csv) for the machine-readable table.

## Florence's common-split rerun

The original BM25 code and tokenizer were retained. The same nine-point grid was
rerun on the shared 1,128 development questions, selecting k1=1.6, b=1.0 by
nDCG@5, then MRR@5. The settings were saved before evaluating all 4,575 test
questions. A fixed default (k1=1.2, b=0.75) remains a within-B control.

[All exported B metrics](../team_metrics/b_bm25.csv) were independently recomputed
from the saved top-10 rankings with the existing team metric implementation.
The maximum absolute difference across all five retrieval metrics at K=1/3/5/10
was zero. See [verification](../results/b_bm25_common/validation.json),
[the development sweep](../results/b_bm25_common/dev_sweep.csv), and
[the report](../results/b_bm25_common/report.md).

The previous 4,563-test-query seed-42 results remain in Florence's historical
report for provenance, but do not enter the shared comparison.

## Timing and answers

The original owner latency fields still use mixed hardware and protocols. A new
[controlled latency table](latency_comparison.csv) compares all selected methods
on the same Mac CPU using 100 identical held-out questions, 3 warmups, and 5
interleaved repetitions (500 measurements per method). Timing starts with the raw
question and ends with evidence text; query encoding, pooling, fusion/refinement,
and lookup are included. Indexing and generation are excluded.

Untimed indexes were prepared on MPS; every measured query runs on CPU. This
comparison applies to these implementations on this machine, not universal speed
claims. See [the report](../results/common_latency/report.md) and
[raw timings](../results/common_latency/raw.csv) for the full protocol and evidence.

C's FLAN-T5-small answer experiment is a 50-question pilot. A/B/D have no final
shared answer scores. [answer_comparison.csv](answer_comparison.csv) therefore
remains header-only. A final comparison needs one generator, prompt, question
sample, context policy, and scoring protocol.

## Remaining integration work

1. Supply A's MRR/nDCG if required for every method; keep blanks until measured.
2. Run the common answer-generation and scoring protocol.
3. Review any further A/C/D code integration with their owners. A has independent
   Git history; B/C/D share the initial baseline. Only Florence's branch and main
   were changed for the shared-split and timing work.
