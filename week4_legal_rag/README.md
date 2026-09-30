# Week 4 — Legal RAG retrieval

**Question:** Does combining keyword and dense retrieval improve legal evidence retrieval?

Hybrid retrieved the labeled supporting passage for **32/80 questions at Top-5**,
compared with **30/80 for BM25** and **23/80 for dense retrieval**.
The two-question gain over BM25 is small; the uncertainty analysis does not
establish a clear hybrid advantage.

| Method | Top-5 passage hits | Hit rate |
|---|---:|---:|
| BM25 | 30/80 | 37.50% |
| Dense | 23/80 | 28.75% |
| Hybrid | 32/80 | 40.00% |

All methods searched the complete 4,876-passage Legal RAG Bench corpus. Settings
were selected on 20 development questions, then assessed on 80 test questions.
The selected methods use different chunk sizes, so passage hits must be read
alongside evidence coverage. Answer correctness and groundedness were not measured.

## Results to review

- [Main results and limitations](results/initial/report.md).
- [Top-100 diagnostic](results/top100/report.md) — separates candidate-retrieval failures from ranking failures.
- [Per-question comparison](results/initial/question_comparison.csv) — examples where hybrid helps or hurts.

## Code and reproduction

[Reproduction guide and metric definitions](REPRODUCING.md).

`run.py` runs the retrieval experiment; `candidate_recall.py` runs the Top-100
diagnostic; `validate_results.py` audits saved results. Dependencies and tests are
kept in this folder. The follow-up [Week 5 experiment](../week5_hierarchy/README.md)
tests passage hierarchy using a different split; its scores are not directly
comparable to this week's results.
