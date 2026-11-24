# Week 11 FinDER retrieval follow-up

The [matched backend findings](results/finder_backends/findings.md) compare the
original question with the same two GPT-4o mini rewrites used in Week 10. All
methods search the same 5,830 pooled FinDER reference passages for the same
200 held-out questions. BM25 is the tuned Week 10 retriever; dense retrieval
uses the locally cached `sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Hybrid combines the two
backend rankings with weighted RRF. The primary 25% dense weight came from the
Week 3 development set and was fixed before these test results. A 50% dense
weight is reported as an exploratory fusion check.

The script replays and verifies the saved BM25 rankings before scoring the
new methods. It makes **no GPT calls**. Query-generation latency and cost are
in the [Week 10 FinDER report](../week10_followup/results/finder_comparison/findings.md);
new MiniLM timings are recorded separately because they use batched inference.

From the repository root:

```sh
.venv-week6/bin/python -m week11_followup.finder_backend_comparison --device cpu
.venv-week6/bin/python -m week11_followup.diagnose_failures
```

The first command writes `summary.csv`, `paired.csv`, `candidate_diagnostics.csv`,
`rankings.jsonl`, `manifest.json`, and `findings.md` to
`week11_followup/results/finder_backends/`. It may overwrite only those files
in that directory. `--model-path`, `--device mps`, and `--output` are available
for a different local checkpoint path, device, or result directory.

The second command writes the [failure analysis](results/finder_backends/error_analysis.md),
`error_analysis.csv`, and `error_analysis_counts.json` beside the main results.
It replays the saved BM25 searches at candidate depths 20 and 100, with no
new GPT or embedding calls. Run it after the backend comparison if those
rankings are regenerated.

Retrieval labels are exact membership in the pooled reference corpus. Scores
are a 200-question pilot, not final-answer accuracy or the full FinDER test
set. The script reports matched differences only within this query set.
