# FinDER RAG Team Benchmark

Shared repository for the four-person FinDER retrieval benchmark. The team compares
dense retrieval, BM25, hybrid retrieval, and threshold/MMR refinement under a common
data and evaluation protocol.

## Legal RAG Bench — Florence's next experiment

Florence's BM25 and hybrid work on `isaacus/legal-rag-bench` is in
[legal_rag/README.md](legal_rag/README.md), alongside a dense comparison baseline.
See the [initial held-out retrieval results](legal_rag/results/initial/report.md).
This experiment has its own corpus, internal 20/80 question split, parameter sweep,
and reproducible metric checks; its scores must not be combined with FinDER scores.
Answer-correctness and groundedness evaluation are prepared but remain unmeasured
until an answer-generating model is selected.

## Main and personal branches

`main` is the shared, working starting point and the destination for reviewed team
contributions. Each teammate develops on their own branch and opens a pull request
into `main` when the work is ready for integration.

| Workstream | Owner | Development branch |
| --- | --- | --- |
| A — Dense retrieval | Yuchen / Person 1 | [`yuchenlu`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/yuchenlu) |
| B — BM25, MRR/nDCG, keyword-versus-dense analysis | Florence / Person 2 | [`feature/person2-bm25`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person2-bm25) |
| C — Hybrid retrieval | Cheryl / Person 3 | [`feature/person3-hybrid`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person3-hybrid) |
| D — Threshold/MMR refinement and latency analysis | Kevin / Person 4 | [`finder-threshold-mmr`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/finder-threshold-mmr) |

All four workstreams now have remote branches. Yuchen's branch has independent
Git history; its saved results are included below, while its application code
remains on that branch pending a separate integration.

The initial commit included Cheryl's Part C implementation and results. These
remain on `main` as the working reference baseline: Person D imports functions
from `finder_hybrid_experiment.py` and checks against its saved rankings. Cheryl's
new branch preserves that starting point for her future changes. Creating a branch
does not remove files from `main`; branches share their existing commit history.

See [GITHUB_SETUP.md](GITHUB_SETUP.md) for checkout, push, and pull-request steps.

## Shared project files

| Path | Purpose |
| --- | --- |
| `data/` and [DATA_LICENSE.md](DATA_LICENSE.md) | FinDER annotations and attribution |
| [config/benchmark_config.json](config/benchmark_config.json) | Dataset checksum, corpus definition, split, and evaluation cutoffs |
| [TEAM_METRICS_SPEC.md](TEAM_METRICS_SPEC.md) | Common metric submission contract |
| [team_metrics/README.md](team_metrics/README.md) | How each owner exports their metrics |
| `scripts/export_method_metrics.py` | Export one method into the common CSV format |
| `scripts/aggregate_team_metrics.py` | Validate submissions and build comparison tables |
| `presentation/` | Tables generated from the currently available submissions |

The existing root-level experiment scripts, notebook, and `results/` contain the
initial reference implementation and Part C experiments. Their detailed methods,
results, caveats, and reproduction commands are preserved in
[Part C: hybrid retrieval](docs/PART_C_HYBRID.md).

Yuchen's app, dense evaluation, and chunking sweep are on `yuchenlu`.
Florence's standalone BM25 project is in `person2_bm25/` on both `main` and
`feature/person2-bm25`, including the shared-split rerun and reproducible timing harness.
Person D's additional code and results are on `finder-threshold-mmr`. A file added
on a personal branch appears on `main` only after its changes are integrated.

## Evaluation rules

- Use the same dataset, corpus construction, query split, and relevance labels
  before comparing methods. The current configuration uses the deduplicated
  reference-passage corpus, not the full 10-K corpus from the FinDER paper.
- Tune on development queries and report held-out test metrics at the agreed K.
- Check each contributor's actual protocol against the common configuration before
  exporting results. Branch organization does not make independently prepared
  datasets or splits compatible.
- Use the same generator, prompt, sample, and context rules for final answer scores.
  The existing FLAN-T5-small answer experiment is a pilot.
- Report the contributor's hardware and latency measurement protocol.

## Current contribution summary

See the [all-four team overview](presentation/team_overview.md) for methods,
reported results, branch links, and integration status, and the
[shared-split comparison](presentation/presentation_summary.md) for all four workstreams.

- **A / Yuchen:** dense MiniLM with development-selected 500-character chunks and
  75-character overlap, max-pooled back to source passages; Recall@5 = 0.2284.
- **B / Florence (Person 2):** standalone BM25, MRR/nDCG, tuning, and error analysis;
  retuned on the shared 1,128-query dev split to k1=1.6, b=1.0. Shared-test
  Recall@5 = 0.3112 on all 4,575 test questions.
- **C / Cheryl:** weighted hybrid RRF, development tuning, and pilot answer
  generation; shared-split Recall@5 = 0.3056.
- **D / Kevin:** threshold and MMR sweeps, paired errors, and repeated latency
  measurements; selected configurations did not improve dense retrieval quality.

`team_metrics/` now includes **A, B, C, and D** on the common 4,575-query split.
Florence's exported metrics were independently recomputed from saved rankings with
the existing team evaluator; all cutoffs match exactly. A's unreported MRR/nDCG
remain blank, and final answer evaluation is still pending.

A new [controlled timing comparison](presentation/latency_comparison.csv) measures
the selected methods on one Mac CPU, using the same 100 held-out questions,
3 warmups, and 5 interleaved repetitions. Index construction is excluded. See
[the timing report](results/common_latency/report.md) for details. These results
support a comparison on this machine and protocol; original mixed-hardware timings
remain provenance only, not a speed ranking.

[Source commits and export provenance](team_metrics/provenance.json) identify
exactly which saved artifacts were used. Florence's shared-split BM25 evaluation
and the controlled timing study were rerun; A/C/D quality scores are the owners'
original saved results. Their branches are unchanged.

## Run the existing reference baseline

From the repository root, create a Python 3.10–3.12 environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -r requirements.txt
python reproduce_c_metrics.py
```

This reproduces Part C, exports its metrics, and rebuilds the current comparison.
The first run downloads `sentence-transformers/all-MiniLM-L6-v2`; CPU is the default.

After compatible, reviewed team submissions have been integrated, rebuild tables:

```bash
python scripts/aggregate_team_metrics.py --k 5
```

The aggregator rejects mismatched dataset, corpus, or split identifiers. Final
answer tables require a common answer protocol; pilot scores remain excluded.

## Reproduce Florence's team submission and controlled timing

Use Python 3.12 with an isolated environment, from the repository root:

```bash
python -m pip install -r requirements-timing.txt
python person2_bm25/run_shared_benchmark.py
```

The BM25 runner verifies the dataset, corpus and qrels, retunes only on development
queries, writes the frozen settings, and then evaluates test. It preserves the
older seed-42 experiments as historical results.

Download the pinned MiniLM revision into the model cache before the offline timing
run (internet is needed only for installation/model download):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', device='cpu', cache_folder='.cache/models')"
python scripts/common_latency.py --index-device mps
python scripts/aggregate_team_metrics.py --k 5
```

`--index-device mps` uses an Apple GPU only for untimed index preparation; all
measured queries use CPU. On other machines omit that option to prepare indexes
on CPU. A CPU-prepared rerun may have small floating-point differences from the
saved MPS-prepared indexes. Keep a full clone with the pinned A/D source commits
available; the harness reads their retrieval functions without changing their
branches.
