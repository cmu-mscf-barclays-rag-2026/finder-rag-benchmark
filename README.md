# FinDER RAG Team Benchmark

Shared repository for the four-person FinDER retrieval benchmark. The team compares
dense retrieval, BM25, hybrid retrieval, and threshold/MMR refinement under a common
data and evaluation protocol.

## Main and personal branches

`main` is the shared, working starting point and the destination for reviewed team
contributions. Each teammate develops on their own branch and opens a pull request
into `main` when the work is ready for integration.

| Workstream | Owner | Development branch |
| --- | --- | --- |
| A — Dense retrieval | Person 1 | Existing workflow unchanged; confirm branch name with its owner |
| B — BM25, MRR/nDCG, keyword-versus-dense analysis | Person 2 | [`feature/person2-bm25`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person2-bm25) |
| C — Hybrid retrieval | Cheryl / Person 3 | [`feature/person3-hybrid`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/feature/person3-hybrid) |
| D — Threshold/MMR refinement and latency analysis | Person 4 | [`finder-threshold-mmr`](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/tree/finder-threshold-mmr) |

Person 1's branch was not present in the remote branch list when this guide was
updated. No branch was created or renamed for Person 1, and Person 4's branch is
unchanged.

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

Person 2's standalone project is in `person2_bm25/` on `feature/person2-bm25`.
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

The current presentation tables contain the submissions available so far; they
are not a completed four-person comparison. See the metric contract for the
checks required before presenting a combined result.

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
