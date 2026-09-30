# Week 5 — Florence Liu: hierarchical retrieval

**Question:** Does grouping child passages into parents improve retrieval?

With unlimited context, hybrid Top-10 gold-evidence hits rose from **33/80 for
children to 49/80 for parents**, while returning about 2.6 times as much text.
Under a common 2,048-token cutoff, the counts were **23/80 and 22/80**: the
Top-10 advantage disappeared with this context policy.

The experiment compares BM25 and hybrid retrieval over children, merged parents,
and child-to-parent expansion, using Kevin Zhang's hierarchy.

- [Meeting brief](results/comparison/MEETING_BRIEF.md): findings and examples.
- [Results table](results/comparison/report.md): Hit@1/5/10, coverage, context size.
- [Protocol and reproduction commands](PROTOCOL.md): settings, split, and 44 → 56 audit.
- [Kevin's Part 2 handoff](kevin_hierarchy/PART2_HANDOFF.md): unchanged imported source.

`hierarchy_experiment.py` runs the experiment; `validate_hierarchy.py` audits saved
results. `requirements-lock.txt` records its environment. `results/comparison/`
contains all Week 5 rankings, metrics and reports. `kevin_hierarchy/` keeps Kevin's
processed corpora, code, analysis and tests together as a byte-identical snapshot
of commit `ed72cf8473dea5df38e56ee1f62ece325bebb788`.

From the repository root, validate the saved run:

```sh
.venv/bin/python -m week5_hierarchy.validate_hierarchy --results week5_hierarchy/results/comparison
```

Run a new experiment without overwriting the saved comparison:

```sh
.venv/bin/python -m week5_hierarchy.hierarchy_experiment --output week5_hierarchy/results/repeat --offline
```

The runner reuses the BM25 implementation from [Week 3](../week3_bm25/README.md)
and model/fusion helpers from [Week 4](../week4_legal_rag/README.md).

The saved numerical results are unchanged by the weekly folder reorganization. Exact execution sources remain in
`results/comparison/source_at_run/`; the manifest points to those snapshots for
historical code-hash verification. New runs hash the reorganized current sources.
