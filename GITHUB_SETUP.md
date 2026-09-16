# Working in the shared team repository

Use this repository for all team contributions:

https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark

`main` holds the shared working baseline and reviewed contributions. Develop on
your assigned branch, push to that branch, and use a pull request to integrate work.
Do not create a separate personal GitHub repository for team submissions.

## Clone once, or open your existing clone

```bash
git clone https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark.git
cd finder-rag-benchmark
```

If you already have this clone, open its folder in your editor. Check the repository
and branch before editing or publishing:

```bash
git remote -v
git branch --show-current
```

`origin` should point to the shared repository above.

## Switch to your existing personal branch

First save and commit your current work before switching branches. Fetch branches:

```bash
git fetch origin
```

Run only the command for your own workstream:

| Contributor | Command |
| --- | --- |
| Yuchen / Person 1 / dense | `git switch yuchenlu` |
| Person 2 / BM25 | `git switch feature/person2-bm25` |
| Cheryl / Person 3 / hybrid | `git switch feature/person3-hybrid` |
| Person 4 / threshold-MMR | `git switch finder-threshold-mmr` |

Git can create a local tracking branch when the matching branch exists on `origin`.
Yuchen's branch is `yuchenlu`. It has independent Git history from `main`, so
coordinate its eventual code integration separately; do not force-push or blindly
merge unrelated histories. The shared overview already includes its saved results.

Cheryl's branch was created from her initial contribution on `main`, so the files
initially look the same. That is expected. Future commits on her branch stay there
until integrated. The inherited reference implementation remains on `main` because
other workstreams reuse it; its inclusion is not a requirement to work on `main`.

## Commit and publish your work

For Person 2, for example:

```bash
git status
git add person2_bm25
git commit -m "Describe the BM25 changes"
git push -u origin feature/person2-bm25
```

Other contributors stage their own changed files and push their own branch. Inspect
`git status` and the staged diff first. Keep environments, credentials, and model
caches out of commits. Do not edit another owner's metric CSV.

On GitHub, open a pull request with **base: main** and **compare: your branch**.
Have the team review the code and evaluation compatibility before merging. The
initial shared baseline stays available throughout; do not force-push shared
history to move old contributions between branches.

## Bring reviewed main changes into your branch

For branches sharing main's history (B, C, and D), start with a clean working tree
on your personal branch. Yuchen's independent-history branch needs a separate
integration plan rather than this merge command:

```bash
git fetch origin
git merge origin/main
```

Resolve any conflicts, run the relevant checks, then push your branch. Merging
`main` brings in shared updates without changing another contributor's branch.

## Submit comparable results

Each owner exports their method into `team_metrics/`. Suggested filenames are:

- `a_dense.csv`
- `b_bm25.csv`
- `c_hybrid_rrf.csv`
- `d_threshold_mmr.csv`

Follow [TEAM_METRICS_SPEC.md](TEAM_METRICS_SPEC.md) and
[team_metrics/README.md](team_metrics/README.md). Check the actual dataset, corpus,
and split before assigning identifiers; incompatible results need a common run,
not relabeling.

After compatible submissions are reviewed and integrated, rebuild tables:

```bash
python scripts/aggregate_team_metrics.py --k 5
```

Retrieval results go to `presentation/comparison_k5.csv`. Final answer comparisons
require the same generator, prompt, sample, context rules, and protocol identifier.
