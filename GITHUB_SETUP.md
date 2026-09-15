# Put this work in the shared GitHub repository

## When the team repository already exists

Clone the existing repository and work on your own branch:

```bash
git clone <TEAM_REPOSITORY_URL>
cd <TEAM_REPOSITORY_NAME>
git switch -c feature/finder-hybrid-c
```

Copy this complete folder into the cloned repository and name it
`finder_benchmark`. Then run:

```bash
cd finder_benchmark
python scripts/aggregate_team_metrics.py --k 5
cd ..
git status
git add finder_benchmark
git commit -m "Add FinDER hybrid retrieval benchmark"
git push -u origin feature/finder-hybrid-c
```

Open the GitHub repository in a browser. Use the **Compare & pull request**
button, ask the team to review the changes, and merge after the checks pass.

## When the shared repository is completely empty

Only in this case, open a terminal inside this folder and run:

```bash
git init
git add .
git commit -m "Add FinDER RAG benchmark"
git branch -M main
git remote add origin <TEAM_REPOSITORY_URL>
git push -u origin main
```

## How each teammate submits results

Each person keeps their detailed output in `results/` and exports their owned
method into `team_metrics/`. Suggested filenames are:

- `a_dense.csv`
- `b_bm25.csv`
- `c_hybrid_rrf.csv`
- `d_threshold_mmr.csv`

After all four files are present, run:

```bash
python scripts/aggregate_team_metrics.py --k 5
```

The presentation files appear in `presentation/`. If the script reports mixed
`dataset_id`, `corpus_id`, or `split_id`, the scores are not comparable. Fix
the experimental setup before presenting them.

Retrieval results go to `comparison_k5.csv`. Final answer results go to
`answer_comparison.csv` only after all four methods use one shared answer
protocol. Pilot answer scores are intentionally excluded from that table.
