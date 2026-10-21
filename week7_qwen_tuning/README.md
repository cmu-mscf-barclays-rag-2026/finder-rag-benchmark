# Week 7 — Qwen hybrid tuning and context budgets

This extends [Florence's Week 6 Qwen experiment](../week6_qwen_hierarchy/README.md)
using its saved 4,096-token-window Qwen rankings. It sweeps BM25 weights of
0, .25, .50, .75, and 1.00 on the 20 development questions, then freezes one
weight for each hierarchy mode and delivery budget. The 80 test questions are
used only after selection. The full [findings](results/findings.md) and
[context-budget curve](results/context_budget_curve.svg) show the results.

Run from the repository root with the existing Week 6 environment:

```sh
.venv-week6/bin/python -m week7_qwen_tuning.run --output week7_qwen_tuning/results_new
.venv-week6/bin/python -m week7_qwen_tuning.plot_curve --results week7_qwen_tuning/results_new
```

`run.py` refuses to overwrite a nonempty output. No Qwen model inference is required. The same pinned
tokenizer measures every delivery budget. The saved 0.75 BM25/Qwen fusion
and BM25 test rankings are checked before evaluation.

Outputs:

- `dev_sweep.csv`: all five weights, four modes, four budgets, and K=5/10.
- `selected.json`: development-selected weight by mode and budget.
- `test_frozen.csv`: only those frozen configurations on test.
- `test_dense.csv`: Qwen dense-only test comparator for every mode and budget.
- `test_per_query.csv`: delivered IDs and evidence metrics for both sets above.
- `parent_wins.csv`: direct-parent wins versus expansion at the same selected
  direct-parent weight, with retrieved parent IDs, ranks, gold-child positions,
  and cutoff cause.
- `findings.md` and `context_budget_curve.svg`: report and visual curve.

The selected strategy at a budget is determined by the development scores;
the test table reports its held-out performance. This avoids picking a
strategy after viewing test outcomes. The 20-question development set is
small, so weight and strategy ties can be unstable.
