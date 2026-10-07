# Cheryl — HyDE generation and parameter robustness

## Scope

Keep `sentence-transformers/all-MiniLM-L6-v2` fixed. Generate from the question
only using pinned `google/flan-t5-base`. The reference answer is embedded in a
separate oracle diagnostic and never passed to the generator, prompt or fusion.

The official dataset has 100 questions and 4,876 passages. It was already inspected
in earlier experiments. These sweeps are **exploratory on that same set**, not a
new held-out evaluation. Do not present the highest sweep result as an unbiased
estimate, or claim significance after selecting the highest-scoring configuration.

## Prespecified grid

- Short-answer versus evidence-passage prompt families.
- Maximum generation lengths: 32 and 96 output tokens (caps, not fixed lengths).
- Greedy decoding versus sampling at temperature 0.7 and top-p 0.9.
- Three question-only prompt variants per family. The first is the single-HyDE
  control; all three form multi-HyDE. Thus multi-HyDE tests prompt diversity too,
  not just repeated generations of one identical prompt.
- Representations: first hypothesis alone; question concatenated with first
  hypothesis; average of three individually normalized embeddings followed by
  normalization; equal-weight RRF of three independently retrieved lists.
- Hybrid: original-question BM25 plus each dense representation; dense weights
  0.25 / 0.50 / 0.75; candidate depths 20 / 100; RRF constant 60.
- Additional sampled evidence-passage length-96 runs with seeds 20261007 and
  20261008; the main grid uses 20261006. Temperature has no effect in greedy mode.
- Final K = 1 / 3 / 5 / 10 for every method.

MiniLM uses its original 256-token limit. We save generated-text token counts and
truncation indicators. No Qwen model or teammate's code is required.

## Run on another computer

From the repository root, use Python 3.12 and a virtual environment:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r person3_hyde/requirements_robustness.txt
python person3_hyde/run_robustness.py
python person3_hyde/validate_robustness.py
```

For Linux CPU-only PyTorch, you can first install the pinned torch version from
`https://download.pytorch.org/whl/cpu`. No API key or GPU is required. The first
run downloads pinned models and the public JSONL source files. Source data and
model weights are not committed; source hashes are saved in `run_manifest.json`.
The dataset revision is pinned too; a changed source file causes an explicit error.

The saved generated CSVs are also inputs for replay: a full run uses them again
when the complete configuration/question signature matches. Delete them only if
fresh generation is intended. Sampling is seeded; hardware and library changes
may still change generated text. Use the committed text/rankings for exact result
replay and the manifest/requirements for rerunning the full experiment.

## Read or verify results without model downloads

```bash
python person3_hyde/validate_robustness.py
```

This validator uses the Python standard library only and independently checks
precision, recall, Hit rate, MRR, nDCG and paired exact McNemar p-values from CSV
ranks. With dependencies installed, regenerate metric CSVs from saved rankings:

```bash
python person3_hyde/run_robustness.py --replay
```

Or open `results/person3_hyde_robustness/summary_k5.csv` directly in Excel/pandas.

## Files

- `summary_k5.csv`: every configuration at Top 5, including oracle-gap fraction.
- `presentation_k5.csv`: compact controls and observed maxima, clearly labelled.
- `retrieval_metrics.csv`: all K values.
- `robustness_by_representation.csv`: range/mean over all dense settings.
- `exploratory_top_configurations.csv`: descriptive ranking, not a final winner.
- `paired_comparisons_k5.csv`: exact McNemar tests, including Holm adjustment
  across the complete exploratory comparison family.
- `generation_comparison_k5.csv`: concise prompt/length/decoding comparison.
- `fixed_weight_comparison_k5.csv`: fixed-weight hybrid comparison.
- `seed_robustness_k5.csv`: repeated stochastic seed results.
- `generation_audit_summary.csv`: output length and truncation summary.
- `generated_*.csv`: actual generated text, question, hypothesis and cache signature.
- `generation_configs.csv` / `.json`: generator configuration, prompts and timing.
- `generation_audit.csv`: length, overlap and MiniLM truncation checks.
- `per_query_results.csv`: gold ranks and Top-5 corpus indices.
- `passage_ids.csv` / `queries.csv`: index-to-passage and question mappings.
- `saved_rankings.npz` / `method_metadata.json`: model-free metric replay.
- `run_manifest.json`: dataset hashes, pinned models and actual package versions.
- `findings.md`: interpretation and simple meeting speaking notes.

Latency is an amortized **batched CPU throughput estimate**, not an interactive
single-query benchmark: single-HyDE generation/embedding is estimated as one third
of measured three-hypothesis cost, and concatenated-query extra encoding is not
included. Do not use these timing columns to claim precise interactive speedups.
A dedicated latency experiment would be needed for a deployment comparison.

`oracle_gap_closed = (Hit@5 - question_dense Hit@5) /
(oracle Hit@5 - question_dense Hit@5)`. Negative values mean worse than the original
question. This is a descriptive measure of this diagnostic gap; the oracle is not
a mathematical upper bound or a deployable method. Retrieval metrics do not measure
answer correctness: the generated text is a search query, not a final grounded answer.
