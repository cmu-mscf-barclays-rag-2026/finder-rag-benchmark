# Person 3: Reference-answer retrieval and HyDE

## New: generation and parameter robustness

The week-of-6-October follow-up is documented in
[ROBUSTNESS_README.md](ROBUSTNESS_README.md). Its code is `run_robustness.py`
and its saved results are in `results/person3_hyde_robustness/`.
Start with `findings.md` and `generation_comparison_k5.csv` in that results folder.
These are exploratory results on the previously inspected Legal RAG Bench set.

This folder implements the Person 3 assignment from Meeting 5:

1. retrieve with the expert reference answer as an **oracle diagnostic**;
2. generate a hypothetical answer using **only the question**;
3. retrieve with the question, oracle answer, and HyDE text under one protocol;
4. optionally fuse original-question BM25 with HyDE-dense retrieval;
5. save metrics and examples showing where HyDE helps or hurts.

## Important distinction

Reference-answer retrieval is not deployable. It uses information that is not
available at query time and is included only to test whether answer-like language
makes the labelled passage easier to retrieve. The real HyDE method sees only the
question.

## Reproduce

Use Python 3.10–3.12 from the repository root:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install --upgrade pip
pip install -r person3_hyde/requirements.txt
python person3_hyde/run_hyde_experiment.py
```

The default `concise_legal` prompt can be compared with the longer variant:

```bash
python person3_hyde/run_hyde_experiment.py \
  --prompt-style long_legal \
  --output-dir results/person3_hyde_long_prompt
```

The Legal RAG Bench JSONL files must be in `data/legal_rag_bench/`. If they are
missing, run the existing Part C experiment once to download them:

```bash
python legal_rag_hybrid_experiment.py
```

The first run downloads MiniLM and FLAN-T5-base. Later runs reuse the model,
corpus-embedding, and generated-HyDE caches. CPU is the default, so no GPU or API
key is required.

## Saved outputs

All outputs are in `results/person3_hyde/`:

| File | Purpose |
|---|---|
| `retrieval_metrics.csv` | Precision, Hit/Recall, MRR, nDCG and latency at K=1/3/5/10 |
| `summary_k5.csv` | Small Top-5 table that can be read directly for slides |
| `paired_comparisons_k5.csv` | Paired wins/losses and exact McNemar p-values |
| `generated_hyde_answers.csv` | Question-only hypothetical answers for auditing/reproduction |
| `per_query_results.csv` | Per-question gold ranks and Top-5 passage IDs |
| `hyde_help_hurt_examples.csv` | Cases where HyDE gains or loses a Top-5 hit |
| `findings.md` | Presentation-ready interpretation |
| `run_summary.json` | Dataset hashes, model/configuration and package versions |
| `generation_sensitivity_k5.csv` | Exploratory prompt/decoding sensitivity check |

Validate the committed files without downloading any model:

```bash
python person3_hyde/validate_results.py
```

Do not describe oracle performance as a production result. For presentation,
compare deployable methods separately and use the oracle only to explain the
query-document wording gap.
