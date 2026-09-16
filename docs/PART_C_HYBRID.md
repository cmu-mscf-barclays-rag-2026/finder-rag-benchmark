> Archived original Part C overview. Run all commands below from the repository root.
> Cheryl develops new Part C work on `feature/person3-hybrid`.
> See the [team overview](../README.md) for the shared repository workflow.

# FinDER RAG Team Benchmark — Part C Hybrid Retrieval

This folder is ready to place inside the team's shared GitHub repository. It
contains the FinDER annotation file, reproducible part-C code, saved results,
and a common metrics format for combining all four workstreams.

## Quick start

Python 3.10, 3.11, or 3.12 is recommended. From this folder, create a virtual
environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux with `source .venv/bin/activate`. On Windows
PowerShell, use `.venv\Scripts\Activate.ps1`. Then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python reproduce_c_metrics.py
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`. CPU is used
by default for more consistent results across computers. Retrieval scores
should reproduce closely, while latency will naturally differ by hardware.

The saved final table can be read without running the models:

```python
import pandas as pd

comparison = pd.read_csv("presentation/comparison_k5.csv")
print(comparison)
```

`presentation/answer_comparison.csv` remains empty until every method has
final answer scores produced with one shared generator, prompt, sample, and
context rule. The included FLAN-T5-small scores are marked as a pilot and are
not treated as a final comparison.

After teammates add their standardized CSV files, rebuild the team table with:

```bash
python scripts/aggregate_team_metrics.py --k 5
```

See `TEAM_METRICS_SPEC.md` for the metric contract and `GITHUB_SETUP.md` for
the shared-repository workflow.

## Purpose

This package contains the work for part C of the team assignment. It compares BM25 sparse retrieval, MiniLM dense retrieval, and a weighted hybrid method on the same FinDER data and the same evaluation labels.

The main result is that the hybrid method improves retrieval at Top 3, Top 5, and Top 10 on the held-out test set. At Top 5, hybrid Recall is 30.56%, compared with 28.87% for BM25 and 21.18% for dense retrieval.

## What counts as the benchmark

FinDER supplies two different types of ground truth:

- `references` contains the expert-selected evidence passages. These are the relevance labels for retrieval metrics.
- `answer` contains the expert-written reference answer. This is the gold answer for answer evaluation.

This experiment deduplicates all `references` into a corpus of 5,830 evidence passages. Each of the 5,703 questions is evaluated against its own reference passage or passages. The split is deterministic: 1,128 development questions are used to choose the hybrid weight and 4,575 questions are held out for final testing.

This is a reference-passage benchmark, not the full-10-K benchmark from the paper. It is suitable for comparing team methods if everyone uses the same corpus construction and split. Its numbers must not be compared directly with the paper's RAGAS Context Recall scores.

## Methods

### BM25

- `k1 = 1.2`
- `b = 0.75`
- Financial punctuation, numbers, percentages, and ticker-like tokens are retained by the tokenizer.

### Dense retrieval

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are L2-normalized.
- Ranking uses cosine similarity.

### Hybrid retrieval

- BM25 and dense rankings each retrieve 100 candidates.
- Rankings are combined with weighted Reciprocal Rank Fusion.
- `RRF k = 60`.
- Dense weights of 0.25, 0.50, and 0.75 are compared on the development set.
- The selected dense weight is 0.25, so the final method gives 75% weight to BM25 ranking and 25% to dense ranking.

## Main held-out test results

| Method | Precision@5 | Recall@5 | Hit Rate@5 | MRR@5 | nDCG@5 | Retrieval latency ms per query |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.0605 | 0.2887 | 0.2964 | 0.2166 | 0.2319 | 0.19 |
| Dense MiniLM | 0.0451 | 0.2118 | 0.2195 | 0.1582 | 0.1688 | 3.63 |
| Hybrid RRF | **0.0644** | **0.3056** | **0.3145** | **0.2205** | **0.2386** | 3.87 |

Relative to BM25 at Top 5, hybrid retrieval improves:

- Precision by 0.39 percentage points, or 6.4% relative.
- Recall by 1.69 percentage points, or 5.9% relative.
- Hit Rate by 1.81 percentage points, or 6.1% relative.
- MRR by 0.39 percentage points, or 1.8% relative.
- nDCG by 0.67 percentage points, or 2.9% relative.

Precision@5 looks numerically low because most FinDER questions have only one relevant reference. Even a perfect retrieval of that one passage produces Precision@5 of 0.20. Hit Rate and Recall are therefore easier to interpret for this dataset.

## Top K results

The complete `retrieval_metrics.csv` file reports metrics at K equal to 1, 3, 5, and 10. Hybrid is not best at Top 1, where BM25 remains slightly stronger. Hybrid becomes better from Top 3 onward because dense retrieval adds some evidence that BM25 misses.

## Answer correctness experiment

The package also contains an exploratory answer-generation test on the same fixed sample of 50 held-out questions. It uses `google/flan-t5-small`, the same prompt, and the Top 3 contexts for every retrieval method.

The exact-match rate is zero for every method, and token F1 and semantic similarity are low. These numbers mainly show that this very small local generator is not strong enough for long financial evidence and numerical reasoning. They should not be presented as a reliable comparison of retrieval quality. For the final project, answer correctness should be rerun with the team's actual answer model and one shared prompt. Recommended answer metrics are semantic similarity, token F1, and an LLM judge such as RAGAS Correctness and Faithfulness.

## Paper baseline

The FinDER paper reports the following total RAGAS Context Recall scores on a representative 10% subset using the full 10-K paragraph corpus:

| Paper method | Context Recall score |
|---|---:|
| BM25 | 11.68 |
| GTE | 17.83 |
| multilingual E5 | 17.36 |
| E5-Mistral | 25.95 |

Those are LLM-scored values on a 0 to 100 scale. They are not the same quantity as the exact Recall@K in this package.

## Files

- `FinDER_Hybrid_C.ipynb`: notebook explaining the experiment and loading the results.
- `finder_hybrid_experiment.py`: complete reproducible experiment.
- `data/train-00000-of-00001.parquet`: official FinDER annotations.
- `results/retrieval_metrics.csv`: all retrieval metrics by method, split, and K.
- `results/hybrid_weight_tuning_dev.csv`: development-set weight selection.
- `results/per_query_rankings.csv`: query-level rankings and first relevant rank.
- `results/hybrid_top100_failures.csv`: first 25 hybrid failures within candidate depth 100.
- `results/answer_correctness.csv`: answer-generation summary for the 50-question sample.
- `results/*_answers.csv`: generated answers and query-level answer scores.
- `results/run_summary.json`: model and run configuration.
- `requirements.txt`: Python dependencies.
- `run_retrieval.py`: one-command retrieval run with a dataset checksum check.
- `reproduce_c_metrics.py`: runs part C, exports its standard CSV, and rebuilds the comparison.
- `config/benchmark_config.json`: frozen dataset, corpus, split, and metric settings.
- `team_metrics/*.csv`: one standardized submission per method owner.
- `scripts/export_method_metrics.py`: converts raw output to the shared format.
- `scripts/aggregate_team_metrics.py`: validates and combines team submissions.
- `TEAM_METRICS_SPEC.md`: required columns and comparison rules.

## Run again

From the package directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python finder_hybrid_experiment.py \
  --parquet data/train-00000-of-00001.parquet \
  --output-dir results \
  --answer-sample-size 50
```

To run only retrieval and skip local answer generation:

```bash
python finder_hybrid_experiment.py \
  --parquet data/train-00000-of-00001.parquet \
  --output-dir results \
  --skip-generation
```

## Recommended team rule

Every team member should use the same Parquet file, reference-passage corpus construction, deterministic split, and metric functions. Only the retrieval method should change. Otherwise, differences in chunking, corpus size, or evidence matching may be mistaken for model improvement.
