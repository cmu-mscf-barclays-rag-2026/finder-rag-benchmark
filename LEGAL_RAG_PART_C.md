# Legal RAG Bench Part C Hybrid Retrieval

This experiment compares BM25, dense retrieval, and weighted reciprocal rank
fusion on `isaacus/legal-rag-bench`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements_legal_rag.txt
python legal_rag_hybrid_experiment.py
```

The script downloads `corpus.jsonl` and `qa.jsonl` from Hugging Face when they
are not already present. Outputs are written to `results/legal_rag_bench/`.

## Evaluation design

The official dataset has one test split containing 100 questions. Each question
has one labelled most-relevant passage. To prevent fusion-weight selection from
using the same question it is evaluated on, the Hybrid result uses deterministic
five-fold out-of-fold tuning. For each fold, the dense RRF weight is selected on
the other four folds and then applied only to the held-out fold.

The report also includes a fixed 25% dense-weight Hybrid result. This weight was
selected previously on FinDER and is applied to Legal RAG Bench without tuning,
so it serves as a clean cross-dataset transfer test.

Because only one passage is labelled per question, Recall@K and Hit Rate@K are
identical. Precision@K is Hit Rate@K divided by K. This version of the dataset
does not directly support evaluation of multi-evidence completeness.

## Outputs

- `retrieval_metrics.csv`: metrics at K = 1, 3, 5, and 10.
- `hybrid_weight_tuning_cv.csv`: weight-selection results for every fold.
- `hybrid_weight_sensitivity_full_test.csv`: descriptive results for all tested
  fixed weights. This is exploratory and is not used for unbiased model selection.
- `per_query_rankings.csv`: reproducible top-five passage IDs and gold ranks.
- `error_analysis_examples.csv`: BM25-only, dense-only, Hybrid-rescue, and
  Hybrid-failure examples.
- `paired_comparisons_k5.csv`: paired bootstrap intervals and exact McNemar
  checks for the fixed Hybrid against each baseline.
- `findings.md`: concise interpretation for the team meeting and presentation.
- `run_summary.json`: dataset, model, parameters, and evaluation notes.

Dataset license: CC BY-NC-SA 4.0. Cite the Legal RAG Bench paper and dataset
when presenting or publishing results.

## Weight sensitivity at Top 5

| BM25 / dense weight | Recall / Hit Rate | MRR | nDCG |
|---|---:|---:|---:|
| 75% / 25% | 0.36 | 0.2490 | 0.2764 |
| 50% / 50% | 0.37 | 0.2412 | 0.2727 |
| 25% / 75% | 0.30 | 0.1898 | 0.2174 |

The 50/50 mix has the highest Top-5 recall, while the 75/25 mix has the best
ranking quality. The dense-heavy setting performs worse, indicating that the
general MiniLM retriever works better as a complement to BM25 than as the main
retriever on this benchmark. This table is exploratory because all 100 official
test questions are used; it should not be presented as an independently tuned
test result.
