# Task A delivery: dense retrieval

## Final held-out result

The chunking configuration was selected on 1,128 development questions and
evaluated once on 4,575 held-out test questions. The selected setting is:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- chunk size: **500 characters**
- chunk overlap: **75 characters**
- primary retrieval depth: **top-k = 5**
- similarity: cosine similarity on L2-normalized embeddings
- source score: maximum score among that source passage's chunks

| Split | Queries | Precision@5 | Recall@5 | Hit Rate@5 | F1@5 |
|---|---:|---:|---:|---:|---:|
| Development (parameter selection) | 1,128 | 0.04965 | 0.23478 | 0.24291 | 0.08196 |
| **Held-out test (final)** | **4,575** | **0.04791** | **0.22840** | **0.23410** | **0.07921** |

The close development and test scores are evidence that the choice did not
substantially overfit the development questions.

## What the parameters mean

- **Chunk size** is the maximum number of characters in each embedded piece of
  a reference passage. Smaller chunks are more focused but produce a larger
  index.
- **Chunk overlap** is the number of characters repeated between adjacent
  chunks. It protects facts near a boundary but increases duplicate text and
  indexing cost.
- **Top-k** is the number of distinct source passages returned for a question.
  Larger k usually increases Recall and lowers Precision.
- **Embedding model** converts questions and chunks to 384-dimensional vectors.
- **Max chunk score** means a source passage receives the cosine score of its
  best-matching chunk. Metrics are evaluated on distinct source passages, not
  on chunks, so changing chunk size does not change the relevance labels.

## What the metrics mean

- **Precision@k** = relevant source passages in the top k divided by k.
- **Recall@k** = retrieved gold source passages divided by all gold source
  passages for that question.
- **Hit Rate@k** = fraction of questions with at least one gold passage in the
  top k.
- **F1@k** = harmonic mean of macro Precision@k and macro Recall@k. F1@5 was
  declared as the parameter-selection objective before examining test results.
- **Latency** is query embedding plus retrieval time per query; index creation
  is excluded. It is hardware-dependent.

Most FinDER questions have one gold passage, so even perfect retrieval often
has Precision@5 = 0.20. Recall and Hit Rate are consequently more intuitive
than the raw Precision number.

## Why the split is valid

FinDER does not provide a separate retrieval test split, so query IDs are split
deterministically: a question is development data when the first eight hex
digits of `SHA1(_id)`, interpreted as an integer, are divisible by five; all
others are test data. This produces 1,128 development and 4,575 test questions.

There is no learned classifier to fit, so a third training split is unnecessary.
All 5,830 deduplicated reference passages remain in the searchable corpus,
including the gold passages for test queries. That is required in an information
retrieval benchmark and is not leakage: question text, answers, and reasoning
are never placed in the corpus. Only development metrics choose parameters;
only the selected configuration is evaluated on test.

## Reproduce

Install `requirements-dev.txt`, then run:

```powershell
python evaluate_dense.py --device cpu
python -m pytest -q
```

CPU is the most portable option. The committed run used CUDA on an RTX 4060
Laptop GPU; model and device are recorded in `results/task_a_run_summary.json`.
Use `--sample-size 100` only for a smoke test—subsampled scores are not final
benchmark results.

Outputs:

- `results/task_a_dense_sweep.csv`: development-only parameter sweep
- `results/task_a_test_metrics.csv`: final held-out metrics at k = 1, 3, 5, 10
- `results/task_a_run_summary.json`: dataset fingerprint and full protocol
- `results/task_a_report.md`: detailed tables and interpretation
