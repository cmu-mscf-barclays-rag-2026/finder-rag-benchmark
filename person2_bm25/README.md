# Florence / Person 2 — FinDER BM25 + MRR / nDCG

A standalone project for **Person 2 / B**: BM25 sparse retrieval, ranking metrics,
and analysis of keyword retrieval versus dense retrieval. Everything lives in this
folder within the shared `finder-rag-benchmark` repository. Person 2's work is
developed on the `feature/person2-bm25` branch.

## Team comparison: use the common split

Florence's current team submission uses `sha1-mod5-dev-v1`: **1,128 development /
4,575 test questions**, on the same 5,830 reference passages as A/C/D.
From the repository root, with PyArrow and pandas installed, run:

```bash
python person2_bm25/run_shared_benchmark.py
```

This verifies the shared dataset, corpus and qrels, runs the nine-point grid only
on development queries, freezes the selected configuration, evaluates test, and
exports `team_metrics/b_bm25.csv`. The measured common-split run selected
`k1=1.6, b=1.0`; see [the full report](../results/b_bm25_common/report.md).
The original tokenizer/scorer and default control are unchanged.

The lower-level CLI also supports `prepare --split-protocol sha1-mod5-dev-v1`.
Its default remains the original seeded split so existing experiments do not
silently change. The legacy instructions below reproduce that earlier split;
they are not the current team submission.

## What you can submit

- BM25 implemented from scratch with an inverted index, configurable `k1` and `b`.
- Tested MRR@k and binary/graded nDCG@k, reusable by all four teammates.
- One `evaluate_rag()` interface: Precision@k, Recall@k, MRR@k, nDCG@k,
  evidence coverage, answer exact match, answer token F1, answer correctness hook,
  search latency, and retrieved document count.
- Fixed corpus/query/qrels exports and checks that prevent incompatible comparisons.
- Top-k experiments, development-only parameter tuning, per-category results,
  per-query rankings, failure examples, and cross-method comparison exports.
- [Methodology and team handoff](METHODOLOGY.md), [keyword vs dense analysis](ANALYSIS.md),
  and [walkthrough notebook](person2_walkthrough.ipynb).

**Answer correctness needs Person 3's scorer and generated answers.** BM25 itself
retrieves evidence. Missing answer scores are `null` / `N/A`, never fabricated.
Exact match is a strict lexical accuracy measure, not semantic correctness.

## 1. Run immediately, offline

The core implementation and tests need only Python's standard library, Python 3.9+.
The following six-query fixture is synthetic and is only a functionality demo:

```bash
cd person2_bm25
python3 -m finder_bm25 prepare --input examples/demo.jsonl --output data/demo
python3 -m finder_bm25 run --data data/demo --partition all --output results/demo
python3 -m unittest discover -s tests -v
```

On Windows, use `py` or `python` instead of `python3`.

## 2. Reproduce the earlier seed-42 experiment

Python 3.11 is recommended for a fresh environment. Only PyArrow is needed to
read the real dataset's Parquet file. No embedding model, LLM, or API key is needed.

**macOS / Linux**

```bash
cd person2_bm25
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m finder_bm25 prepare
python -m finder_bm25 run --top-k 3 5 10
```

**Windows PowerShell**

```powershell
cd person2_bm25
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m finder_bm25 prepare
python -m finder_bm25 run --top-k 3 5 10
```

`prepare` downloads the 13 MB public Parquet snapshot once and verifies its SHA-256.
It processes **all 5,703 questions** and pools/deduplicates their reference passages.
The published split is called `train`; this project creates its own deterministic
20% development / 80% test query partition, seed 42. These are local partitions,
not official FinDER holdouts. The corpus includes references from both partitions.

**Corpus scope:** the default is the pool of annotated reference passages, matching
the existing project's reference-based approach. FinDER also provides `10-k.zip`.
This project does **not** reproduce retrieval over that full filing corpus. All
teammates must agree to use the same prepared corpus for these results to compare.

For a quick real-data run, limit **queries**, not the corpus:

```bash
python -m finder_bm25 run --limit 100 --top-k 3 5 10 --output results/quick
python -m finder_bm25 search "CBOE Data and Access Solutions revenue 2023" --top-k 5
```

## 3. Tune BM25 on development queries

Default: `k1=1.2`, `b=0.75`; no stemming, stopword removal, synonym expansion or
query-specific company filtering. Unicode case normalization, possessive removal
and number tokenization are shared between queries and documents.

```bash
python -m finder_bm25 sweep --top-k 5
```

This tests `k1 = 0.8 / 1.2 / 1.6` and `b = 0.25 / 0.75 / 1.0` **on dev only**.
`results/dev_sweep/best_config.json` selects the highest dev nDCG@5, breaking ties
by MRR@5, then the first sorted configuration. Use the selected numbers once on test:

```bash
# Replace these with best_config.json values; this is a command example.
python -m finder_bm25 run --k1 1.2 --b 0.75 --method bm25_tuned --output results/bm25_tuned
```

Do not select parameters from test results. This is an exploratory local benchmark,
not a claim of generalization to unseen companies: related company queries may
appear in both local partitions.

## 4. Share the same evaluation with everyone

Share the entire `data/shared/` directory. It contains:

| File | Purpose |
| --- | --- |
| `corpus.jsonl` | Searchable text with stable `doc_id` and reference offsets |
| `queries.jsonl` | Query IDs/text, answers, categories, local dev/test assignments |
| `qrels.json` | Ground-truth relevance: query ID → document ID → grade |
| `protocol.json` | Normalization, chunking and partition settings |
| `manifest.json` | Source revision, counts and shared-data fingerprint |

Everyone indexes **only `corpus.jsonl` text**, preserving its `doc_id`. Query text,
gold answers and query labels belong to evaluation and must not become indexed
metadata used by the retriever. Do not restrict retrieval to a query's known gold
company or reference IDs.

For Person 1/3/4, wrap the existing retriever so that
`search(query, top_k)` returns `[{'doc_id': ..., 'score': ...}, ...]`, best first:

```python
from pathlib import Path
from finder_bm25.data import load_bundle
from finder_bm25.evaluation import collect_run, evaluate_rag
from finder_bm25.reporting import save_evaluation

bundle = load_bundle(Path("data/shared"))
# teammate_retriever must index bundle['corpus'] and return those exact doc_ids.
run = collect_run(bundle, teammate_retriever, method="dense", top_k=5, partition="test")
evaluation = evaluate_rag(bundle, run)
save_evaluation(bundle, run, evaluation, Path("results/dense/k5"))
```

`collect_run` times retrieval, preserves query IDs and records the corpus fingerprint.
For teammates working elsewhere, [METHODOLOGY.md](METHODOLOGY.md) describes the JSON
run contract. Missing measurements should be `null`; empty retrieval is `doc_ids=[]`.

Then run a fair comparison using real teammate output:

```bash
python -m finder_bm25 compare --runs results/bm25/k5/run.json results/dense/k5/run.json --output results/comparison
```

The command **recomputes** the same metrics for each run and rejects different
corpora, query sets, partitions or cutoffs. It exports a common table and per-query
ranking improvements/regressions for the keyword-vs-dense analysis.

### Add Person 3's answer outputs

The predictions JSONL format is one record per query:

```json
{"query_id": "an_actual_finder_query_id", "answer": "The model's generated answer."}
```

```bash
python -m finder_bm25 evaluate --run results/bm25/k5/run.json --answers predictions/bm25.jsonl --output results/bm25_with_answers/k5
```

This computes exact match and token F1. To apply the team's semantic/factual
correctness evaluator, pass the same function and version for every method:

```python
evaluation = evaluate_rag(
    bundle, run,
    answer_scorer=person3_correctness,  # (prediction, gold_answer) -> float in [0, 1]
    answer_scorer_name="team_correctness_v1",
)
```

Generate an answer from **that method's retrieved contexts at that k** before
scoring. Keep the generation model, prompt, temperature and scoring configuration
fixed. Person 3 owns this generation/correctness stage. The CLI comparison computes
lexical answer metrics; custom correctness comparisons use `evaluate_rag()` with
the shared scorer and `reporting.table()` on its summaries.

## 5. Optional shared chunking

Default whole references give the cleanest exact evidence-ID labels for Person 2.
If the team agrees to use 500-character chunks and 50-character overlap:

```bash
python -m finder_bm25 prepare --chunk-size 500 --overlap 50 --output data/shared_500_50
python -m finder_bm25 run --data data/shared_500_50 --output results/bm25_500_50
```

All methods must use this newly prepared bundle. The splitter works on characters
with word boundaries; it is not LangChain's recursive splitter. Identical size
numbers alone do not make two independently built corpora identical.

In chunk mode, relevance is inherited from the gold reference. A relevant chunk
can contain only part of the answer evidence. Read the metric caveat in
[METHODOLOGY.md](METHODOLOGY.md) before comparing chunk sizes.

## Output files

Each `results/bm25/k5/`-style directory contains:

- `run.json`: complete rankings, scores, query IDs, parameters and measured latency.
- `summary.json` / `summary.csv`: common metrics and scoring coverage.
- `per_query.csv`: rank of first relevant evidence and all query-level metrics.
- `by_category.csv`: results grouped by FinDER category.
- `report.md`: readable table, caveats and concrete failure examples.

`results/bm25/summary.md` compares all requested cutoffs. Results and downloaded
data are ignored by Git; share the prepared bundle deliberately with the team.
The small [RESULTS.md](RESULTS.md) records the verified real-data experiment.

## Project files

| File | Responsibility |
| --- | --- |
| `finder_bm25/bm25.py` | BM25 retrieval and tokenizer |
| `finder_bm25/metrics.py` | Person 2's MRR/nDCG and shared lexical metrics |
| `finder_bm25/data.py` | Corpus, stable IDs, qrels, local partitions, fingerprints |
| `finder_bm25/evaluation.py` | Common `collect_run()` / `evaluate_rag()` |
| `finder_bm25/reporting.py` | Reports, tables, category breakdowns, run comparison |
| `finder_bm25/download.py` | Pinned and checksum-verified dataset download |
| `finder_bm25/__main__.py` | CLI commands |
| `tests/test_person2.py` | Offline metric and integration tests |

## Sources

- [FinDER dataset, schema, attribution and CC BY-NC 4.0 license](https://huggingface.co/datasets/Linq-AI-Research/FinDER).
- [FinDER paper](https://arxiv.org/abs/2504.15800): Choi et al., 2025.
- [Lucene BM25 documentation](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html): positive IDF and parameter interpretation.
- [Stanford IR textbook: ranked retrieval evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).

This Python scorer uses the conventional `(k1 + 1)` numerator and exact token
lengths; it is not a byte-for-byte reproduction of Lucene's scoring implementation.
