# Legal RAG Bench: Data and Evaluation

Part A of the team's retrieval benchmark: a full dataset audit, a shared loader, and a method-independent evaluator. The notebook is the presentation entry point. All code, comments, and reports are in English.

## Findings

The inspected revision contains **4,876 passages and 100 questions**. Every question has **one annotated supporting passage**: 100% single-gold and 0% multi-gold. The questions reference 95 distinct gold passages.

This benchmark is useful for semantic retrieval: its questions were deliberately written with different wording from their supporting passages. It does **not** directly test retrieval of several jointly necessary passages. Its source is a long legal reference work, but the distributed corpus is already divided into short passages. Neither long-document retrieval nor multi-evidence completeness should be claimed from this dataset alone.

Measured passage text length is 209 words at the median, 382 at the 95th percentile, and 452 at the maximum. These are whitespace-delimited words, not model tokens. The audit also reports lengths for 724 ID-derived section groups, which are not reconstructed original Word documents.

## Open the notebook

Open `Legal_RAG_Bench_A.ipynb`. Its saved outputs include schema validation, length statistics, evidence-label counts, the team split, and evaluator checks. The summary cells use the included aggregate files and can run without downloading the dataset. A separate optional cell loads the source records for interactive inspection.

## Reproduce

From this folder, with Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m legal_rag_a download
python -m legal_rag_a profile
python -m unittest discover -s tests -v
```

The download is pinned to Hugging Face commit `db0b31dc6d195ce9916897e1ac5e4e6209736c8a`. Raw files stay in `.cache/` and are excluded from Git. The loader verifies file hashes and fails on invalid IDs or unresolved evidence. No API keys, GPU, embedding model, or answer-generating LLM are required for Part A.

## Shared interface for retrieval implementations

```python
from legal_rag_a import load_benchmark, retrieval_text, evaluate

corpus, questions = load_benchmark()
passage_ids = [p['id'] for p in corpus]
texts = [retrieval_text(p) for p in corpus]
# Index texts and preserve passage_ids in the same order.
# Only question['question'] goes to the retriever; answers and gold_ids are labels.
# predictions = {question_id: [ranked_passage_id, ...], ...}
# per_query, summary = evaluate(questions, predictions, set(passage_ids))
```

The API can evaluate a query subset. The CLI below enforces the selected team split. All methods must use the same corpus, text policy, query split, and k values. Use `text_only` for the initial comparison. `title_text_footnotes` is available as a separately reported experiment.

Save a JSONL file containing exactly one row per evaluated query, using this format:

```json
{"query_id": "1", "passage_ids": ["1.2-c2-s2", "1.1-c1-s1"]}
```

This illustrates the format only; it is not a measured retrieval result. IDs must come from the corpus. Rankings are ordered best first, with unique passage IDs. Explicit empty lists are valid; missing query rows and unknown IDs are rejected.

```bash
python -m legal_rag_a evaluate --predictions predictions.jsonl --method bm25 --split test --out results/local_bm25
```

Use `--split dev` for tuning, `--split test` for our held-out 80 questions, or `--split official_test` for all 100 questions. Submit only the query IDs belonging to the selected split; see `results/query_splits.csv`. Repeat with dense, hybrid, and graph-assisted rankings. Latency must be measured by the retrieval code; the evaluator does not invent it from saved rankings.

## Evaluation protocol

The official data has only a test split. Our optional exploratory split contains 20 development and 80 test queries. It groups questions sharing a gold passage, hashes group IDs with a fixed seed, and allocates 20% of groups to development. The split is reproducible and keeps shared gold passages together. It does not establish independence across legal topics or parent sections. All corpus passages remain searchable in both splits.

Select retrieval parameters on development questions, freeze them, and then evaluate test questions. All-100 results after this tuning are descriptive, not a clean held-out result or an exact reproduction of the paper. With just 100 questions, small score differences need cautious interpretation.

The evaluator reuses the metric conventions of our FinDER code: fixed-k precision, per-query recall, reciprocal rank, binary nDCG, and macro averaging. It adds explicit validation and labeled-evidence completeness. See [METRICS.md](METRICS.md) for formulas and limitations.

## What has been executed

- Download and hash validation of both complete source files.
- Schema, label-link, length, duplicate-text, and split audits over every record.
- Oracle and empty-output controls through the evaluator for all 100 questions.
- Unit tests with hand-calculated single- and multi-evidence cases.

Oracle controls intentionally use gold labels. They validate calculations and are **not retrieval model results**. Part A does not include BM25, dense, hybrid, GraphRAG, or generation performance claims. The synthetic multi-evidence test verifies the metric implementation; it does not add multi-evidence labels to the benchmark.

## Live latency and team display names

The notebook now displays **All Evidence Hit** and **Evidence Coverage** explicitly.
The internal `all_gold_included_at_k` field remains compatible with previous exports.

`legal_rag_a.timing.benchmark_retrievers` adds reusable single-query timing, with
warm-ups, randomized interleaving, repetitions, raw measurements, mean/median/p95,
and optional GPU synchronization. Notebook section 9 joins actual timing summaries
to the retrieval metrics and exports the team comparison table at k=5.

Build retrievers before timing and register adapters in the notebook's `RETRIEVERS`
dictionary. Each adapter takes `(question_text, k)` and returns original passage IDs.
The adapter includes encoding, search, fusion/refinement, and evidence lookup. Record
hardware, model versions, text policy, and cache behavior alongside the resulting files.
No adapters are registered by default, so no model performance is claimed. Fourteen
tests cover ranking metrics and the timing protocol; controlled-clock timing tests
are validation checks rather than measured retrieval performance.

## Files

| File | Purpose |
|---|---|
| `Legal_RAG_Bench_A.ipynb` | Dataset findings and evaluation walkthrough with saved outputs |
| `legal_rag_a/data.py` | Pinned download, source integrity, schema checks, shared text policy |
| `legal_rag_a/profile.py` | Dataset statistics and reproducible group-based split |
| `legal_rag_a/evaluation.py` | Standard retrieval and annotated-evidence metrics |
| `legal_rag_a/timing.py` | Shared single-query retrieval timing protocol |
| `legal_rag_a/__main__.py` | Download, profile, and evaluation commands |
| `config.json` | Common dataset, k values, text policy, and split settings |
| `DATASET_REPORT.md` | Schema, measured lengths, suitability, and limitations |
| `METRICS.md` | Metric formulas and the interpretation of completeness |
| `results/` | Aggregate audit outputs, ID/length tables, controls, and manifests |
| `tests/test_evaluation.py` | Ranking and coverage checks with known answers |
| `DATA_LICENSE.md` | Source attribution and upstream licensing discrepancy |

## Add to the team repository

Upload the entire `legal_rag_bench_a` folder at the team repository root on a branch such as `legal-rag-data-evaluation`. Its README belongs inside this folder, so it does not replace the existing repository README. Keep the folder structure intact. The ZIP excludes raw data, environments, and caches. Create a pull request when ready for teammates to integrate it.

## Sources

- [Dataset and data card](https://huggingface.co/datasets/isaacus/legal-rag-bench)
- [Paper, sections 3 and 4](https://arxiv.org/html/2603.01710v1)
- [Official evaluation code](https://github.com/isaacus-dev/legal-rag-bench)

This module is a team retrieval evaluation utility, not the authors' full end-to-end evaluation pipeline.
