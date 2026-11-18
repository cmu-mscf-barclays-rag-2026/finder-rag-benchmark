"""Write a concise README from the completed, audited experiment."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build():
    results = ROOT / "results"
    load = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
    manifest, audit = load("manifest.json"), load("independent_audit.json")
    with (results / "summary.csv").open(encoding="utf-8") as source:
        rows = [r for r in csv.DictReader(source) if r["scope"] == "split_language"
                and r["group"] == "test/english" and r["k"] == "10"]
    names = {"bm25": "BM25 OCR", "ocr": "Dense OCR", "image": "Dense image", "combined": "OCR/image fusion"}
    families = {"original": "Original", "multiquery": "MultiQuery", "dspy_initial": "DSPy initial", "dspy_optimized": "DSPy optimized"}
    table = ["| Backend | Query method | Hit@10 | Recall@10 | nDCG@10 |",
             "|---|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r["method"]):
        backend, family = r["method"].split("/")
        table.append(f"| {names[backend]} | {families[family]} | {float(r['hit']):.4f} | {float(r['recall']):.4f} | {float(r['ndcg']):.4f} |")
    changes = []
    for backend, name in names.items():
        subset = {r["method"].split("/")[1]: r for r in rows if r["method"].startswith(backend + "/")}
        best = max(subset, key=lambda key: float(subset[key]["ndcg"]))
        delta = float(subset[best]["ndcg"]) - float(subset["original"]["ndcg"])
        changes.append(f"- {name}: {families[best]} has the highest test nDCG@10 within this backend ({float(subset[best]['ndcg']):.4f}; change from Original {delta:+.4f}).")
    text = """# ViDoRe Finance: MultiQuery and DSPy retrieval

This experiment tests whether alternative question formulations improve financial page retrieval. It reuses the previous ViDoRe Finance corpus and Qwen3-VL-Embedding-2B page vectors. GPT-4o-mini generates query alternatives using LangChain MultiQueryRetriever or DSPy, before and after MIPROv2 instruction optimization. No answer reader or judge is evaluated.

**Open `ViDoRe_Query_Expansion.ipynb` for the executed showcase.** It displays the data, tuning, frozen settings, results, examples and validation. Its cells read saved results and make no API calls.

## Scope and evaluation

Every query searches all **2,942 pages**. All **309 English questions and their five translations** are processed: 1,854 variants. Conservative relevance-signature grouping keeps translations together. The English split is 31 training, 31 validation and 247 optimizer-held-out test questions; other languages follow the same groups.

English validation nDCG@10 selects the settings. The headline results below use the **247 English test questions**. These questions appeared in earlier project experiments, so the test is held out from this optimizer rather than a newly unseen benchmark. Complete-English and multilingual aggregates include development questions and are labeled descriptive in the notebook.

## Results

TABLE

FINDINGS

Expansion modestly improves BM25 under some configurations, but it does not improve held-out nDCG@10 for the dense OCR, image or OCR/image-fusion baselines. Optimized prompts do not consistently beat the initial DSPy prompt. These observations do not establish that expansion never helps: the notebook shows per-question gains/losses, cutoff sensitivity and paired intervals.

## Tuning and controls

| Parameter | Values tested |
|---|---|
| BM25 k1; b | 0.8 / 1.2 / 1.5 / 2.0; 0 / 0.5 / 0.75 / 1 |
| Dense OCR/image fusion OCR weight | 0 / 0.25 / 0.5 / 0.75 / 1 |
| MultiQuery prompt | General / financial |
| Alternatives used | First 1 / 3 / 5 of the same five-query output |
| Candidate depth per query | 20 / 50 / 100 |
| Query-level RRF constant | 20 / 60 / 100 |
| Original-question weight | 0 / 0.5 / 1 / 2 |
| DSPy MIPROv2 | Seeds 7 / 19; six candidates; 12 trials plus default evaluation per seed |

The run evaluates 21 baseline and 2,160 expansion settings on validation only. Common baseline settings freeze at BM25 k1=0.8, b=0.75 and OCR/image weights 0.25/0.75. A 31-question validation set can overfit this grid. `protocol.json` and `selection.json` preserve the rules and selected configurations.

DSPy optimizes a query-generation instruction using BM25 retrieval nDCG, not answer accuracy. It uses three alternatives plus the original, candidate depth 100 and RRF constant 60 during optimization. Final programs have zero demonstrations; embedding weights are unchanged. The selected instruction transfers to the other backends.

The actual LangChain `MultiQueryRetriever.generate_queries()` interface supplies alternatives. A shared ranked-search engine retrieves candidates and applies weighted reciprocal rank fusion. This replaces the stock unordered document union. OCR/image fusion combines two dense rankings; it is not BM25/dense hybrid retrieval. Exact cosine search is used, with no approximate-neighbor index. The prior 4,096-token OCR and 786,432-pixel image caps are retained.

`matched_dspy_control.csv` compares initial and optimized prompts with identical downstream parameters. `previous_setting_controls.csv` preserves BM25's previous defaults and 50/50 dense fusion.

## Metrics and limits

All methods report Hit, Recall, Precision, truncated MRR, linear-grade nDCG, all-judged-page hit and complete-answer-page hit at k=1, 5, 10 and 20. Complete-answer-page hit means retrieving any grade-2 page and applies only to eligible questions. Recall measures judged-page coverage; retrieving all judged pages can require alternative sources that are not all necessary to answer.

These page scores do not establish fact-level completeness, answer correctness or groundedness. Existing draft span annotations are not promoted to validated ground truth. Unjudged pages are treated as nonrelevant. Paired intervals are pointwise group bootstraps, with no multiple-testing correction or report-level resampling.

## Runtime and validation

COST

API generation latency, GPU batch-embedding time and cached CPU lookup/fusion time are reported separately. Cached ranking time excludes generation and encoding and is not end-to-end latency. Page embeddings are reused, so no new page-indexing time is claimed. All generators request five alternatives even when a configuration uses one. Format repairs are counted in the manifest.

AUDIT

The 12-input fresh GPU embedding check passed; see `encoder_spot_check.json`. Versions, source hashes and pinned data/model revisions are recorded in the manifest.

## Files and reproduction

- `experiment.py`, `bridges.py`, `api_runtime.py`: full experiment, framework interfaces and checkpointed API transport.
- `benchmark_core.py`: reused pinned data loader, encoder, BM25 and metrics.
- `audit_results.py`, `supplement.py`, `check_encoder.py`, `tests/`: independent checks and supplementary comparisons.
- `results/`: full top-20 rankings, qrels, generated queries, compressed per-query metrics, summaries, prompts, settings and sanitized usage.

Use Python 3.12, matching CUDA torch/torchvision builds, and `pip install -r requirements.txt`. Configure `OPENAI_API_KEY` locally or the Windows-encrypted `~/AppData/Local/BarclaysRAG/openai-key.xml`. Never upload credentials. Run from this folder:

```powershell
.\\run_all.ps1
```

This downloads the pinned sources, builds missing embeddings, and makes billable API calls. To reuse existing artifacts, provide `-Python`, `-DataDir`, `-ModelDir`, `-VectorsDir` and `-CacheDir`. Use a fresh experiment cache if prompts/model/protocol change. GPU/API reruns need not be bit-identical.

`per_query.csv.gz` can be read directly with `pandas.read_csv`. Model weights, source reports, embedding caches and credentials are excluded. Upload the extracted `vidore_query_expansion/` folder alongside the previous benchmark folder; retain existing projects.

Sources: [dataset](https://huggingface.co/datasets/vidore/vidore_v3_finance_en), [Qwen model](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [MultiQueryRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever), [MIPROv2](https://dspy.ai/current/api/optimizers/MIPROv2/), [GPT-4o-mini pricing](https://developers.openai.com/api/docs/models/gpt-4o-mini). Source data/models retain their own terms.
"""
    cost = (f"Recorded unique successful requests: **{manifest['api_requests']:,}**. "
            f"Estimated recorded-usage cost: **${manifest['estimated_api_usd']:.4f}**, including optimization, "
            "at $0.15/M input and $0.60/M output tokens without cache discounts. "
            "This excludes interrupted/unrecorded calls and is not a billing statement.")
    checks = (f"All eight behavioral tests passed. The independent audit verified **{audit['metric_rows_independently_checked']:,} "
              f"per-query metric rows** and **{audit['headline_summary_rows_checked']} headline summary rows**. "
              "No relevance-signature group crosses the train/validation/test boundary. "
              "Paired comparisons use 2,000 relevance-group bootstrap resamples; intervals are pointwise and do not resample reports.")
    for old, new in (("TABLE", "\n".join(table)), ("FINDINGS", "\n".join(changes)), ("COST", cost), ("AUDIT", checks)):
        text = text.replace(old, new)
    (ROOT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build()
