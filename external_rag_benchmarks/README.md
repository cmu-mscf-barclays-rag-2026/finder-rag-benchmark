# FiQA and LongBench v2 retrieval experiments

External evaluation of BM25, Qwen dense retrieval and rank fusion, extending the team's Legal RAG experiments.

Open `External_Benchmarks.ipynb` for the showcase and `RESULTS.md` for the measured findings. The revised LongBench experiment compares 2,048 tokens, 8,192 tokens and all top-40 retrieved chunks. Results are saved for viewing without rerunning models.

## The two evaluations

**FiQA:** search the complete 57,638-entry BEIR corpus. A fixed hash selection takes 50 official development and 100 official test questions. Compare document Hit, Recall, Precision, MRR and nDCG at 1/5/10. Multiple relevant documents are judged relevant answers; they are not necessarily facts that must all be returned to construct one answer.

**LongBench v2:** a small adaptation of the official benchmark. The original hash sample contains ten short Single-Document QA and ten short Multi-Document QA items: six project development questions and fourteen project test questions. This budget follow-up reuses those previously inspected questions. Each question searches only its supplied context. For BM25, Qwen dense and equal-weight hybrid, hold the top-40 ordering fixed and compare 2,048 tokens, 8,192 tokens and all retrieved text. A no-context condition uses the same reader. This is an exploratory follow-up, not a new untouched test or the full official 503-question evaluation.

Neither dataset provides the exact source-span annotations from the Legal RAG pilot. FiQA uses document relevance and LongBench uses answer correctness. No source-span completeness scores are fabricated.

## Methods and controls

- Sparse: positive-IDF BM25, k1=1.5 and b=0.75; lowercase word tokens.
- Dense: Qwen3-Embedding-0.6B, pinned revision, 512-token embedding window, normalized last-token vectors, exact CPU cosine search. Encoding uses CUDA when available.
- Fusion: weighted reciprocal rank fusion, constant 60 and up to 100 candidates from each retriever. FiQA shows equal fusion and a separate development-selected weight; it never tunes on test results. LongBench uses fixed equal fusion.
- FiQA indexes each original corpus entry as one retrieval unit. Dense inputs longer than 512 Qwen tokens are truncated; the measured truncation rate is reported.
- LongBench uses 256-token chunks with 32-token overlap, measured using the same pinned MiniLM counting tokenizer as earlier work. This tokenizer does not generate embeddings. The revised experiment uses the same top forty chunks across all budgets. All retrieved text means those forty complete chunks, not the entire original source.
- Answering: Qwen2.5-1.5B-Instruct, pinned revision, greedy generation, up to 32 new tokens. Every complete prompt is counted with the reader tokenizer before generation, with space reserved for output. The follow-up checks against its supported 32,768-token capacity and never silently truncates prompts. Gold letters are used only for scoring. Invalid responses count as incorrect.

## Run

Python 3.10+; a CUDA GPU is recommended. Install a suitable PyTorch build for your GPU before the other requirements if necessary. A CPU can run the code but will be slower.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_fiqa.py
python run_longbench_budgets.py contexts
python run_longbench_budgets.py answer
python audit_results.py
python audit_budget_results.py
```

Pinned processed input files are included. `prepare_data.py` rebuilds them from Hugging Face and downloads the full raw LongBench file to the ignored cache. The data revisions and selection hashes are in `data/selection_manifest.json`.

The budget runner reuses the original candidate rankings in `results/longbench/contexts.json`; no new embeddings are required. The original top-10 experiment remains available through `run_longbench.py retrieve` and `run_longbench.py answer`. Its files are retained as an auditable baseline.

`config.json` preserves the original recorded settings and FiQA signatures. **`longbench_budget_config.json` controls the revised budget comparison.** New outputs are in `results/longbench_budget_sweep/`, including the context construction, prompt-capacity checks, 200 predictions, comparison and paired wins/losses. All 200 generation conditions were run locally; the original answers are used only for comparison checks.

Models and embedding caches default to `.cache/`. Optionally set `RAG_MODEL_CACHE` to an existing Hugging Face model-cache directory to reuse downloads. GPU model weights, raw downloads, and vectors are excluded from the upload.

Completed embedding blocks and generated responses are reused on resume. Use a fresh copy with new results/cache directories if you change the model or protocol. Saved outputs represent the exact recorded configuration, not a promise that new hardware will reproduce timing exactly.

## Optional school API answering

`run_openai_reader.py` can reuse the saved LongBench contexts with an OpenAI model allowed by the school project. See `OPENAI_SETUP.md`. Start with a one-request dry run and a one-request connection test. An accepted invitation and locally configured project key are required. No OpenAI calls have been made in the saved results.

## Upload folder

Upload this entire `external_rag_benchmarks` folder at repository root beside `legal_rag_spans`, `legal_rag_hierarchy` and `qwen_embeddings`. Extract the outer ZIP first. The notebook, code, processed data, tests, manifests and results are included.

See `DATA_AND_MODELS.md` for source attribution and licensing. See `METRICS.md` for interpretation and limitations.
