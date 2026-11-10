# HyDE transfer on ViDoRe V3 Finance EN

Fixed 60 English queries selected by SHA256 before generation; complete 2,942-page corpus from six banks' reports. Official source: https://huggingface.co/datasets/vidore/vidore_v3_finance_en at revision `7f432c176d82e27546501ad8064a713ac3071809` (CC-BY-4.0). Provided page OCR is downloaded without the 1.26GB embedded image payloads; every page's upstream cache revision and nontruncation are verified. Raw data, source manifests, and generated paragraphs are retained locally.

The exact previous FiQA Focused and Short templates transfer without outcome-driven edits. Fixed GPT-5.4 snapshot/settings and pinned MiniLM CPU float32 encoder; question-only and BM25 controls; pure HyDE and 50/50 mixes. Top-k 5/10/15/20/25. Four predeclared nDCG@10 comparisons use paired two-sided sign-flip tests (100,000 draws) and Holm adjustment over those four tests. Other metrics/subgroups are descriptive. This is a custom text-only query subset, not official full MTEB or visual retrieval reproduction.

From repository root:

```bash
.venv/bin/python person3_meeting9_transfer/download.py
.venv/bin/python person3_meeting9_transfer/run.py prepare
.venv/bin/python person3_meeting9_transfer/run.py baseline
# Main orchestrator executes pending_calls.jsonl with shared budget/checkpoint,
# then saves successful_api_outputs.json in the result directory.
.venv/bin/python person3_meeting9_transfer/run.py evaluate
.venv/bin/python person3_meeting9_transfer/validate.py
```

This runner never sends API calls or loads credentials. Successful generations are supplied by the central orchestrator. Reproduction never repeats successful paid generation. Corpus vectors resume in 128-page blocks. All ranking data, exact prompts, metadata, grade mapping, and source hashes are preserved. Gold answers, page content and relevance labels never enter generation prompts.

The page OCR is truncated to the first 256 MiniLM tokens for dense indexing; BM25 reads full OCR. The token audit records how many pages exceed that limit. Source pages can contain tables/charts whose layout is imperfectly represented by OCR. No visual embedding model or answer reader is evaluated here; the dataset remains suitable for a future matched text-versus-image comparison using the same page IDs, queries and qrels. Local query encoding/search timings exclude network generation and do not establish production end-to-end latency.

## Full English-split extension

The remaining 249 English questions were frozen in a separate directory before the 60-query HyDE retrieval results were read. This completes all 309 English queries with the same two previously fixed prompts; the extension was motivated by full-split coverage, not significance. It adds 498 generation jobs. The 60-query primary family remains unchanged; the249-query primary family separately contains the same four planned comparisons with Holm4 adjustment. Combined 309-query tables are descriptive only.

```bash
.venv/bin/python person3_meeting9_transfer/run.py prepare --cohort remaining249
.venv/bin/python person3_meeting9_transfer/run.py baseline --cohort remaining249
# Same central paid-call orchestrator, separate successful_api_outputs.json.
.venv/bin/python person3_meeting9_transfer/run.py evaluate --cohort remaining249
.venv/bin/python person3_meeting9_transfer/validate.py --cohort remaining249
.venv/bin/python person3_meeting9_transfer/merge_full.py
```

`vidore_finance_remaining249_v1/` reuses the exact cached corpus vectors from `vidore_finance_v1/vectors/`; it encodes only the new query/HyDE vectors. `stage_provenance.json` records the extension freeze and budget estimate. Do not call `evaluate` until all 498 outputs are present. The merged report never runs new significance tests on the combined sample.
