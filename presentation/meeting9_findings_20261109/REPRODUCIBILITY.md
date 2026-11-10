# Reproduction and source map

## Included-data reproduction

`reproduce.py` reads only the accompanying CSV/JSON files. It checks all 249 paired rows, means and win/loss/tie counts; recreates 100, 000 shared sign vectors with seed 20261109; verifies their SHA256, all 10 extreme counts and raw p-values; and recomputes Holm 10. It also checks that reader accuracy is marked unusable, then draws four figures.

```bash
cd presentation/meeting9_findings_20261109
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python reproduce.py
```

Output goes to`reproduced/`, leaving published figures and original research artifacts untouched. `--out PATH` selects another directory. This validates score-table arithmetic and plots; it does not independently rerun retrieval or establish source-level statistical independence.

`requirements.txt` pins the three packages used for this reproduction. `environment.json` records other observed execution versions; it is not a universal full-experiment environment lock. Font rendering and local timing may differ across systems.

## Experiment source

Paths below are relative to the repository root:

| Source | Purpose |
|---|---|
| [person3_meeting9_ablation](../../person3_meeting9_ablation) | Length × constraint prompt templates and evaluation |
| [person3_meeting9_transfer](../../person3_meeting9_transfer) | Public dataset download, frozen60/249 cohorts, dense/HyDE transfer |
| [person3_meeting9_fullpage](../../person3_meeting9_fullpage) | Complete OCR chunking and page max scoring |
| [person3_meeting9_refinement](../../person3_meeting9_refinement) | Frozen four-candidate unfiltered fusion selection |
| [person3_meeting9_constraints](../../person3_meeting9_constraints) | Fixed question-only bank router and all-candidate scoring |
| [person3_meeting9_constrained_hybrid](../../person3_meeting9_constrained_hybrid) | Six fixed bank-constrained hybrids |
| [person3_meeting9_evidence](../../person3_meeting9_evidence) | Source-quote availability prototype and draft annotations |
| [person3_meeting9_workflow](../../person3_meeting9_workflow) | Durable API ledger, HNSW, memory and stage audits |
| [person3_meeting9_followup](../../person3_meeting9_followup) | Missing2×2 control, finite tests, fixed reader, rejected raters, cost and independent audits |
| [person3_hyde_pilot/run.py](../../person3_hyde_pilot/run.py) | Earlier frozen Short template reused by transfer |

The source map links use the repository layout; supporting`person 3_cross_dataset`and`person 3_openai_hyde`modules already exist in the base repository. Cached full experiments additionally require their saved inputs and artifacts.

## Full-experiment replay boundary

The new source is included for inspection and extension. Full replay requires the retained corpora, qrels, selections, saved rankings, model/tokenizer caches, embeddings and generation/reader outputs in the original`results/person 3_meeting 9_*`layout, plus earlier Legal/FiQA caches. These large execution artifacts and API ledgers are retained locally rather than duplicated in this concise findings publication. Raw GPT answers and pending human-review forms are also retained locally.

Public new dataset: `vidore/vidore_v 3_finance_en`, revision`7 f 432 c 176 d 82 e 27546501 ad 8064 a 713 ac 3071809`; all309 English queries and2,942 pages. Encoder:`sentence-transformers/all-MiniLM-L6-v 2`, revision`1110 a 243 fdf 4706 b 3 f 48 f 1 d 95 db 1 a 4 f 5529 b 4 d 41`, CPUfloat32, attention-mask mean pooling, L2 normalization, max256 tokens. Full-OCR chunks use224 tokens with32 overlap. Model snapshots and detailed settings remain in the included frozen protocol records.

All source files are copied from the executed workspace. In three API wrappers, the original fixed organization identifier is replaced with`OPENAI_ORG_ID`environment configuration; scientific models, prompts, scoring and output limits are unchanged. `source_manifest.json` records both original and published hashes and this operational-only transformation. Original cache-bound implementation hashes therefore refer to the untouched execution copies. Use the original execution archive for exact manifest-bound replay, or establish a separate, newly frozen run for extensions; do not rewrite old receipts to make changed code pass.

Default findings reproduction requires no credential and makes no API calls. Paid runners are separate and are unnecessary to verify these tables. Successful cached calls should not be repeated. The original study has a cumulative$7 guard; the published estimate is$4.570655 including historical reserves, not a billing statement. New paid work needs its own current ledger and reviewed budget.
