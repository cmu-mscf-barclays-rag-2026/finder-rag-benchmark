# Week 9 — evidence sufficiency on financial reports

Florence's work from the **November 9, 2026** meeting: a question-aware evidence
evaluation prototype, a small annotated question set, and controlled retrieval
and context-efficiency comparisons on the finance part of ViDoRe/MTEB.

Start with the [meeting brief](MEETING_BRIEF.md), [measured findings](results/findings.md), and
[annotation protocol](annotations/PROTOCOL.md). The
[review packet](annotations/REVIEW.md) links each drafted requirement to its
source PDF pages; the [corpus audit](results/corpus_audit.json) records OCR
lengths and annotation/qrel differences.

## What this adds

- Fact recall, all-required-fact coverage, facts per 1,000 delivered tokens,
  marginal evidence per unique page, and the minimum sufficient budget in the
  tested grid. Alternative sources and jointly necessary evidence are distinct.
- Ten source-grounded **draft** questions with twenty required facts, exact
  source offsets, query classifications, and a review log. Default scoring
  rejects unreviewed annotations; `--allow-draft` explicitly enables the pilot.
- A revision-checked adapter for ViDoRe V3 Finance EN, including every page,
  English question and original positive relevance label. Images remain
  available upstream, but this experiment uses the supplied OCR text.
- BM25, MiniLM and Qwen256 full-corpus baselines, Top-5/10/20, token budgets of
  2,000/4,000/6,000/8,000 and sibling policies 0/0, 1/1, 2/1. All methods use
  the same pages, questions, relevance labels, delivery ruler and packing rule.
- Per-question retrieval/delivery traces, paired comparisons, query/content
  type breakdowns, local timing observations and examples where relevance
  metrics disagree with draft fact coverage.

The prototype is implemented and measured; **human validation is still needed**.
No visual-retrieval, HNSW, external embedding API or generated-answer accuracy
result is claimed. Those are follow-up experiments, not completed work.

## Dataset and controls

Source: [ViDoRe V3 Finance EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en),
revision `7f432c176d82e27546501ad8064a713ac3071809`, `test` split.
The actual downloaded set contains 2,942 pages in six 2024 banking reports,
309 English queries and 1,545 translated queries. Only English is evaluated;
translations are not counted as independent questions. The associated MTEB
task is [Vidore3FinanceEnRetrieval.v2](https://docs.mteb.org/overview/available_tasks/retrieval/).
These are local OCR baselines, not an official MTEB leaderboard submission.

Each corpus page is a retrieval unit. The full 2,942-page corpus is searched,
including nonrelevant pages and 19 pages with empty OCR. Empty pages remain in
the retrieval corpus but cannot consume context or establish text support.
Only the question text enters retrieval; answers, qrels, and fact annotations
are evaluation inputs. Published queries containing answer hints remain in the
full baseline and are flagged in the annotation protocol.

MiniLM and Qwen256 both use 256-token **document** encoding limits with their
own tokenizers. MiniLM uses mean pooling; Qwen uses normalized last-token
pooling and a finance retrieval instruction for queries, with no instruction
on pages. Both models search by exact normalized dot product. All query texts
fit within the respective query windows. This compares model configurations;
equal token counts from different tokenizers do not imply identical visible
characters. `--method qwen` provides a separate 4,096-token document window;
that longer run was stopped before ranking and has no reported score.

**BM25 sees full OCR text; both dense configurations see only page prefixes.**
MiniLM truncates 2,764 of 2,942 pages (93.9%) at its 256-token limit. Treat
sparse-versus-dense differences as configuration comparisons, not evidence
that sparse retrieval intrinsically wins on finance. A longer-window dense
run or matched-prefix BM25 control is needed to isolate that factor.

Delivery uses the pinned Week 6 MiniLM measurement tokenizer independently
of the encoder. Every method packs complete pages, skips pages that do not
fit, and continues in candidate order. No partial-page baseline advantage is
allowed. Page text is unmodified OCR; titles, bank names, answers and facts
are not prepended. Returned tokens exclude any reader prompt or answer budget.

For siblings, candidates are seed, previous page, next page, second previous
page as applicable; overlapping page IDs are removed before token accounting.
Physical PDF page numbers define adjacency. Corpus IDs are **not** PDF page
order: some upstream pages are ordered lexicographically. Expansion never
crosses a report boundary. There is no backfill from seeds beyond the chosen k.

Ranked metrics are Hit/Recall/MRR/nDCG at the seed cutoff. Delivered metrics
use the entire packed context (including siblings), so delivered nDCG uses
the number of delivered pages rather than the seed cutoff. The all-judged-
relevant rate remains a relevance diagnostic, **not** an evidence-sufficiency
metric. Multi-label query/content-type groups overlap.

Summary metrics are unweighted means across questions. Fact recall is a macro
average, and evidence efficiency is the mean of each question's facts per
1,000 tokens, rather than a pooled ratio. Sufficiency is the fraction of
questions whose annotated required facts are all covered.

## Reproduce

Use the existing `.venv-week6` environment and pinned local models. The
[Week 9 requirements](requirements.txt) record the direct package versions
used with Python 3.12.5, including Sentence Transformers and the download
dependencies. No additional packages were installed for the measured runs.

For a fresh checkout, create the environment and populate the model cache
once (these setup commands require network access):

```sh
python3.12 -m venv .venv-week6
.venv-week6/bin/python -m pip install -r week9_evidence/requirements.txt
.venv-week6/bin/hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --cache-dir .cache/models
.venv-week6/bin/hf download Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --cache-dir .cache/models
```

All model loads in the experiments are offline; a missing model fails
instead of downloading silently. Run the following commands from the repository root:

```sh
# Dataset download: text/labels only, not the 1.2 GB of page images.
.venv-week6/bin/python -m week9_evidence.vidore --output .cache/vidore_v3_finance_en

# Choose new output directories; completed outputs are never overwritten.
.venv-week6/bin/python -m week9_evidence.run --dataset .cache/vidore_v3_finance_en --method bm25 --output week9_evidence/results_new/bm25
.venv-week6/bin/python -m week9_evidence.run --dataset .cache/vidore_v3_finance_en --method minilm --device cpu --output week9_evidence/results_new/minilm
.venv-week6/bin/python -m week9_evidence.run --dataset .cache/vidore_v3_finance_en --method qwen256 --device mps --output week9_evidence/results_new/qwen256
.venv-week6/bin/python -m week9_evidence.check_qwen --dataset .cache/vidore_v3_finance_en --vectors week9_evidence/results_new/qwen256/vectors.npz --output week9_evidence/results_new/qwen256_cpu_check.json

# Repeat this scoring command for bm25, minilm and qwen256.
.venv-week6/bin/python -m week9_evidence.score_pilot --dataset .cache/vidore_v3_finance_en --run week9_evidence/results_new/bm25 --annotations week9_evidence/annotations/finance_pilot.jsonl --allow-draft --output week9_evidence/results_new/pilot_bm25

# After all three retrieval runs and pilot scoring runs complete:
.venv-week6/bin/python -m week9_evidence.report --dataset .cache/vidore_v3_finance_en --results week9_evidence/results_new

.venv-week6/bin/python -m unittest discover -s week9_evidence/tests -v
.venv-week6/bin/python -m unittest discover -s week8_retrieval/tests -v
```

Use `--device cpu` when MPS is unavailable. Cache keys include model revision,
device, dtype and package versions. GPU access may need permission outside
the execution sandbox. Interrupted Qwen encoding resumes from cached vectors;
use a new/empty results directory. `--rankings` can reuse a JSONL ranking file
to re-evaluate delivery without model inference. `score_pilot` checks that its
recomputed delivery exactly matches the saved context IDs and token counts.

The data downloader checks the `x-revision` response header on **every** page,
rejects truncated cells, verifies row continuity and records local SHA-256
checksums. The viewer serves its current revision; if it changes, the adapter
fails closed rather than silently mixing versions. Preserve the ignored data
cache or deliberately review/update the revision. Pinned raw Parquet files
remain available in the upstream repository for archival reconstruction.

The annotation builder can reproduce the authored draft against the pinned
OCR without any generation calls:

```sh
.venv-week6/bin/python -m week9_evidence.annotations.build_pilot --dataset .cache/vidore_v3_finance_en --output /tmp/finance_pilot_rebuilt.jsonl
```

Result CSV/JSONL/Markdown files are small enough to review in Git. Raw data,
models and per-input embedding caches stay in ignored `.cache`; generated
`vectors.npz` files are also ignored. Dataset annotations are distributed as
CC-BY-4.0; original report text/images retain their publishers' rights as stated
in the [dataset card](https://huggingface.co/datasets/vidore/vidore_v3_finance_en).

## Qwen end-token correction

The inherited Week 6 encoder slices a token list after special tokens have
been added. For truncated pages, this removes Qwen's final `<|endoftext|>`
token, so last-token pooling uses an ordinary content token. Short queries
retain their final special token. A first finance run using that path failed
an independent CPU comparison and is isolated in `results/invalid_qwen256/`.
It is **not a valid model-performance result** and is excluded from reports.

The new [finance encoder](qwen.py) lets the tokenizer truncate while preserving
the special-token template and asserts that the final token remains intact.
This checkpoint's appended `<|endoftext|>` differs from its configured chat
`eos_token_id`; the check validates the actual suffix. Corrected vectors use
a distinct cache identity and cannot reuse the affected page vectors.
KV caching is disabled because this is embedding, not autoregressive generation.

The [six-input smoke check](results/qwen_smoke_check.json) compares batched MPS
outputs with independent single-input CPU outputs; minimum cosine agreement
is above 0.9999999. This is a diagnostic sample, not a full-corpus equivalence
proof. Recheck saved vectors after a new full run with:

```sh
.venv-week6/bin/python -m week9_evidence.check_qwen --dataset .cache/vidore_v3_finance_en --vectors week9_evidence/results/qwen256/vectors.npz --output week9_evidence/results/qwen256_cpu_check.json
```

`results/source_snapshot/run_initial.py` preserves the runner hash used for
BM25 and MiniLM. `run_before_eos_fix.py` preserves the invalid Qwen runner,
and `run_corrected.py` preserves the corrected Qwen runner before a metadata-only
wording cleanup. The frozen MiniLM manifest's note about a different Qwen
window refers to the original 4,096-token option; the reported comparison
uses Qwen256. Numeric configuration fields and measured outputs are unchanged.
The current Qwen256 run uses batches of 32 (8,192 padded tokens), float32,
and the corrected tokenizer path. Historical Week 6/8 outputs are unchanged;
window comparisons that truncate inputs need a separate audit before reuse.
