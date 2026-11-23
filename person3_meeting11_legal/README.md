# Meeting 11 Legal MultiQuery study

Controlled BM25, dense and hybrid retrieval with original questions, cached default MultiQuery, and one frozen legal-domain MultiQuery prompt. All methods share the original 4,876 passages and the saved Meeting 10 passage-disjoint 10/10/80 split. This is an exploratory continuation on a previously inspected benchmark.

The study tests three backends, three query strategies, Top-5/8/10, and evidence budgets of 5k/8k/10k `o200k_base` tokens. No reader answers are generated. The legal prompt is manually specified, not automatically optimized with DSPy or GEPA.

## Completed outputs

- Raw local checkpoints: `results/person3_meeting11_legal/v1/`.
- English report, tables and three charts: `../outputs/meeting11_20261123/` relative to repository root.
- The exact prompt, source checksums, code checksum and primary nine-test family were frozen before new generations in `protocol.json` and `freeze_receipt.json`.
- Existing default query generations and historical artifacts are immutable inputs. All 100 cached original-model responses were checked against `gpt-4o-mini-2024-07-18`, temperature0, output cap256.

## Execute or resume

Run from the repository root using an environment containing `numpy pandas scipy scikit-learn torch transformers openai tiktoken langchain-classic langchain-core matplotlib`. Model weights and the dataset must already be cached at the pinned revisions in `protocol.json`. LangChain Classic 1.0.8 supplies the original native line parser. Never run old full-benchmark generation stages to prepare this study.

```bash
python person3_meeting11_legal/run.py prepare
python person3_meeting11_legal/run.py default
python person3_meeting11_legal/run.py generate
python person3_meeting11_legal/run.py legal
python person3_meeting11_legal/run.py evaluate
python person3_meeting11_legal/run.py budget
python person3_meeting11_legal/profile.py
python person3_meeting11_legal/verify.py
python person3_meeting11_legal/report.py
```

`generate` is the only paid stage. It reuses successful request identities and saves each successful result immediately. A pending or unknown request is retained as reserved spend and is never automatically retried. Preserve the API ledger and query checkpoint together. Do not remove them to bypass a failed call. All generation outputs completed in this execution, so a normal resume needs no new requests.

The local execution uses the existing `.venv/bin/python` for offline MiniLM ranking, and `../work/meeting10_venv/bin/python` for generation, token-budget analysis, auditing and plotting. The former contains cached PyTorch/Transformers; the latter contains LangChain and tiktoken. A single environment with all declared dependencies can run every command above. The tokenizer cache is local at `../work/meeting11_20261123/tiktoken_cache/`.

For final aggregation after either ranking stage, run `evaluate` to combine both cached prompt variants before `budget`, `verify` and `report`. Individual ranking stages expose their own partial-stage score summary; the final report requires all nine methods.

## Controls and interpretation

- The encoder stays pinned to MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32, mean pooling, normalized vectors and a 256-token encoder limit. This encoder limit is independent of the evidence-delivery budget.
- Original-query BM25 and dense top100 rankings replay exactly. All 300 historical BM25 MultiQuery per-k rankings replay exactly at k5/8/10.
- Query fusion preserves the original implementation: truncate each query route to k, then RRF60 to k. Backend fusion is a separate step: each hybrid query first combines BM25/dense top100 rankings at equal weights.
- The fixed-top100 query-fusion sensitivity analysis is secondary and must not replace the primary historical convention based on which score is larger.
- The primary test family is nine exact paired McNemar contrasts on Hit@5. All nine are Holm-adjusted together. Other depths, budget cells, MRR/nDCG and candidate-union diagnostics remain descriptive.
- Full delivered Hit/Recall requires the entire labeled passage. Partial touch is separately recorded and is not evidence-span sufficiency or final-answer correctness.

## API safety and costs

Only the existing ignored local `.env.local` is used for Group004 credentials, with permissions600. Do not display or export it. The reused budget gateway performs a read-only snapshot-access check before new generation. It disables automatic retries, uses durable pre-request reservations, and fixes a $0.25 new-spend guard and the existing $7 cumulative guard. Pricing was checked on the official model page on 23 November 2026: $0.15 input / $0.60 output per million tokens, ignoring cached-input discounts. API usage is saved per successful request.

## Audit and report replay

`verify.py` independently constructs the BM25 matrix, recomputes dense and hybrid rankings from saved vectors, rebuilds both fusion conventions, checks primary tests and Holm9, reconstructs every packed context, and reconciles the API ledger. It makes no API requests.

The exported `reproduce.py` rebuilds the English report and charts from its included tables:

```bash
cd ../outputs/meeting11_20261123
python -m pip install numpy pandas matplotlib
python reproduce.py --input . --output .
```

Raw generation text, the private API ledger, embeddings and full ranking checkpoints remain local; the exported tables support aggregate and figure reproduction without generation calls.
