# GPT-5.4 HyDE transfer: FiQA and LongBench v2

This experiment transfers the prior legal retrieval setup to two complete public test collections. It uses the prior focused160 profile with domain wording adapted before seeing outcomes. Original96 is omitted: 94/100 earlier legal outputs were cut off by that length cap. No weights, prompts or methods are selected using the new test outcomes.

## Scope and metrics

* **FiQA (BEIR)**: all 648 test queries, the full 57,638-document corpus and all 1,706 positive query-document judgments. Primary metric: nDCG@10. Hit@5 means at least one judged-relevant document, while Recall@5 counts the fraction of all judged-relevant documents. They are different on this dataset. The corpus is not restricted to relevant documents. Dense representations use the first 256 MiniLM tokens of each full document; BM25 indexes full documents. All returned documents, including coincident query/document ID strings, are evaluated; no BEIR identical-ID filter is applied.
* **LongBench v2**: all 503 multiple-choice questions, across six domains. Each question searches only its own full context, split into 224 MiniLM token chunks with 32-token overlap. No leading-context cap or difficulty/length filtering. Primary metric: answer accuracy with a fixed GPT-5.4 reader given the top five chunks. There are no official chunk relevance labels, so Hit@5 cannot be computed. This is a custom RAG adaptation, not a reproduction of the official full-context leaderboard or its 512-token RAG setting. The no-context control uses the same reader prompt with an empty text section.

The Hugging Face distribution labels its single LongBench v2 split `train`; those 503 public benchmark items are used only for evaluation here. No model is fitted on them. The validation implementation supports replay of all 1,509 dense rankings. The completed audit replayed 219/1,509 dense rankings using retained caches, all 2,515 RRF rankings, and all 5,030 logical predictions. Main evaluation is complete; exhaustive dense replay remains incomplete after cache loss. See completion_status.json.

Dense validation first checks exact ranks, then allows only score-order/cutoff differences within 1e-6 for float32 near ties (initial raw-graph query encodings versus the equivalent O2 query cache). Every such case is listed in `longbench2_retrieval_validation.json`; saved rankings and predictions are never rewritten by validation. RRF ranks must match exactly.

Nine retrieval methods are saved: question dense, BM25, standard RRF hybrid at dense weights .25/.50, pure focused HyDE dense, question/HyDE mixed dense, pure HyDE hybrid at .25, and mixed HyDE hybrid at .25/.50. LongBench also has a no-context reader control.

All dense vectors use the same pinned `all-MiniLM-L6-v2`, normalized. The mixed query is `normalize(.5*question + .5*HyDE)`. BM25 always uses the original question. Fusion uses RRF k=60 and each list's top 100. LongBench HyDE and retrieval receive the question only; only the reader sees the four choices. Neither sees the answer label. Identical complete reader prompts are executed once and shared across methods, so some observations are deliberately dependent.

Encoding currently uses offline PyTorch CPU float32, with local pinned weights/tokenizer, masked mean pooling and L2 normalization. Earlier equivalent float32 caches were computed using PyTorch and ONNX graphs; these are reused. ONNX/PyTorch compatibility checks showed coordinate differences below 1e-6. A safety review later rejected an unknown Microsoft telemetry connection, so the active encoder no longer imports ONNX Runtime at all. Hugging Face telemetry and network access are explicitly disabled for encoding. The resumed PyTorch backend matched 32 existing cached vectors to a maximum coordinate difference of 2.91e-7; see `offline_backend_compatibility.json` and `telemetry_policy.json`. No quantized/bfloat16 embeddings are used. Identical chunk texts share an encoding cache. Use `setup_encoder.py` to restore the pinned PyTorch weights and tokenizer.

Both generator and reader use `gpt-5.4-2026-03-05`, reasoning none, temperature 0, store false, no tools. Output caps: HyDE 160, reader 32. API calls use six workers, no automatic retries, a durable SQLite checkpoint, and a $30 run-wide protection. The cost estimate uses the previous experiment's conservative $2.50/$15 per million input/output tokens, without cache discounts; it is not a billing statement.

Six transport attempts failed without an OpenAI response/usage record during the long run. After a successful read-only model-access probe from a refreshed worker, the missing logical calls were manually reissued. Original attempts remain archived with a $0.097505 total upper-cost allowance; they are never erased or treated as confirmed free. Successful outputs are reused. Subsequent reader workers are refreshed between complete batches, with no requests interrupted by that refresh. `transport_recovery.json`, `unresolved_transport_attempts.json` and `cost_accounting.json` document this exception to the no-automatic-retry workflow.

## Reproduce

Keep this directory at the repository root. Restore the exact versions from `environment.json` (including CPU PyTorch, sentence-transformers, OpenAI, numpy, pandas, scipy and scikit-learn), plus pyarrow/httpx/socksio, ijson and onnxruntime. Configure the authorized project key in the ignored repository `.env.local` as `OPENAI_API_KEY` using a secure credential setup; never commit that file.

```bash
python person3_cross_dataset/download_data.py
python person3_cross_dataset/setup_encoder.py
python person3_cross_dataset/run.py protocol
python person3_cross_dataset/run.py generate
python person3_cross_dataset/run.py fiqa
python person3_cross_dataset/run.py index_longbench
python person3_cross_dataset/run.py longbench
python person3_cross_dataset/run.py answers
python person3_cross_dataset/run.py summarize
python person3_cross_dataset/validate.py
```

`index_longbench` is optional; `longbench` builds any missing index. Successful API calls are never repeated by a resumed run. Failed or unknown attempts stop automatically and require explicit investigation. Missing local indexes can be rebuilt without API use. The fixed output path rejects protocol changes; use a new directory/version for a different protocol. Full contexts and dense corpus indexes are excluded from the result package; pinned downloads and hashes reproduce them.

## Statistical interpretation

Six candidate/control comparisons per dataset were recorded in `protocol.json` before test results. FiQA uses two-sided paired sign-flip permutation tests of nDCG@10 (100,000 draws, seed 20261028). LongBench uses exact paired McNemar tests of answer accuracy. Holm correction covers all 12 comparisons together. Other metrics, subgroups, and the observed best method are descriptive. Historical legal experiments are outside this correction family. Public benchmarks can overlap model pretraining, and a good LongBench score can reflect the reader's own knowledge; the no-context control is therefore important. Differences between the financial, legal and long-context results cannot be attributed only to domain.

## Validation after environment recovery

The complete 503-question experiment can be checked with `python person3_cross_dataset/validate.py --available-dense` when old embedding caches are missing. This independently checks all answer predictions, source spans, reader prompts, reported accuracies, API accounting, and all RRF ranks; dense ranks are recomputed only where a complete embedding cache is present. Exact coverage and missing question IDs are recorded in `results_validation.json` and `longbench2_available_retrieval_validation.json`. `report.py --available-dense` reports that scope explicitly. The default validation command still requires all 1,509 dense rankings. No evaluation outputs or scoring rules are changed by this validation mode.

Encoder checkpoint ZIP parts preserve exact float32 vectors. Restore each part using `python person3_cross_dataset/encoder_checkpoint.py --restore PATH`, then rebuild missing per-context indexes with `run.py index_longbench`. See `encoder_checkpoints.json` for retained checkpoint identities.
