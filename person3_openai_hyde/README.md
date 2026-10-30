# Cheryl: OpenAI HyDE generator experiment

This run replaces the local hypothetical-passage generator with the fixed OpenAI snapshot `gpt-5.4-2026-03-05`, using the same previously inspected 100 Legal RAG Bench questions, 4,876 passages, and MiniLM weights. It tests original prompt / 96 output tokens and focused prompt / 160 output tokens. The focused profile changes prompt and length together. API prompts contain only the question and prompt instructions, never gold answers or passages.

Each profile evaluates pure HyDE dense, equal normalized question/HyDE vector mixing, pure HyDE hybrid with 25% dense, mixed hybrid with 25% dense, and mixed hybrid with 50% dense. Controls include original-question dense, BM25, and standard hybrid at both 25% and 50% dense. RRF constant 60; each input has depth 100; stable corpus-index tie breaking. The same-weight hybrid controls distinguish generation effects from weight changes.

## Run

Install `openai`, `sentence-transformers`, `huggingface-hub`, `torch` (CPU is sufficient), `numpy`, `pandas`, `scipy`, `scikit-learn`; some proxy environments also need `socksio`. Exact versions for this execution are in the output `environment.json`.

Configure an untracked `.env.local` in the repository root containing `OPENAI_API_KEY`. Never commit this file. Use the Group 004 project credential. The runner reads it without printing or copying it to outputs.

Place the dataset's corpus.jsonl and qa.jsonl in data/legal_rag_bench; the runner verifies their SHA256 against the pinned revision recorded in protocol.json. Run from the repo root:

```
python person3_openai_hyde/run.py prepare
python person3_openai_hyde/run.py pilot
python person3_openai_hyde/run.py generate
python person3_openai_hyde/run.py evaluate
```

The pilot uses five IDs chosen by SHA256 of the query ID before inspecting outcomes, with both profiles. Generation resumes from saved outputs. This local execution uses four concurrent API requests; SDK automatic retries are disabled. A shared lock protects all checkpoint and budget updates. Any failed/unknown attempt is retained and stops subsequent automatic retries. Generation timings are concurrent request durations, not isolated latency benchmarks.

The nominal USD 1 run guard uses conservative pre-request reservations and token-based standard-price estimates. It is not an organization billing cap, and actual organization billing remains authoritative. Price inputs, output tokens, incomplete statuses, model identifiers, prompts, and outputs are all recorded. Empty outputs stop generation; nonempty length-capped outputs are retained and audited.

`evaluate` uses no generation API calls and saves all top-100 rankings, per-question metrics, summary metrics at k=1/3/5/10, and twelve predefined exact McNemar comparisons with Holm adjustment across all twelve. MRR/nDCG comparisons are descriptive. Hit@5 is evidence retrieval, not final-answer accuracy. Prior SmolLM3/LFM2.5 aggregate observations lack raw outputs and are historical context only.

This is exploratory reuse of an inspected question set. It does not establish held-out generalization. No final-answer evaluation is included. OpenAI tokenization, chat formatting and decoding controls differ from the local-model experiments, so this is not a perfectly isolated model-only comparison.
