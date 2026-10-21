# Cheryl: stronger HyDE generators and preliminary answer quality

This is a controlled, local follow-up to `person3_hyde`. Use the saved tables in
`results/person3_stronger_hyde/`; no API key is required for reproduction.

## Fixed experiment

- Legal RAG Bench snapshot `db0b31dc6d195ce9916897e1ac5e4e6209736c8a`: 100 questions, 4,876 title+text passages. Exact SHA256 checks are enforced.
- MiniLM-L6-v2 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized embeddings, maximum 256 tokens.
- Original-question BM25 k1=1.2, b=0.75; RRF constant60, depth100, BM25/dense weights75%/25%.
- One hypothetical passage per question, original evidence-passage prompt, greedy decoding, maximum96 generated tokens, repetition penalty1.2 and no-repeat3-gram rule.
- Qwen3-0.6B and Qwen3-1.7B use their official chat templates with thinking disabled. Generator arithmetic is CPU bfloat16; embeddings remain float32. Model revisions and configuration are in `experiment_contract.json`.
- FLAN's matched evidence-passage/96-token/greedy outputs are reused; no FLAN generation is repeated. They are re-embedded under the current environment. Historical multi-HyDE maximum39% is a different, selected configuration and is not the matched FLAN control.

The task is to test the generator, not retune the retriever. Chat templates, model tokenizers and architectures necessarily differ; a96-token cap is not an identical word budget. Higher parameter count does not establish higher quality on this task. Greedy decoding and the96-token cap match the previous control; they are not claimed optimal for Qwen. The retained initial three-query float32 timing pilot is excluded from the primary bfloat16 outputs.

## Reproduce

Python3.12; CPU with bfloat16 support recommended. Create and activate a virtual environment, then install CPU PyTorch and the pinned remaining dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
pip install -r person3_stronger_hyde/requirements.txt
python person3_stronger_hyde/setup_data_models.py
python person3_stronger_hyde/run.py prepare
python person3_stronger_hyde/run.py generate --model qwen06
python person3_stronger_hyde/run.py generate --model qwen17
python person3_stronger_hyde/run.py retrieve
python person3_stronger_hyde/run.py answer
python person3_stronger_hyde/summarize.py
python person3_stronger_hyde/validate.py
```

All generation stages checkpoint after every query and reuse completed entries. Do not edit the experiment contract in an existing output directory. Model/data downloads and corpus encoding are setup costs, excluded from generation timing.

## Answer pilot and interpretation

Eight question IDs are selected by ascending SHA256 of the ID, before new model outcomes are inspected. This is a workload-limited, fixed exploratory pilot, not a fresh test set. All methods use the same Qwen3-0.6B final-answer model, prompt,128-token generation cap and2048 passage-token context budget (excluding separators). The top5 passages are packed in rank order; the last passage may be truncated. Gold-only context is a separate diagnostic. Every supplied context and generated answer is saved.

`answer_reviews.csv` is an explicit model-assisted qualitative review, not independent legal-expert annotation. Correctness:2=all essential reference propositions answered correctly;1=partially correct/incomplete;0=wrong or no substantive correct answer. Grounding:2=all substantive claims supported by the supplied context;1=mixed support;0=unsupported/contradicted. A grounded abstention may receive grounding2 and correctness0. Citation format/validity is mechanical and does not itself prove factual support. Numerical accuracy is N/A unless a question asks for a numerical result; incidental statute numbers, dates, quantities and case identifiers do not make a question numerical.

## Limits

All100 questions were previously inspected. No held-out, causal, or general superiority claim is justified. Paired McNemar p-values are exploratory and the comparison family is recorded. Generated hypothetical legal statements are unverified retrieval inputs, not legal advice. Full100-query final-answer accuracy has not been measured; the preliminary answer table is only for the eight listed IDs. Generator and embedding runtime versions differ from the historical FLAN run; controls are checked and any rank drift is recorded.

## Sources

- [Official Legal RAG Bench](https://huggingface.co/datasets/isaacus/legal-rag-bench)
- [Official Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Official Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B)
