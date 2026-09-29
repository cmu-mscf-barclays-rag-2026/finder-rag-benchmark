# Person 3 — Reference-answer and HyDE retrieval

## Main Top-5 results

| Method | Hit/Recall@5 | MRR@5 | nDCG@5 | End-to-end ms/query |
|---|---:|---:|---:|---:|
| Question BM25 | 0.31 | 0.2295 | 0.2500 | 0.23 |
| Question Dense MiniLM | 0.29 | 0.1610 | 0.1933 | 6.15 |
| Question BM25 + Question Dense RRF | 0.36 | 0.2490 | 0.2764 | 6.43 |
| Reference Answer Dense (oracle) | 0.72 | 0.5498 | 0.5929 | 5.64 |
| Question-only HyDE Dense | 0.15 | 0.0883 | 0.1033 | 243.76 |
| Question BM25 + HyDE Dense RRF | 0.37 | 0.2460 | 0.2765 | 244.04 |

## Interpretation

- Using the expert-written reference answer changes Hit/Recall@5 by
  +0.43 versus the original question. This is an **oracle diagnostic**, not
  a deployable method, because the answer is unavailable when a user asks a question.
- Question-only HyDE changes Hit/Recall@5 by -0.14 versus question-dense.
- Replacing question-dense with HyDE-dense inside the same BM25-heavy hybrid changes
  Hit/Recall@5 by +0.01. Both hybrids are deployable and use only the
  question, so this is the fairest test of HyDE's incremental value.
- With only 100 questions, small differences should be described as directional.
  See `paired_comparisons_k5.csv` for exact paired p-values.

## Generation sensitivity

An exploratory generation check found that HyDE results changed materially with the
prompt and decoding configuration. The longer prompt produced HyDE-dense Hit@5 of
0.27 and HyDE-hybrid Hit@5 of 0.32; the concise no-repeat configuration produced
0.15 and 0.37. This instability is itself an important finding: the current local
generator does not provide a robust standalone retrieval improvement. The saved
primary result uses the concise configuration, and `generation_sensitivity_k5.csv`
records both exploratory runs.

## What each comparison tests

1. **Question Dense:** the unchanged dense baseline.
2. **Reference Answer Dense:** whether answer-like wording closes the query-document
   vocabulary gap; it provides an approximate upper-bound diagnostic.
3. **Question-only HyDE Dense:** whether a locally generated hypothetical legal answer
   improves dense retrieval without seeing the gold answer.
4. **Standard Hybrid:** the existing BM25 + question-dense control under the same RRF
   weight.
5. **Question BM25 + HyDE Dense RRF:** whether exact terms from the original question
   and semantic terms from HyDE complement each other.

The generator is `google/flan-t5-base` with deterministic greedy decoding. A compact
local generator keeps the experiment reproducible on a laptop, but its legal knowledge
is limited. Poorly formed or overly generic hypothetical answers can reduce retrieval
quality; `hyde_help_hurt_examples.csv` makes those cases auditable.
