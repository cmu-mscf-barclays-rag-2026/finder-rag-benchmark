# Week 8 — retrieval depth, siblings, MMR, and additional datasets

This folder runs Florence's Meeting 8 experiments on the frozen Legal RAG
Bench split. It uses the Week 6 Qwen 4,096-token embedding window, saved
Top-100 dense rankings, cached normalized vectors, and the same pinned
MiniLM measurement tokenizer. The script fails if the cached vectors do not
reproduce the saved rankings. It performs no Qwen inference.

The Qwen rerun requires the ignored Week 6 embedding and tokenizer caches.
On a fresh clone, recreate the pinned Week 6 environment and rerun its full
experiment with `--device mps --dtype float32` on a compatible Mac to populate
the cache. Week 6 cache keys include device and package versions, so a cache
made under another encoder identity will not match the frozen manifest. The
committed Week 8 result files can be reviewed without those caches.

Read the [Qwen experiment findings](results/findings.md) and the
[FiQA baseline findings](fiqa_results/findings.md) for measured results.

```sh
.venv-week6/bin/python -m week8_retrieval.run --output week8_retrieval/results_new
.venv-week6/bin/python -m week8_retrieval.neighbor_steps --output week8_retrieval/results_new
```

The grid evaluates Top-10/15/20 under 4,000/6,000/8,000-token delivery
budgets. It compares child-only retrieval to directional sibling windows
`0/2`, `1/1`, `1/2`, `2/1`, and `2/2` (before/after). MMR reranks the frozen
Top-100 child candidates using Qwen cosine similarity at λ=1.0, 0.7, and
0.5. λ=1.0 is the frozen relevance-only ranking. Candidate pools, rankings,
and budgets are held fixed across the comparisons. Sibling policies include
only complete chunks that fit; the child-only baseline follows the existing
Week 7 protocol and can end with a partially included chunk. The output
includes per-question evidence IDs, gold-text coverage, returned tokens,
coverage per 1,000 tokens, and mean pairwise word-set Jaccard overlap. Test
comparisons are descriptive; the 20 development questions are too few for
stable tuning among this many configurations.
`neighbor_steps.py` separately compares the first and second preceding and
following siblings under a shared whole-chunk packing rule.

## FiQA Task 2 / BEIR FiQA

[FiQA Task 2](https://sites.google.com/view/fiqa/home) has query-to-answer
relevance labels. The [BEIR FiQA distribution](https://github.com/beir-cellar/beir)
provides a standard corpus, queries, and train/dev/test qrels. Download the
archive into the ignored `.cache` directory and extract it there:

```sh
mkdir -p .cache
curl -L --fail -o .cache/fiqa.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
unzip .cache/fiqa.zip -d .cache
.venv-week6/bin/python -m week8_retrieval.beir_eval --dataset .cache/fiqa --split test --output week8_retrieval/fiqa_results_new
```

`beir_eval.py` accepts any dataset with the same BEIR folder schema and can
evaluate another retriever's JSONL file with `--rankings path`, where each
row is `{"query_id": "...", "doc_ids": ["first", "second"]}`. Without
`--rankings`, it runs the repo's BM25 implementation. It reports ranked and
whole-document delivered Hit@k, Recall@k, all-judged-relevant rate, MRR,
nDCG, and token use at each budget. Relevant FiQA answers can be alternatives;
the all-judged-relevant rate is **not** a validated measure of all evidence
needed to answer a question. The original FiQA site restricts Task 2 data
to noncommercial use. Dataset files remain in the ignored cache.

## LongBench v2

[LongBench v2](https://huggingface.co/datasets/zai-org/LongBench-v2) supplies
long contexts, questions, four choices, and answer letters. Its public
schema does not supply passage IDs or evidence spans, so it cannot support
retrieval Hit/Recall/nDCG without new annotations. It can support answer
accuracy after a retrieval-and-answer pipeline. Download `data.json` from
the dataset page, then prepare query-scoped chunks:

```sh
.venv-week6/bin/python -m week8_retrieval.longbench_v2 prepare --input .cache/longbench_v2/data.json --output .cache/longbench_v2/prepared
.venv-week6/bin/python -m week8_retrieval.longbench_v2 score --prepared .cache/longbench_v2/prepared --predictions predictions.jsonl
```

The prepared `queries.jsonl` contains questions and choices; `answers.jsonl`
holds the answer keys separately. `chunks.jsonl` contains text chunks with
`query_id` and `chunk_id`. Retrieval must search only chunks with the matching
`query_id` to avoid mixing unrelated contexts. Prediction rows use
`{"query_id": "...", "answer": "A"}`.
The adapter is tested on a small synthetic example; the full 465 MB dataset
was not downloaded for this retrieval experiment because it lacks retrieval
labels.

## Reproduce checks

```sh
.venv-week6/bin/python -m unittest discover -s week8_retrieval/tests -v
```
