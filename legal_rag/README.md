# Florence — Legal RAG Bench: BM25, dense, and hybrid

This experiment is separate from the FinDER benchmark. It indexes the complete
[Isaacus Legal RAG Bench](https://huggingface.co/datasets/isaacus/legal-rag-bench)
corpus (4,876 passages), not just the 95 passage IDs labeled by its 100 questions.
The pinned dataset revision and file hashes are checked before any experiment.

Initial completed outputs: [results/initial/report.md](results/initial/report.md).

## Reproduce

From the repository root, in a Python 3.12 environment:

```sh
python -m pip install -r legal_rag/requirements-lock.txt
python legal_rag/download.py --output legal_rag/data
python -m unittest legal_rag.test_run -v
python legal_rag/run.py --data legal_rag/data --output legal_rag/results/run1
```

The first run downloads the pinned MiniLM model. Use `--model-cache PATH` to choose
its cache and `--offline` once downloaded. Optional `--embedding-cache PATH`
caches corpus vectors using a hash of model settings, library versions, and text. Output must be new/empty. The run uses
CPU with four Torch and BLAS threads. Dataset/model downloads and index building
are outside timed query retrieval. Do not commit model caches or raw data.

## Protocol

- Corpus text only: no reference answers in indexing, retrieval, or generation
  prompts. Titles and footnotes are not separately appended. Preserve source IDs.
- Internal split: group questions by SHA-256 of their gold passage's exact text;
  sort groups, shuffle with seed 42, allocate complete groups until at least 20%
  of questions are in development. Remaining questions form held-out evaluation.
  The dataset officially supplies only a test split: this internal split is not
  an official benchmark split. All corpus passages remain searchable in both.
- Chunk sizes: original supplied passage, 128, and 256 MiniLM tokens INCLUDING
  special tokens. Child chunks overlap by 20% of their content-token budget.
  Keep exact parent character offsets. Retokenization is checked before encoding.
- Original passages may exceed MiniLM's 256-token limit. Record truncation counts
  explicitly; original-mode BM25 can see more text than its dense counterpart.
  Child-chunk modes avoid this asymmetry. This initial dense model is a continuity
  baseline, not a claim about the best legal embedding model.
- BM25: reuse Florence's existing scorer, `k1` in {0.8,1.2,1.6}, `b` in
  {0.25,0.75,1.0}. Dense: normalized MiniLM embeddings, exact cosine search.
- Weighted reciprocal rank fusion: `w/(60+BM25_rank) + (1-w)/(60+dense_rank)`;
  absent candidates contribute zero. Each retriever supplies up to 100 chunks.
  Try `w` in {0.25,0.5,0.75} using the best development BM25 per chunk size.
  This is a staged search, not an exhaustive hybrid parameter grid.
- Select one configuration per method by development nDCG@5, then MRR@5, then
  Recall@5, with grid order resolving exact ties. Freeze before test retrieval.
- Report K={1,3,5,10,20} from the selected configurations. K counts actual chunks,
  not distinct source passages. Multiple chunks from one source consume slots.
  Different methods may select different chunk sizes; inspect context lengths.
- Warm latency: three warmups per method, three repeats of all held-out queries
  with shuffled/interleaved methods. Include online query embeddings and fusion;
  exclude embedding/index construction, expansion of result text, and evaluation.
  These are top-100 candidate-search latencies shared across the reported cutoffs,
  not independently timed per-K runs. No generation latency is reported.

## Metric meanings and limitations

Every question has ONE gold passage. Recall@K, Hit Rate@K, and All Evidence Hit@K
therefore coincide when credit is based on parent-ID matching. Retrieving even a
small child of the gold passage earns a parent hit; that is not proof of answer
sufficiency. MRR@K and nDCG@K award credit at the first matching chunk rank only;
repeated children cannot earn additional relevance credit. Misses score zero.

`evidence_coverage` means **gold-passage character coverage**: union length of
retrieved child spans belonging to the gold passage divided by its text length.
Overlap counts once. With whole passages it also equals Hit Rate. This is a text
coverage proxy, NOT semantic evidence completeness or a measure of all legal
conditions in the answer. True multi-evidence evaluation needs new annotations.

The corpus contains 4,658 unique exact texts across 4,876 IDs. Primary scoring
preserves all official IDs. A different ID with identical text can still be a
strict-ID miss; review such cases rather than silently changing the labels.
Only 100 questions are available, so these results are preliminary: avoid treating
small differences as a robust ranking or claiming general legal-domain accuracy.

## Answer correctness: pending model selection

The dataset supplies reference answers. The [benchmark paper](https://arxiv.org/html/2603.01710v1)
evaluates correctness and groundedness separately. Retrieval metrics do not replace
these measures. No generator or judge is run by the retrieval script.

`generation_inputs.jsonl` contains question/context prompts for each selected
method at K=5 and an oracle using the gold passage. Reference answers are absent.
`answer_review.csv` contains the reference answers and blank evaluation fields.
Use the SAME generator/version, decoding settings, instructions, and a predeclared
context budget across methods; regenerate contexts if the chosen budget clips them.
MiniLM token counts are diagnostic, not the eventual generator's token counts.

Review rubric:

1. Correctness (0/1): the answer entails the required reference conclusions,
   includes material conditions/exceptions, and does not materially contradict them.
   An abstention on these answerable questions is not a correct answer.
2. Groundedness (0/1): substantive factual claims are supported by the supplied
   retrieved context. A correct answer from model memory can still be ungrounded.
3. Save rationale, reviewer identity/model version, and uncertain cases. Calibrate
   on development examples; audit a sample and disagreements manually. Do not use
   the test judgments to retune retrieval and still call the result held-out.
4. Report correctness, groundedness, and BOTH together. An abstention may avoid
   unsupported claims; report abstentions separately rather than praising that as
   answer quality. Do not equate lexical exact match with long-form correctness.

## Outputs

- `report.md`, `test_metrics.csv`: selected held-out results and caveats.
- `dev_sweep.csv`, `selected.json`: complete development search and frozen choices.
- `manifest.json`, `split.json`: pinned inputs, environment, truncation, query IDs.
- `retrievals.jsonl`, `per_query.csv`: inspectable ranks, text spans, metrics.
- `latency_raw.csv`: every timed measurement.
- `generation_inputs.jsonl`, `answer_review.csv`: prepared, NOT evaluated answers.

The raw data remains under its upstream license; see the dataset card. This code
preserves attribution and does not replace or reinterpret the upstream license.

After all prepared answer rows are generated and reviewed, fill `reviewer` with
the reviewer identity or judge model/version, `abstained` with 0/1, and run:

```sh
python legal_rag/evaluate_answers.py \
  --reviews legal_rag/results/run1/answer_review.csv \
  --inputs legal_rag/results/run1/generation_inputs.jsonl \
  --output legal_rag/results/run1/answer_metrics.csv
```

Missing judgments, duplicate rows, mismatched question sets, and unreviewed answers
are rejected. Save the generator model/version, prompt, decoding settings, judge
rubric/version, and actual context budget with that later answer experiment.

Independently verify a completed run:

```sh
python legal_rag/validate_results.py --data legal_rag/data --results legal_rag/results/initial
```

`requirements-lock.txt` records the execution environment; `requirements.txt` lists
the main dependencies for a fresh compatible environment. No API key is required
for the retrieval experiment. Initial completed results use the CPU only.

For presentation examples and paired Hit@5 differences:

```sh
python legal_rag/analyze_results.py --results legal_rag/results/initial
```

`question_comparison.csv` flags hybrid rescues, regressions, and shared misses.
`paired_hit5_intervals.csv` gives exploratory percentile intervals from 10,000
paired gold-passage-group bootstrap resamples (seed 42). They quantify sampling uncertainty
for this internal split; they do not include model-selection uncertainty or justify
claims about other legal datasets.

## Top-100 candidate-recall diagnostic

[Report](results/top100/report.md) and [per-question classifications](results/top100/per_query.csv).
This reuses the frozen initial configurations and the same 80 held-out questions;
it does not retune parameters. BM25 finds the labeled parent within Top-100 for
53/80 questions; dense and hybrid each find it for 61/80. Hybrid's pre-fusion union
contains it for 65/80; truncating the fused ranking to 100 drops four of those hits.

```sh
python legal_rag/candidate_recall.py \
  --data legal_rag/data \
  --baseline legal_rag/results/initial \
  --output legal_rag/results/top100_repeat
```

Optional `--model-cache PATH --embedding-cache PATH --offline` reuses local caches.
Use the original locked environment. The script verifies that each reproduced
Top-20 ranking matches the baseline before reporting deeper recall. The baseline's
locally saved `retrievals.jsonl` is required; rerun the initial experiment if absent.
`rankings.jsonl` saves Top-100 IDs, scores, and hybrid component lists.

A miss at Top-5 but a hit at Top-100 is potentially recoverable by reranking.
Absence from Top-100 means a reranker limited to that list cannot recover the gold
parent; it does not mean the passage is absent from the corpus or never retrievable.
K counts chunks, and parent-ID recall does not measure answer correctness.
