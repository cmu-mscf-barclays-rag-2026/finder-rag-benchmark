# Keyword retrieval versus dense retrieval

## What Person 2 should explain

BM25 rewards exact query-term matches, weighting rare terms more strongly and
normalizing document length. It does not understand that an unfamiliar ticker,
abbreviation or paraphrase refers to another expression. A dense retriever compares
learned vector representations and can match related expressions without literal
overlap, depending on its model and input truncation.

These are **mechanistic hypotheses**, not measured dense-vs-BM25 results for this
project. The actual BM25 measurements are in [RESULTS.md](RESULTS.md). The paired
dense comparison must use Person 1's rankings on the shared prepared bundle.

| Query characteristic | BM25 hypothesis | What to inspect in dense output |
| --- | --- | --- |
| Exact company name, rare technical phrase, specific year | Literal matches can rank highly. | Does semantic similarity preserve the requested entity and year? |
| Ticker in query, only full company name in evidence | Missed company term can weaken ranking. | Can the embedding model link the ticker to the right company? |
| Abbreviations such as “rev”, “cap alloc”, “mgmt” | Token mismatch can lose useful evidence. | Does the model recognize the financial abbreviation in context? |
| “Revenue growth” versus “sales increased” | Synonyms do not match unless other terms overlap. | Are paraphrases matched without drifting to another company? |
| Repeated boilerplate about risk or governance | Common disclosure words can retrieve the wrong filing. | Do context and entity clues improve ordering? |
| Long tables and exact numeric amounts | Term overlap can help, but BM25 cannot perform arithmetic. | Check number/entity errors and model token-limit truncation. |
| Multiple supporting references | Finding one passage improves MRR but may leave recall low. | Does the method retrieve and rank every needed passage? |

Neither BM25 nor dense retrieval alone calculates an answer. A relevant passage at
rank 1 improves MRR but says nothing about whether a generator subsequently computes
the correct result. That distinction is why the team also needs answer correctness.

## Observed BM25 examples from the real test run

Default `k1=1.2`, `b=0.75`, whole references, fixed 4,563-query local test partition:

| Query ID | Query topic | First gold rank within top 10 | Observation |
| --- | --- | --- | --- |
| `b703f322` | CrowdStrike revenue growth across FY22–FY24 | 1 | The gold financial table contains the full company name and revenue. Several abbreviated query tokens are absent, yet the gold passage ranks first. |
| `b33fcee7` | CBOE Data & Access Solutions revenue change | 3 | The matching income table is retrieved, but an executive-officer passage ranks first. Recall alone hides this ordering problem; RR@5 is 1/3 and nDCG@5 is 0.5. |
| `8ac98b17` | CRWD liquidity at January 31, 2024 | 9 | Gold evidence is outside top 5. The top result describes KeyCorp liquidity. The gold passage lacks `crwd` and uses `January` instead of `Jan`; these mismatches are candidates for a dense-retrieval comparison. |

These observations come from saved `results/bm25/k5/run.json`,
`results/bm25/k10/per_query.csv`, and the shared reference text. They do not establish
that dense retrieval fixes the same examples; test that with Person 1's run.

At k=5, Legal has the highest category nDCG (0.3003), while Shareholder return has
the lowest (0.1520). The full category table is in `results/bm25/k5/by_category.csv`.
This describes this local query split, not a universal property of these topics.

## Analysis workflow once Person 1 shares dense results

```bash
python -m finder_bm25 compare --runs results/bm25/k5/run.json results/dense/k5/run.json --output results/comparison
```

1. Read `comparison.md`: aggregate Precision/Recall/MRR/nDCG and per-query win/loss/tie counts.
2. Read `query_deltas_1.csv`: positive deltas mean the second method outperforms BM25.
3. Inspect at least five examples in each direction, using `run.json`,
   `corpus.jsonl` and `qrels.json` to see both retrieved and missed gold evidence.
4. Label observed differences: exact entity/year, acronym, paraphrase, boilerplate,
   partial evidence, table/numeric or incomplete ground-truth judgment.
5. Compare categories and reasoning flags using `by_category.csv` and `per_query.csv`.
6. Add the same generator and Person 3's answer evaluator to both methods before
   asserting that a retrieval improvement improved final answers.

Do not generalize from a few hand-picked examples; quote the paired aggregate
results and use examples to explain them. A per-query difference is not itself a
statistical significance test. Record failed hypotheses as well as successful ones.

## Sponsor-facing summary template

> We evaluated BM25 and [dense model] on [N] identical FinDER queries using a shared
> corpus and relevance judgments. At k=5, BM25 achieved MRR [x] and nDCG [y], while
> dense achieved [a] and [b]. We observed [measured pattern] in [count/category].
> These results use the pooled annotated-reference corpus. Answer-quality scores
> are [measured with the shared generator/evaluator / still pending].

The FinDER paper is useful background, but its full-filing retrieval and RAGAS
evaluation use a different protocol. Its reported results cannot fill this project's
dense row. [FinDER paper](https://arxiv.org/abs/2504.15800).
