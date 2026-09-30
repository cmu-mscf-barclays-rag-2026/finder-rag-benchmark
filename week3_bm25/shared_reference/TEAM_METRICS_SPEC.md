# Team metrics contract

The final comparison is valid only when every row has the same `dataset_id`,
`corpus_id`, and `split_id`. The aggregation script stops with an error when
these values differ.

Required fields:

| Field | Meaning |
|---|---|
| `schema_version` | Metrics file format version |
| `owner` | Team member or workstream |
| `method_id` | Short stable method name |
| `method` | Human-readable method label |
| `dataset_id` | FinDER dataset version |
| `corpus_id` | Exact retrieval corpus construction |
| `split_id` | Exact development/test split rule |
| `split` | Must be `test` for the final comparison |
| `k` | Number of retrieved passages evaluated |
| `n_queries` | Number of held-out questions |
| `precision_at_k` | Relevant passages in Top-K divided by K |
| `recall_at_k` | Fraction of gold passages retrieved in Top-K |
| `hit_rate_at_k` | Fraction of questions with at least one hit in Top-K |
| `mrr_at_k` | Mean reciprocal rank of the first hit within K |
| `ndcg_at_k` | Ranking quality within K |
| `latency_ms_per_query` | Retrieval time only; indexing excluded |
| `answer_n` | Number of questions used for answer evaluation |
| `answer_exact_match` | Normalized exact-answer match rate |
| `answer_token_f1` | Token overlap with the gold answer |
| `answer_semantic_similarity` | Semantic similarity to the gold answer |
| `generator_model` | Shared generator used for answer evaluation |
| `answer_status` | `not_run`, `pilot`, or `final` |
| `answer_protocol_id` | Shared sample/prompt/generator configuration ID |

Rules for the final presentation:

1. Compare retrieval methods on `split=test`, normally at K=5.
2. Do not combine scores produced from different corpora or splits.
3. Use one shared generator, prompt, Top-K context, token limit, and question
   sample for answer correctness.
4. Mark preliminary local answer results as `pilot`. The aggregator will not
   put them into `answer_comparison.csv`.
5. Report latency hardware separately because timing is machine-dependent.
6. Keep the FinDER paper baseline in a separate table because its RAGAS Context
   Recall is not the same metric as exact Recall@K here.
