# Cheryl — question-only HyDE robustness (week of 6 October 2026)

## Protocol and limits

100 Legal RAG Bench questions; 4,876 passages; fixed MiniLM and FLAN-T5-base,
pinned model revisions. Generation sees only the original question. Eight prompt ×
length × decoding configurations plus two repeated stochastic seeds. Four dense
representations per configuration, plus BM25 fusion weights/depths. This yields
284 method/configuration rows at each K. The same 100 questions were inspected
previously: **all sweep results are exploratory, not new held-out test results**.
No Qwen or teammate dependency. Final answer correctness was not evaluated here.

## Recomputed controls

| Method | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| question_bm25 | 0.31 | 0.2295 | 0.2500 |
| question_dense | 0.29 | 0.1610 | 0.1933 |
| standard_hybrid | 0.36 | 0.2490 | 0.2764 |
| reference_answer_oracle | 0.72 | 0.5498 | 0.5929 |

## Formulation comparison

Read `generation_comparison_k5.csv` for one row per generation setup and columns for
single-HyDE, question+HyDE, average-embedding multi-HyDE, and multi-ranking RRF.
Read `fixed_weight_comparison_k5.csv` for their BM25-hybrid variants at dense weight
0.25 and candidate depth 100, the same control setting as last week.

The strongest **observed** dense representation was `short_answer_len32_t0_seed20261006__question_plus_hyde`:
Hit@5 0.29, versus question dense 0.29. Its descriptive
oracle-gap fraction is 0.0% relative to the oracle
0.72. This is a sweep maximum, not an independently validated winner.

The strongest **observed fixed-weight** HyDE hybrid was `short_answer_len32_t0_seed20261006__multi_rrf__hybrid_a0.25_d100`:
Hit@5 0.39, versus standard hybrid 0.36. Paired gains/losses:
7 gained and 4 lost;
unadjusted exact McNemar p=0.5488, Holm-adjusted p=1
across all 280 exploratory comparisons. Do not claim significance from a
selected sweep maximum. Full parameter sweeps are in `summary_k5.csv`.

The higher Hit@5 is also a ranking tradeoff: the selected hybrid's MRR@5 is
0.2260, versus 0.2490 for standard hybrid; nDCG@5 is
0.2666, versus 0.2764. More Top-5 hits therefore
does not establish better ranking quality. Read `presentation_k5.csv` for a
compact descriptive table; maxima are explicitly labelled exploratory.

## Stability

representation,hit5_min,hit5_mean,hit5_max,n_settings
hyde_single,0.04,0.088,0.19,10
multi_average,0.07,0.138,0.21,10
multi_rrf,0.06,0.123,0.2,10
question_plus_hyde,0.24,0.2679999999999999,0.29,10


Seed checks for sampled evidence-passage generation (length cap 96):

representation,hit5_mean,hit5_min,hit5_max,n_seeds
hyde_single,0.09333333333333334,0.08,0.11,3
hyde_single_hybrid,0.27666666666666667,0.26,0.3,3
multi_average,0.19333333333333336,0.17,0.21,3
multi_average_hybrid,0.3333333333333333,0.33,0.34,3
multi_rrf,0.17666666666666667,0.15,0.2,3
multi_rrf_hybrid,0.33666666666666667,0.33,0.35,3
question_plus_hyde,0.24666666666666667,0.24,0.26,3
question_plus_hyde_hybrid,0.3333333333333333,0.33,0.34,3


These ranges cover this small local generator and this benchmark only. They do
not establish that HyDE broadly succeeds or fails with stronger generators.

## Timing and audits

`generation_audit_summary.csv` records empty outputs and generated-text truncation
under MiniLM's original token limit. `generation_configs.csv` records three-hypothesis
generation throughput. Timing columns are batched CPU estimates; single-generation
cost is approximated by dividing total three-hypothesis time by three and extra
question+HyDE encoding is not timed. They are not interactive latency measurements.

## Easy meeting notes

- I kept MiniLM fixed and tested question-only HyDE generation, so the embedding
  model stayed the same across all comparisons.
- I compared short answers, evidence-style passages, the question plus generated
  text, and three hypotheses combined in two ways.
- I also changed output length, decoding, fusion weights and candidate depth,
  and repeated one sampled setting with three seeds.
- Question dense retrieved the correct passage for 29 out of 100 questions;
  the reference-answer oracle retrieved it for 72. The oracle uses the
  dataset answer and is only a diagnostic.
- The best observed dense setup reached 29 hits, and the
  best observed hybrid at the original weight reached 39,
  compared with 36 for standard hybrid.
- These are exploratory results. I will show the full ranges and seed checks,
  rather than treating the highest score as a reliable final improvement.
- The next step is to freeze a configuration and validate it on new questions,
  then test a stronger generator if the local model remains the bottleneck.

## Files to open during the meeting

1. `generation_comparison_k5.csv` — prompt/length/decoding and formulation comparison.
2. `fixed_weight_comparison_k5.csv` — fair comparison at unchanged hybrid weight.
3. `seed_robustness_k5.csv` — consistency across random seeds.
4. `help_hurt_examples.csv` and `generated_*.csv` — concrete question examples.
5. `paired_comparisons_k5.csv` — paired tests and multiplicity adjustment.

Run `python person3_hyde/validate_robustness.py` to independently verify CSV metrics
with no model download. Model-free replay is documented in `ROBUSTNESS_README.md`.
