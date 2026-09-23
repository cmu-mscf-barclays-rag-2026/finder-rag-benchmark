# Legal RAG Bench Part C Findings

## Main result

At Top 5, the fixed Hybrid RRF configuration transferred from FinDER retrieved
the labelled passage for 36 of 100 questions. BM25 retrieved
31, while dense MiniLM retrieved 29.

| Method | Recall and Hit Rate at 5 | MRR at 5 | nDCG at 5 |
|---|---:|---:|---:|
| BM25 | 0.31 | 0.2295 | 0.2500 |
| Dense MiniLM | 0.29 | 0.1610 | 0.1933 |
| Hybrid RRF alpha 0.25 | 0.36 | 0.2490 | 0.2764 |

The fixed Hybrid improved Hit Rate at 5 by 5 percentage points over BM25 and by
7 points over dense MiniLM. Relative to BM25, Hybrid added six unique successes
and lost one BM25 success. The exact paired test is not statistically significant
at the 5% level, so this should be presented as promising directional evidence,
not a definitive win.

## Weight sensitivity

| BM25 / dense weight | Recall and Hit Rate at 5 | MRR at 5 | nDCG at 5 |
|---|---:|---:|---:|
| 75% / 25% | 0.36 | 0.2490 | 0.2764 |
| 50% / 50% | 0.37 | 0.2412 | 0.2727 |
| 25% / 75% | 0.30 | 0.1898 | 0.2174 |

The 50/50 weighting finds one additional labelled passage in the Top 5, while
the BM25-heavy 75/25 weighting ranks relevant passages slightly earlier and has
the strongest MRR and nDCG. Giving dense retrieval 75% of the weight reduces all
Top-5 metrics. These full-test comparisons are exploratory; the unbiased result
remains the transferred 75/25 configuration and the nested five-fold estimate.

## Interpretation

Legal RAG Bench deliberately uses questions with low lexical overlap. Dense
retrieval therefore contributes complementary candidates, while BM25 remains
useful for legal terms and named concepts. The result supports using a
BM25-heavy Hybrid rather than replacing BM25 with a general-purpose embedding
model.

## Dataset limitation

The current benchmark labels one most-relevant passage per question and has only
100 questions. It does not directly measure multi-evidence completeness, and
Recall at K equals Hit Rate at K in this experiment.
