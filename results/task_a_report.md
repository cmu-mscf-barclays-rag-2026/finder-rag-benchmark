# Task A - Dense retrieval with held-out evaluation

## Protocol

The final experiment uses all 5,703 FinDER questions and the deduplicated union
of their 5,830 expert reference passages. Answers and reasoning are not indexed.
The deterministic `sha1-mod5-dev-v1` split assigns 1,128 questions to
development and holds out 4,575 questions for the final test.

Each source passage is divided into chunks for MiniLM embedding. At query time,
chunk cosine scores are max-pooled by source passage, and distinct source
passages are ranked. This is important: the gold labels and Precision/Recall
denominators remain identical across chunking configurations.

The predeclared selection rule is maximum development F1@5, breaking ties by
higher Recall@5 and then fewer chunks. The test split is not used for selection.

## Development parameter sweep at k=5

| Chunk size | Overlap | Index chunks | Precision@5 | Recall@5 | Hit@5 | F1@5 |
|---:|---:|---:|---:|---:|---:|---:|
| **500** | **75** | 44,679 | **0.04965** | **0.23478** | **0.24291** | **0.08196** |
| 1,000 | 150 | 22,153 | 0.04752 | 0.22636 | 0.23493 | 0.07855 |
| 1,500 | 225 | 15,348 | 0.04734 | 0.22326 | 0.23138 | 0.07812 |
| 1,000 | 0 | 21,470 | 0.04610 | 0.21912 | 0.22695 | 0.07617 |
| 1,000 | 300 | 24,634 | 0.04787 | 0.22843 | 0.23582 | 0.07916 |

The 500/75 configuration is selected. Smaller chunks provide a more focused
semantic match and improve development F1@5, but create about twice as many
vectors as the 1,000/150 baseline and almost three times as many as 1,500/225.

At a fixed 1,000-character size, overlap 300 performs modestly better than 0 or
150 on development retrieval, but costs 14.7% more chunks than zero overlap.
The differences are small enough that overlap should not be treated as the main
source of improvement.

## Final held-out test results

Only the selected 500/75 configuration is evaluated on test.

| k | Precision@k | Recall@k | Hit Rate@k | F1@k |
|---:|---:|---:|---:|---:|
| 1 | **0.13027** | 0.12596 | 0.13027 | **0.12808** |
| 3 | 0.06761 | 0.19392 | 0.19956 | 0.10027 |
| 5 | 0.04791 | 0.22840 | 0.23410 | 0.07921 |
| 10 | 0.02877 | **0.27333** | **0.27934** | 0.05205 |

Increasing k gives the expected precision-recall tradeoff. From k=1 to k=10,
Recall increases by 14.74 percentage points and Hit Rate by 14.91 points, while
Precision drops by 10.15 points. Use k=5 for the team's standardized comparison;
k=10 is preferable only when retrieval coverage matters more than context noise
and prompt length.

The development-to-test change at k=5 is small: Precision falls from 0.04965 to
0.04791 (0.17 percentage points), and Recall falls from 0.23478 to 0.22840
(0.64 points). This supports—but does not prove—that the selected parameters
generalize beyond the development questions.

## Final Task A row

| Method | Precision@5 | Recall@5 |
|---|---:|---:|
| Dense MiniLM, chunk 500, overlap 75 | **0.04791** | **0.22840** |

## Timing and limitations

The committed CUDA run measured 0.663 ms per held-out query for query embedding
plus retrieval, excluding index construction. This figure is hardware-specific
and should not be compared with timings from a different machine or device.

This benchmark retrieves from FinDER's annotated reference-passage corpus, not
raw full 10-K filings. It measures evidence retrieval, not answer correctness.
The passage-level relevance labels are exact, but a relevant long passage can
still contain much material unrelated to the particular question. Additional
random seeds are unnecessary for the hash split, but model changes should be
selected on the same development IDs and evaluated only once on test.
