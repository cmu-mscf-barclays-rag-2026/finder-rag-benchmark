# FiQA: focused GPT-5.4 HyDE transfer

All 648 BEIR FiQA test questions were evaluated against all 57,638 corpus documents, using all 1,706 positive relevance judgments. All 45 aggregate metric rows were independently recomputed from the saved rankings and passed validation.

The legal dataset improvement did not transfer uniformly to finance. Original-question dense retrieval achieved nDCG@10 **0.3687** and Hit@5 **57.25%**. Pure HyDE was lower at **0.3124** and **46.91%**. Combining the question and HyDE vectors brought performance back close to the original question: **0.3668** and **57.41%**. Adding HyDE to the 50/50 hybrid improved over that matched hybrid baseline (**0.3452 → 0.3522** nDCG@10), but remained below original-question dense retrieval. These are descriptive comparisons; the joint paired-test results are in `paired_tests.csv` and the complete `findings.md`.

| Method | Hit@5 | Recall@5 | MRR@5 | nDCG@10 |
| --- | --- | --- | --- | --- |
| question_dense | 0.5725 | 0.3671 | 0.4341 | 0.3687 |
| bm25 | 0.3765 | 0.2255 | 0.2702 | 0.2286 |
| standard_hybrid_d25 | 0.4985 | 0.3038 | 0.3742 | 0.3050 |
| standard_hybrid_d50 | 0.5386 | 0.3306 | 0.4055 | 0.3452 |
| focused160_pure_dense | 0.4691 | 0.2994 | 0.3611 | 0.3124 |
| focused160_mix_dense | 0.5741 | 0.3707 | 0.4274 | 0.3668 |
| focused160_pure_hybrid_d25 | 0.4892 | 0.2962 | 0.3675 | 0.3006 |
| focused160_mix_hybrid_d25 | 0.4985 | 0.3043 | 0.3737 | 0.3068 |
| focused160_mix_hybrid_d50 | 0.5617 | 0.3399 | 0.4183 | 0.3522 |

Hit@5 means at least one relevant document; Recall@5 counts the share of all relevant documents. They differ because FiQA can have multiple relevant documents per question. Financial HyDE generation cost was approximately $1.214825 at the prior run’s conservative token rates. MiniLM, mixing weights and RRF settings were fixed from the earlier experiment; the focused prompt was adapted to finance before outcomes. No train/dev tuning or test-based method selection was performed. Dense encoding uses each document’s first 256 MiniLM tokens, while BM25 indexes full documents. Coincident query/document ID strings are retained, so this is not an exact BEIR leaderboard evaluator reproduction.
