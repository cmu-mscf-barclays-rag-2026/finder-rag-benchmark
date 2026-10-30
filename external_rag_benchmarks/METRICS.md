# Reading the results

## FiQA

Each query has judged relevant document IDs. Binary Hit@k asks whether any was found. Recall@k is the fraction of judged relevant documents found. Precision@k divides relevant results by k. MRR@k rewards the first relevant rank. nDCG@k discounts relevance by rank and normalizes against the ideal ranking; gain is 2^relevance minus 1. Unjudged results receive zero relevance under these benchmark labels.

`all_judged_relevant_retrieved` is a diagnostic of retrieving every positive relevance label. It is not proof of complete required evidence: multiple answers may be alternatives and the judgments may not be exhaustive. These metrics use document IDs and do not enforce a context budget or evaluate final answers.

The fusion weight is selected only on the 50 development questions by nDCG@10, then Recall@10, then proximity to equal weight. Equal fusion remains a separately reported baseline. A selected weight of 0 is dense-only and a weight of 1 is sparse-only; either outcome is reported without claiming a distinct hybrid improvement.

## LongBench v2 pilot

The revised experiment fixes retrieval depth at forty and changes only the delivered-body budget: 2,048, 8,192, or all top-40 chunks. There is no additional cutoff in the last condition, and all prompts must fit the model's actual native-token capacity with generation allowance. This condition does not supply the full original source.

The same questions were inspected in the earlier top-10 pilot. Budget comparisons are therefore exploratory follow-ups, with all settings reported rather than a claim of new held-out validation. The original 2,048-token contexts are checked against the follow-up contexts; increasing retrieval depth does not change that prefix when the original top ten already filled the small budget.

Accuracy is the fraction of generated A/B/C/D responses matching the published correct letter. Invalid responses count as incorrect and a valid-response rate is shown. The same local reader and decoding settings are used for all retrieval methods. The no-context condition indicates how often that reader can answer without retrieved source text.

There are no official passage relevance labels or source-span labels here. Do not calculate retrieval recall or evidence completeness by matching the answer letter in context. A correct answer may reflect prior knowledge or guessing. An incorrect answer may arise from retrieval or reader errors; answer accuracy alone cannot distinguish those causes.

Every query searches its own supplied context. Single- and multi-document labels describe the official task category; this pilot's results do not establish that multiple required evidence spans were retrieved. Only the short QA categories are sampled. This adaptation cannot be compared directly with the official whole-benchmark leaderboard.

The delivered-context budget counts body text using a pinned WordPiece tokenizer. Reader prompt tokens are separately measured with the reader tokenizer and checked against the registered limit. Embedding window, retrieval depth, delivered-body budget and reader input size are different quantities.

`paired_changes.csv` compares each larger budget with 2,048 on the same questions and retriever: wins change an incorrect answer to correct, losses change correct to incorrect, and ties preserve correctness. These counts show mixed effects that a single average can hide. No significance claim follows from the fourteen-question sample.

## Runtime

FiQA corpus encoding and sparse-index construction are recorded separately. Per-query timestamps describe one observed pass, including query encoding and CPU search for dense retrieval. LongBench generation time excludes retrieval. These are descriptive component costs on one laptop, not a repeated end-to-end latency benchmark. Download and model-load times are excluded from encoding/generation totals.

## Relation to earlier work

The earlier Legal RAG pilot measures inclusion of assistant-proposed supporting spans. This folder uses the native labels available in each new dataset. Scores across these three evaluations have different meanings and should not be pooled into one accuracy column.
