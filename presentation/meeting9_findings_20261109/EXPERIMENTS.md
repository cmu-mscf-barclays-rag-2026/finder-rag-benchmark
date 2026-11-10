# Research questions and outcomes

9 November 2026. Earlier FiQA 628 results motivated this work; they were not regenerated. All significance claims below are exploratory, with separate correction families as recorded in the protocols.

| # | Question | Experiment | Outcome |
|---|---|---|---|
| 1 | Does prompt length or constraint wording explain the benefit? | Legal 20+FiQA 20; 2×2 length/instruction design; 160 generations | None of 12 planned comparisons passes Holm 12; causal explanation remains unresolved. |
| 2 | Does HyDE transfer to a new dataset? | All 309 viDoRe Finance English queries against 2, 942 pages; Focused/Short prompts frozen | On 249 queries, both 50/50 mixes beat first 256-token Question dense within Holm 4; Short does not establish superiority over Focused. |
| 3 | Is page truncation a bottleneck? | First 256-token page vectors vs 14, 017 full-OCR chunks with page max aggregation | All 309 Question dense 0.3085→0.4385; Focused mix 0.3432→0.4563. Coverage and aggregation change together; descriptive only. |
| 4 | Does simple dense+BM25 fusion beat BM25? | Four unfiltered candidates; select on development 60, then retain that choice | Selected Question Hybrid .50 essentially matches BM25 on 249 (0.513354 vs 0.513310), with slightly lower Hit/Recall@10. No reselection. |
| 5 | Does a question-only bank constraint help? | Fixed alias/ambiguity rules, same mask for all methods |249-query BM25 0.5133→0.5691; Focused dense mix 0.4330→0.5184.13 queries lose some positive candidates; complete qrels denominators stay unchanged. |
| 6 | Does constrained HyDE Hybrid add value? | Three dense queries × two RRF weights, six fixed settings | Focused Hybrid .50 has the highest 249-query nDCG@10 among the six: 0.6100. |
| 7 | Is its gain entirely due to filtering? | Filter off/on × Question/Focused, RRF weight.50 | HyDE increment+0.0316 without filter,+0.0360 with filter. Interaction descriptive only. |
| 8 | Is the hybrid gain robust? | Frozen 10-test family, 100 k paired sign flips, Holm 10; leave-one-report-out sensitivity | Focused .50 vs BM25 p=.00328; vs Question .50 p=.00045. All six report omissions keep positive effects; shared-source dependence remains unadjusted. |
| 9 | Does the finding extend beyond Top 5? | Save Hit/Recall/MRR/nDCG at 5/10/15/20/25 | Focused Hybrid has higher descriptive Hit at all five depths than BM25 and Question Hybrid .50. No new per-depth tests. |
| 10 | Is relevant-document delivery enough? | Legal 20 fact/quote drafts; 19 eligible; fixed context budgets | AtTop 10/1024 WordPiece, draft complete-quote availability 21.1% Question, 42.1% Focused, 36.8% Short. This is not validated semantic sufficiency. |
| 11 | Do retrieval gains improve final answers? | Fixed 20-query reader, four conditions; 80 logical answers/66 calls; joint then isolated judges | Reader execution complete. Both raters rejected after predefined validity failures; no answer-accuracy conclusion. Human review pending. |
| 12 | Can HNSW reduce search cost? | FiQA 628 cached vectors; exact plus six HNSW settings, two query types | ExampleM32/ef64: 12.2× lower warm search p 50, 99.32% exact-neighbor Recall@10; small qrel score loss, larger index. |
| 13 | What is the quality/cost/latency tradeoff? | Retained API usage plus matched local CPU profiling and independent reductions | Focused generation~$0.001924/query; reader 20 fresh generation+reader~$0.007958/query. Generation is seconds, retrieval milliseconds; stage sums are modeled, not E2E measurements. |

Raw aggregate tables are in[data](data). Exact settings, source revisions and audit records are in[protocols_and_checks.json](data/protocols_and_checks.json). The failed answer raters are not presented as positive or negative accuracy evidence.
