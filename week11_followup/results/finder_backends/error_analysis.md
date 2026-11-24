# FinDER error analysis

The same 200 held-out questions, pooled reference-passage labels, and saved Week 10 GPT rewrites are used. `error_analysis.csv` contains the question, rewrites, first judged-reference rank, and failure class for each question.

## Where BM25 MultiQuery misses

At Top-10 it hits 76/200 questions and misses 124/200. For 101 misses, **none** of the original or two rewritten BM25 searches returns a judged reference in its Top-20. For the other 23 misses, a judged reference enters at least one Top-20 candidate list but does not reach the fused Top-10. The first group needs better candidate retrieval; the second can potentially benefit from ranking or fusion changes.

## What happens at candidate depth 100

The tuned BM25 index replays every saved Top-20 source ranking exactly. Retrieving 100 instead of 20 from each of the same three formulations raises candidate-level question coverage from 99/200 to 134/200. Of the 101 questions with no judged reference in any Top-20 source list, 35 gain a candidate by depth 100 and 66 still have none. But applying the same equal-weight RRF to all Top-100 lists yields 71/200 Hit@10 versus 76/200 at source depth 20: 0 gains and 5 losses. Candidate recall improves, but the extra candidates do not improve the final Top-10 under this fusion rule.

## What the rewrites change

Against the original BM25 Top-10, MultiQuery gains 12 question hits and loses 4. In 8 of the gains, the original BM25 Top-20 already contained a judged reference, so the rewrites helped promote evidence. The other 4 gains brought judged evidence absent from the original Top-20. Of the 4 losses, 3 still have a judged reference in the fused Top-20, showing a ranking problem.

Two examples for manual review: question `632f159a` uses the term *input metrics* and the judged passage says *input method*; its rewrites use *advancement* and *status*, moving the reference from rank 9 to 18. Question `a0adb935` uses *Dec 23*, which a rewrite renders as *December 23* even though the judged reference is a fourth-quarter 2023 table; that may alter the intended date. The reference moves from rank 5 to 17. These examples suggest mechanisms; they do not establish a general causal pattern.

## What hybrid adds

Hybrid MultiQuery gains 9 Top-10 hits relative to BM25 MultiQuery and loses 8. All 8 losses retain a judged reference within hybrid's Top-20, so their failure is rank placement. Among its 123 Top-10 misses, 25 have a judged reference in the union of the fused BM25 and dense Top-20 lists; 98 do not. This is a diagnostic of those two backend lists, not every original per-formulation candidate.

## Next controlled tests

1. Audit a small sample from each failure class in `error_analysis.csv`, including whether the benchmark reference supports the question. Record abbreviation, entity, date, and financial-term drift in the rewrites.
2. On a development split, test a fusion or reranking rule that gives the original query more weight or protects its high-ranked evidence. Check whether it can promote the newly found depth-100 candidates without losing current Top-10 hits. Evaluate the frozen rule on a fresh test split and compare Top-5 with Top-10.
3. For references still absent from all three Top-100 searches, test a stronger dense encoder or domain-aware reformulations. A fusion adjustment alone cannot recover references absent from every source list.

The depth-100 comparison uses the same 200 test questions and is exploratory; choose any new ranking rule on development data and validate it on fresh held-out questions. These are retrieval labels only. They do not establish whether the passages fully answer the question.
