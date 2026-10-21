# Span-level evidence pilot: results

Actual CPU BM25 retrieval over the full canonical corpus, evaluated on the 20 inherited development questions. All evidence labels remain assistant-proposed drafts. No test-set score, new dense-model experiment or answer-generation result is claimed.

## Corpus and annotations

2,785 reconstructed parents, 4,876 original children and 15,617 grandchildren. Grandchildren contain at most 128 standalone WordPiece tokens. There are 27 proposed evidence spans across 20 questions; six questions have multiple spans within the same original gold passage. Human-approved questions: **0**.

## Top five results with a 2,048-token body budget

| Corpus | All spans included | Evidence token coverage | Whole gold child included | Mean returned tokens |
|---|---:|---:|---:|---:|
| parent | 35.0% | 35.0% | 35.0% | 2038.1 |
| child | 40.0% | 40.0% | 40.0% | 1834.7 |
| grandchild | 10.0% | 24.9% | 0.0% | 621.1 |

## Findings

- Fine-grained evaluation can credit useful text without requiring the entire original gold chunk. Grandchild top-5 results include every proposed span for 2/20 questions, despite never returning a complete original gold child. This illustrates why the evidence target should not be defined by retrieval chunk boundaries.
- Grandchildren are not the strongest retriever in this fixed experiment: they have lower full-evidence inclusion than children. Top five grandchildren also return far fewer tokens; the common budget is a cap, not a guarantee of equal context size.
- Parent full-span inclusion falls from 45% without a cap to 35% at 2,048 tokens for k=5. Child inclusion stays at 40%. These figures are specific to the draft pilot and are not directly comparable to the earlier 80-question Qwen evaluation.
- A boundary control confirms that one whole span and two adjacent pieces give identical coverage, while a missing tail lowers it. That control is not a retrieval result.

## Verification and unresolved work

22 software tests passed. An independent character-set audit reproduced all 360 query/settings and 18 aggregate rows, including token coverage and context budgets. All span slices and source hashes match. These checks cannot establish semantic sufficiency.

Human review of REVIEW_SHEET.md is still required. Q48 has a possibility/probability mismatch; Q25 and Q58 also merit explicit review. Alternative sufficient evidence remains unannotated. No truly multi-document questions have been verified. Scores must be regenerated if labels change.

## Suggested meeting summary

> I built a 20-question pilot that labels exact supporting text in our existing sources instead of treating a whole chunk as the answer. The evaluator tracks that text across parent, child and smaller chunks, including when the context budget cuts text off. I ran a BM25 demonstration and checked the scores independently. The key distinction is that source offsets are verified, while the proposed evidence selections still need human review before this becomes a validated benchmark.
