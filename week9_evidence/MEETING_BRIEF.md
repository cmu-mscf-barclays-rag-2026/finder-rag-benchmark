# Week 9 — Florence's November 9 meeting follow-up

This week's deliverable is a working evidence-sufficiency prototype and a
finance retrieval trial on **309 English questions and 2,942 ViDoRe pages**.
The prototype distinguishes alternative sources for one fact from separate
facts that must all be supported. It measures fact recall, all-required-fact
coverage, evidence per token, marginal fact gain and the minimum sufficient
budget in the tested grid.

**The annotation pilot is a draft, not validated ground truth:** ten questions,
twenty required facts, exact source spans and a [human-review packet](annotations/REVIEW.md).
No reader model or visual embedding model was evaluated.

## Measured findings

| Retrieval configuration | Hit@10 | nDCG@10 |
|---|---:|---:|
| BM25, full OCR pages | 0.8964 | 0.5333 |
| MiniLM, 256-token page prefixes | 0.6311 | 0.3107 |
| Qwen3-Embedding-0.6B, 256-token page prefixes | 0.7476 | 0.3945 |

Qwen outperformed MiniLM under the chosen configurations. BM25 had the highest
scores, but it sees full pages while the dense models see short prefixes.
MiniLM truncates 93.9% of pages. These results do not establish that sparse
retrieval beats full-page dense retrieval.

At a fixed 6,000-token delivery budget, increasing seed depth from 10 to 20
changed Qwen's complete-page hits from 208/309 to 209/309 and BM25's from
244/309 to 244/309. Expanding two preceding and one following page reduced
Qwen's hits to 192/309 and BM25's to 227/309. The earlier legal-corpus gains
from deeper retrieval and siblings therefore did not transfer directly to
this whole-page finance setting.

On the ten-question draft pilot at Top10 and 4,000 tokens, page Hit was 70%
for BM25, 80% for MiniLM and 70% for Qwen. **All three covered every drafted
requirement for only 5/10 questions.** Some single pages covered every fact
despite incomplete relevant-page recall; other relevant pages missed required
facts. This illustrates the intended distinction, subject to annotation review.

## Implementation finding and validation

The first Qwen attempt exposed a bug in the inherited encoder: manual token
slicing removed the final end-of-text token from long inputs. Week 9 now uses
tokenizer-managed truncation, checks that suffix and separates the corrected
cache from affected vectors. The invalid run is clearly isolated and excluded
from the comparison above.

The final saved Qwen vectors passed an independent **12-input CPU spot check**,
including vectors computed in full-sized batches: minimum cosine agreement
was 0.99999982. All **15 tests** passed. Historical Week 6/8 outputs were not
rewritten; truncated-window results need a separate audit.

## Next research decision

Review the ten question decompositions and their alternative support against
the source PDFs before interpreting sufficiency as answerability. The review
packet flags ambiguous reporting periods, an entity mismatch, answer leakage,
OCR corruption and supporting pages absent from published qrels. Then freeze
the rubric and evaluate new questions. A longer-window dense run or a
matched-prefix BM25 control can isolate the current input-length difference.

See [full findings and costs](results/findings.md),
[paired comparisons](results/paired_comparisons.csv), and
[reproduction commands](README.md).
