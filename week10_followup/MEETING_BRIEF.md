# Week 10 meeting brief — Florence Liu

## Last meeting's retrieval follow-ups

The ViDoRe finance grid reuses the same saved rankings for 309 English
questions over 2,942 OCR pages. It tests seed depths 5, 8, 10, and 20;
0/0, 1/0, 0/1, and 1/1 physical-neighbor policies; and 6k, 10k, and 12k
context-token budgets. It records the pages actually packed, seed pages
excluded by the budget, skipped neighbors, unused tokens, and relevance
metrics. All 5,562 configurations overlapping Week 9 reproduce their saved
delivered IDs and token counts exactly.

At 6k tokens, Qwen256 Top-10 without neighbors delivers 6.57 pages on
average and hits a judged relevant page for 208/309 questions. Top-20
delivers 7.26 pages and 209/309 hits, while excluding 12.74 of its twenty
seeds on average. Raising the budget to 12k lets Qwen256 Top-20 deliver
13.32 pages and 240/309 hits. This separates context-capacity saturation
from ranking depth: ranked Top-20 hits 82.5% of questions before packing,
versus 67.6% delivered at 6k and 77.7% at 12k.

Neighbor expansion changes which evidence fits. For Qwen256 Top-10 at 6k,
adding one preceding page gains hits on 12 questions but loses hits on 18;
one following page gains 10 and loses 14. At 12k the corresponding preceding
comparison gains 12 and loses 13, and following gains 7 and loses 10. On the
unreviewed ten-question fact rubric, preceding neighbors increase drafted
fact recall from .525 to .683 at 6k. This small pilot needs human review and
does not overturn the full benchmark page-relevance result.

The same depth and budget settings on FiQA's 648 test questions behave
differently. All ten Top-10 BM25 answer passages fit within 6k tokens, using
2,393 tokens on average; increasing the budget to 12k changes no Top-10
retrieval score. At Top-20, 19.60 passages fit at 6k and all twenty fit at
10k. FiQA's tested limit is mostly candidate depth, while ViDoRe's
full-page limit is often context capacity. FiQA passages and ViDoRe pages
have different relevance judgments, so their raw Hit rates are not pooled.

A controlled Qwen page-input change from 256 to 1,024 tokens raises ranked
Top-10 Hit from .748 to .822 and delivered Top-10/6k Hit from .673 to .773
on the same 309 questions. The latter gains 39 question-level hits and loses
8, while mean delivered pages change only from 6.57 to 6.61. All query
embeddings match exactly, and a twelve-input independent CPU check passes.
The longer window still truncates 1,039 pages; the
[full findings](results/qwen_window_compare/findings.md) separate this
effect from budget packing.

Detailed traces and metrics: [ViDoRe grid](results/context_grid/findings.md),
[FiQA grid](results/fiqa_grid/findings.md). A short-window Qwen cosine-overlap
diagnostic finds at most 4.0% of delivered neighbors above .8 similarity to a
delivered seed in the tested Top-10 settings; this does not establish
full-page factual independence. See the [overlap findings](results/redundancy/findings.md).

## FinDER query expansion

The FinDER experiment uses a single tuned BM25 index over 5,830 pooled
reference passages. LangChain MultiQueryRetriever generates two alternative
queries and includes the original. A DSPy `BootstrapFewShot` program generates
two alternatives after optimizing Recall@5 on 16 development questions.
Both methods search the same index to depth 20 per query and fuse unique
candidate lists with equal-weight reciprocal-rank fusion. Evaluation uses
the same 200 hash-selected held-out questions from FinDER's 4,575-question
test partition for both generators.

Baseline BM25 Hit@5 is .290 and Hit@10 is .340 on this sample. With GPT-4o
mini, MultiQueryRetriever scores .285/.380 at Top-5/Top-10 and DSPy scores
.305/.380. With the pinned local Qwen3-0.6B generator, MultiQuery scores
.305/.355 and DSPy scores .270/.330. Thus GPT's two expansion methods both
add .040 Top-10 Hit over baseline, while Qwen MultiQuery adds .015 and Qwen
DSPy loses .010. GPT DSPy beats Qwen DSPy by .050 Top-10 Hit on the same
questions; its paired bootstrap interval is +.015 to +.085. The GPT
MultiQuery-versus-Qwen Top-10 difference is +.025 with an interval crossing
zero. Top-5 gains over baseline are small and uncertain for every method.

Each expansion method makes one generation call per test question, in
addition to DSPy's three development optimization calls. GPT test-stage
token-based estimates are $0.00652 for MultiQuery and $0.01657 for DSPy at
published list prices. Mean end-to-end query latency is about 1.0/1.5
seconds for GPT MultiQuery/DSPy and 3.7/7.0 seconds for local Qwen on this
machine; these are different execution paths, not a hardware-normalized
model speed ranking. The sample baseline is below the historical full-test
BM25 Hit@5 of .321; the pilot scores should not be extrapolated to all test
questions. These are retrieval metrics, not answer accuracy or complete
evidence sufficiency.

Full paired metrics, token counts, and latency: [FinDER findings](results/finder_comparison/findings.md).

## Cross-dataset interpretation and team handoff

The [combined comparison](results/comparison/comparison.md) keeps corpus,
query set, retrieval unit, cutoff, budget, and metric scope with each row.
Its deltas are within matched settings. The committed Legal RAG Bench
MultiQuery study improves its Top-5 retrieval hit rate from .325 to .425 on
80 questions, while its DSPy study optimizes the answer reader rather than
query retrieval. The committed LongBench v2 study scores multiple-choice
answer accuracy: MultiQuery .308 versus its .356 baseline, and DSPy .346.
Those changes cannot be pooled with FinDER retrieval deltas because the
pipeline stage and outcome differ. The aggregator reads both committed
studies and can add later teammate files using the
[handoff schema](team_results/README.md).

Kevin's new committed ViDoRe query-expansion study adds a separate
247-English-question test partition and four backends: BM25 OCR, dense OCR,
dense image, and OCR/image fusion. On BM25, MultiQuery raises nDCG@10 from
.504 to .519; its initial DSPy query program reaches .524, while the
optimized version reaches .519. The original question remains best at
Top-10 for the other three backends by nDCG. The full 64-row Top-1/5/10/20
English test table is in [the team ViDoRe CSV](results/comparison/team_vidore_query_expansion.csv).
This study uses GPT-4o mini generation, MIPROv2 instruction optimization,
and a different test subset and retrieval setup from Florence's grid. Its
recorded 7,639 successful generation requests have an estimated $0.7695
list-price token cost across its multilingual run and tuning; that scope is
much larger than Florence's 200-question FinDER pilot. Kevin's
meeting action to report Top-3 and the unique image-versus-OCR gains remains
open; the committed query-expansion study does not supply those diagnostics.
