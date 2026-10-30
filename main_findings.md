# GPT-5.4 HyDE — main findings

**Legal gains did not transfer consistently to FiQA or LongBench v2.**

![Nine retrieval methods across three datasets](presentation/openai_hyde/cross_dataset.svg)

| Dataset | Primary metric | Question dense | Pure HyDE | Question + HyDE | Takeaway |
|---|---|---:|---:|---:|---|
| Legal · 100 questions | Hit@5 | 29% | **47%** | **45%** | ↑ Improvement on a reused development set |
| FiQA · 648 questions | nDCG@10 | **0.3687** | 0.3124 | 0.3668 | ≈ Mixed HyDE; ↓ pure HyDE |
| LongBench v2 · 503 questions | Answer accuracy | **44.53%** | 41.55% | 41.75% | ↓ Dense HyDE; BM25 best observed at **46.72%** |

All HyDE values above use focused160. Compare within rows; the three metrics are different. Bold marks the highest value among the three displayed dense methods, not statistical significance.

## What changed this week

| Component | Previous local-model work | This run |
|---|---|---|
| HyDE generator | SmolLM3 / Qwen experiments | `gpt-5.4-2026-03-05` via API; reasoning `none` |
| Prompt / budget | Original prompt, 96 output tokens | Retained original96 on legal; added focused160 with domain wording |
| Dense encoder | `all-MiniLM-L6-v2` | Same pinned encoder; no OpenAI embeddings or fine-tuning |
| Retrieval | Dense, vector mixing, BM25/RRF | Fixed question/HyDE 50:50 mix; matched d25/d50 hybrid controls |
| Evaluation | Legal retrieval | Legal + full FiQA test + all 503 LongBench questions |
| Answer generation | No legal answer evaluation | Fixed GPT-5.4 reader for LongBench only, top 5 real chunks |

## How the methods differ

| Method | Dense query | BM25 query | Final ranking |
|---|---|---|---|
| Question dense | Original question | — | Dense similarity |
| Pure HyDE dense | Generated hypothetical paragraph | — | Dense similarity |
| Question + HyDE dense | Normalize(0.5 question vector + 0.5 HyDE vector) | — | Dense similarity |
| Standard hybrid | Original question | Original question | RRF fusion |
| HyDE hybrid | Pure or mixed HyDE vector | Original question | RRF fusion |

RRF uses the top 100 from each route, constant 60, and dense weights 25% or 50%. Generated paragraphs guide retrieval; they are not evidence supplied to the reader.

## Evidence strength

| Finding | Paired inference / qualification |
|---|---|
| Legal mixed dense: 29% → 45% | Holm p = 0.0185 within the legal 12-test family |
| Legal pure dense: 29% → 47% | Holm p = 0.0707; not significant at adjusted 5% |
| FiQA pure HyDE: 0.3687 → 0.3124 | Significant decline; Holm p ≈ 0.00012 |
| FiQA mixed hybrid d50: 0.3452 → 0.3522 | Matched-weight comparison not significant; Holm p ≈ 0.9246 |
| LongBench | No planned comparison significant after Holm correction |

Legal questions were previously inspected. FiQA and LongBench share a separate 12-test correction family. LongBench tests are question-level and do not adjust for shared contexts. The observed best method is descriptive, not independently selected.

## Completion & details

| Result set | Evaluation | Validation | Open report |
|---|---|---|---|
| Legal | ✓ 100 questions; 200 HyDE generations | Independent replay passed | [Legal findings](results/person3_openai_hyde/findings.md) |
| FiQA | ✓ 648 queries / 57,638 documents | Saved-result validation passed | [Transfer findings](results/person3_cross_dataset/gpt54_focused160_v1/findings.md) |
| LongBench | ✓ 503 questions; 5,030 logical predictions | All predictions + 2,515 RRF rankings checked; dense replay **219/1,509**, remaining cache unavailable | [Transfer findings](results/person3_cross_dataset/gpt54_focused160_v1/findings.md) |

LongBench is a custom top-5-chunk RAG evaluation, not the official full-context leaderboard. Full contexts were searched; only selected chunks reached the reader. No new model pilot is included here.

**Next:** freeze a small three-dataset pilot; vary only the HyDE generator first, keep the LongBench reader fixed, then separately test embeddings / reranking / evidence budget. Query drift and limited evidence coverage are hypotheses to test.
