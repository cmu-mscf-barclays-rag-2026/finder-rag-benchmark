# Cross-dataset results for presentation

Meeting 10 follow-up · 18 November 2026 · 41 selected headline rows

Read each change against the **control in its own row**. The datasets use different questions, retrieval units, and outcome metrics.

| Study | Test n | Setting | Method | Primary metric | Score | Compared with | Δ | Secondary |
|---|---:|---|---|---|---:|---|---:|---|
| FinDER · Florence | 200 | Ranked passages · K=10 | BM25 original | Hit@10 | 34.0% | — | — | nDCG@10 24.4% |
| FinDER · Florence | 200 | Ranked passages · K=10 | MultiQuery · Qwen | Hit@10 | 35.5% | BM25 original | +1.5 pp | nDCG@10 24.7% |
| FinDER · Florence | 200 | Ranked passages · K=10 | MultiQuery · GPT-4o mini | Hit@10 | 38.0% | BM25 original | +4.0 pp | nDCG@10 26.0% |
| FinDER · Florence | 200 | Ranked passages · K=10 | DSPy · Qwen | Hit@10 | 33.0% | BM25 original | -1.0 pp | nDCG@10 23.0% |
| FinDER · Florence | 200 | Ranked passages · K=10 | DSPy · GPT-4o mini | Hit@10 | 38.0% | BM25 original | +4.0 pp | nDCG@10 26.1% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 6k tokens | BM25 OCR | Hit@10 | 79.0% | — | — | Recall 49.6% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 6k tokens | MiniLM 256 | Hit@10 | 57.6% | — | — | Recall 29.5% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 6k tokens | Qwen 256 | Hit@10 | 67.3% | — | — | Recall 38.1% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 6k tokens | Qwen 1024 | Hit@10 | 77.3% | Qwen 256, same budget | +10.0 pp | Recall 46.0% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 12k tokens | BM25 OCR | Hit@10 | 89.6% | — | — | Recall 58.6% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 12k tokens | MiniLM 256 | Hit@10 | 62.8% | — | — | Recall 34.1% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 12k tokens | Qwen 256 | Hit@10 | 74.4% | — | — | Recall 44.2% |
| ViDoRe · Florence | 309 | Delivered OCR pages · K=10 · 12k tokens | Qwen 1024 | Hit@10 | 82.2% | Qwen 256, same budget | +7.8 pp | Recall 54.0% |
| FiQA · Florence | 648 | Delivered answer passages · K=10 · 6k tokens | BM25 original | Hit@10 | 48.5% | — | — | Recall 29.7% |
| FiQA · Florence | 648 | Delivered answer passages · K=20 · 6k tokens | BM25 original | Hit@20 | 55.9% | — | — | Recall 35.9% |
| FiQA · Florence | 648 | Delivered answer passages · K=20 · 12k tokens | BM25 original | Hit@20 | 56.0% | — | — | Recall 36.1% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · BM25 OCR | Original query | nDCG@10 | 50.4% | — | — | Hit@10 87.4% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · BM25 OCR | MultiQuery · GPT | nDCG@10 | 51.9% | Original, same backend | +1.5 pp | Hit@10 88.3% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · BM25 OCR | DSPy initial · GPT | nDCG@10 | 52.4% | Original, same backend | +2.1 pp | Hit@10 88.3% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · BM25 OCR | DSPy optimized · GPT | nDCG@10 | 51.9% | Original, same backend | +1.5 pp | Hit@10 89.1% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense OCR | Original query | nDCG@10 | 51.7% | — | — | Hit@10 85.4% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense OCR | MultiQuery · GPT | nDCG@10 | 50.7% | Original, same backend | -0.9 pp | Hit@10 84.6% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense OCR | DSPy initial · GPT | nDCG@10 | 51.0% | Original, same backend | -0.6 pp | Hit@10 85.0% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense OCR | DSPy optimized · GPT | nDCG@10 | 50.7% | Original, same backend | -1.0 pp | Hit@10 85.0% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense image | Original query | nDCG@10 | 57.9% | — | — | Hit@10 90.7% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense image | MultiQuery · GPT | nDCG@10 | 57.2% | Original, same backend | -0.7 pp | Hit@10 91.5% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense image | DSPy initial · GPT | nDCG@10 | 57.5% | Original, same backend | -0.4 pp | Hit@10 89.5% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · Dense image | DSPy optimized · GPT | nDCG@10 | 56.9% | Original, same backend | -1.0 pp | Hit@10 90.7% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · OCR/image fusion | Original query | nDCG@10 | 57.6% | — | — | Hit@10 91.5% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · OCR/image fusion | MultiQuery · GPT | nDCG@10 | 55.4% | Original, same backend | -2.2 pp | Hit@10 90.7% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · OCR/image fusion | DSPy initial · GPT | nDCG@10 | 57.4% | Original, same backend | -0.2 pp | Hit@10 90.3% |
| ViDoRe · Kevin | 247 | Ranked pages · K=10 · OCR/image fusion | DSPy optimized · GPT | nDCG@10 | 57.1% | Original, same backend | -0.5 pp | Hit@10 91.5% |
| Legal · team branch | 80 | Ranked passages · K=5 | BM25 original | Hit@5 | 32.5% | — | — | — |
| Legal · team branch | 80 | Ranked passages · K=5 | MultiQuery · GPT | Hit@5 | 42.5% | BM25 original | +10.0 pp | — |
| Legal · team branch | 80 | Answer · Top-8 · 2,400 words | BM25 + fixed reader | Lexical token F1 | 27.9% | — | — | — |
| Legal · team branch | 80 | Answer · Top-8 · 2,400 words | MultiQuery + fixed reader | Lexical token F1 | 30.8% | BM25 + fixed reader | +3.0 pp | — |
| Legal · team branch | 80 | Answer · Top-8 · 2,400 words | BM25 + DSPy reader | Lexical token F1 | 25.2% | BM25 + fixed reader | -2.7 pp | — |
| LongBench v2 · Yuchen | 402 | Answer · Top-8 · 2,400 words | BM25 + fixed reader | MC accuracy | 35.6% | — | — | — |
| LongBench v2 · Yuchen | 402 | Answer · Top-8 · 2,400 words | Graph RAG + fixed reader | MC accuracy | 32.8% | BM25 + fixed reader | -2.7 pp | — |
| LongBench v2 · Yuchen | 402 | Answer · Top-8 · 2,400 words | MultiQuery + fixed reader | MC accuracy | 30.8% | BM25 + fixed reader | -4.7 pp | — |
| LongBench v2 · Yuchen | 402 | Answer · Top-8 · 2,400 words | BM25 + DSPy reader | MC accuracy | 34.6% | BM25 + fixed reader | -1.0 pp | — |

## Presenter notes

- FinDER uses 200 hash-selected test questions from 4,575; its scores are a pilot. The retrieval corpus is 5,830 pooled reference passages.
- Florence’s ViDoRe rows score **delivered** OCR pages after context packing on 309 questions. Kevin’s rows score **ranked** pages on a different 247-question optimizer-held-out English split. Their raw scores should not be compared as a paired test.
- Kevin’s nDCG changes use each backend’s original query as the control. Florence’s Qwen1024 changes use the same Qwen encoder at 256 page tokens and the same budget.
- Legal DSPy and LongBench DSPy optimize the answer reader. FinDER and Kevin use DSPy for query rewriting. Legal answer F1 is a lexical proxy; LongBench is multiple-choice accuracy.
- The table selects headline cutoffs. The underlying files retain other K values, budgets, paired intervals, and per-question detail.

## Sources

- Florence local: [combined results](../../week10_followup/results/comparison/comparison.csv), [FinDER details](../../week10_followup/results/finder_comparison/findings.md), and [Qwen-window details](../../week10_followup/results/qwen_window_compare/findings.md).
- Kevin `kevinzhang@8df8598`: [64 English test rows](../../week10_followup/results/comparison/team_vidore_query_expansion.csv), derived from `vidore_query_expansion/results/summary.csv` in his branch.
- Legal `feature/person3-hybrid@18bf882`: [team metric rows](../../week10_followup/results/comparison/team_method_comparison.csv).
- LongBench `yuchenlu@8738b54`: `results/longbench_v2_methods/accuracy.csv` in that branch; the BM25, MultiQuery, and DSPy scores reconcile to the team metric rows.
