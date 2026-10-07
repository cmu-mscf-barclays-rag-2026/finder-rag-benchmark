# Florence Liu — Barclays research updates

Start with the [Week 6 Qwen hierarchy experiment](week6_qwen_hierarchy/README.md)
or the completed [Week 5 findings](week5_hierarchy/results/comparison/MEETING_BRIEF.md),
or choose a week below. Each folder contains that week's findings, results and code.

| Week | Research question | Main finding | Review |
|---|---|---|---|
| 3 — BM25 | How well does keyword retrieval perform on FinDER? | Development-tuned BM25 achieved 31.12% Recall@5 on 4,575 test questions. | [Summary and results](week3_bm25/README.md) |
| 4 — Legal RAG | Does hybrid retrieval improve legal evidence retrieval? | Hybrid found the labeled passage for 32/80 questions at Top-5, versus BM25's 30/80; the gain was small and uncertain. | [Summary and results](week4_legal_rag/README.md) |
| 5 — Hierarchical retrieval | Does grouping child passages into parents help? | Hybrid Top-10 hits increased from 33/80 to 49/80 with unlimited context; the advantage disappeared under a 2,048-token cutoff. | [Summary and results](week5_hierarchy/README.md) |
| 6 — Qwen hierarchy | Can longer embedding windows and budget-aware sibling selection preserve hierarchy benefits? | Qwen improved dense child Hit@10 from 24/80 to 44/80 under 2,048 tokens, but direct parents reached only 36/80; hierarchy helped only with unlimited delivery. | [Findings and protocol](week6_qwen_hierarchy/README.md) |

These weeks use different experiments and evaluation settings. Week 3 uses
FinDER; Weeks 4–6 use Legal RAG Bench, with Week 4 on a different question split. Scores
should be compared within each week's controlled experiment, not across weeks.
Retrieval scores measure evidence inclusion, not the correctness of generated answers.

For implementation details, each week's summary links to its reproduction guide.
The Week 3 team comparison is [labeled separately](week3_bm25/results/team_comparison/README.md);
inherited baseline code and supporting files are kept under
[Week 3 shared reference](week3_bm25/shared_reference/README.md).
