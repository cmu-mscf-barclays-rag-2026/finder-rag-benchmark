# MultiQuery and DSPy across datasets

Meeting 10 · 17 November 2026

Compare methods **within each dataset**. Legal token-F1 and LongBench multiple-choice accuracy are different metrics.

| Dataset | Test n | Answer metric | BM25 + reader | MultiQuery + reader | BM25 + DSPy CoT |
|---|---:|---|---:|---:|---:|
| Legal | 80 | Lexical token-F1 | 0.2788 | 0.3085 | 0.2518 |
| LongBench v2 | 402 | MC accuracy | 0.3557 | 0.3085 | 0.3458 |

- **Legal:** MultiQuery improves Hit@5 from 32.5% to 42.5% (exploratory p=0.02148). None of the answer comparisons passes Holm3.
- **LongBench:** MultiQuery accuracy is lower than BM25. DSPy's paired interval includes zero, and its optimizer retained the default instruction.
- **Scope:** Both use `gpt-4o-mini`, Top 8 and a 2,400-word evidence budget. Corpus units, prompts, splits and answer formats differ. DSPy changes the reader, not retrieval.

Legal was run and independently audited locally. LongBench comes from the teammate's hash-verified [published artifacts](https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark/blob/8738b543b8770a29acb5d82c481b1a67fe530140/GRAPH_RAG_README.md#longbench-v2-with-gpt-4o-mini-multiqueryretriever-and-dspy); it was not rerun.

[Legal results and charts](../meeting10_legal_20261117/README.md) · [Six-row table with costs and intervals](team_primary_comparison.csv) · [All 43 metric rows](team_all_metrics_long.csv) · [Detailed settings and provenance](DETAILS.md)
