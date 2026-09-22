# Task B - interpretable graph retrieval

## Protocol

The fixed graph configuration was evaluated once on all 100 public Legal RAG
Bench questions against the complete 4,876-passage corpus. Both methods use
`sentence-transformers/all-MiniLM-L6-v2`, cosine similarity, and the same title
+ passage + footnote representation. The graph begins with 10 dense seeds,
expands one hop, and uses a graph weight of 0.35.

The graph is built only from corpus structure and text. Questions, answers,
gold passage IDs, and relevance labels do not create or weight edges. Because
the benchmark has only one public test split, these numbers are a fixed public
benchmark result, not a tunable held-out estimate.

## Results

| Method | k | Precision | Recall / Hit Rate | MRR | nDCG |
|---|---:|---:|---:|---:|---:|
| Dense cosine | 1 | 0.0800 | 0.08 | 0.0800 | 0.0800 |
| Dense + graph | 1 | 0.0800 | 0.08 | 0.0800 | 0.0800 |
| Dense cosine | 3 | 0.0767 | 0.23 | 0.1450 | 0.1668 |
| Dense + graph | 3 | **0.0800** | **0.24** | **0.1517** | **0.1744** |
| Dense cosine | 5 | **0.0560** | **0.28** | 0.1560 | **0.1870** |
| Dense + graph | 5 | 0.0540 | 0.27 | **0.1587** | 0.1869 |
| Dense cosine | 10 | **0.0380** | **0.38** | 0.1686 | **0.2186** |
| Dense + graph | 10 | 0.0360 | 0.36 | **0.1713** | 0.2166 |

At the primary `k=5`, graph expansion produces 3 wins, 4 losses, and 93 ties
relative to dense retrieval. Its slight MRR improvement means some retrieved
gold passages move higher, but its one-point Hit Rate decline means the method
does not improve overall evidence coverage.

## Interpretability and cost

- Explanation coverage: **100%**
- Graph-influenced top-5 results: **85.6%**
- Graph-only top-5 results: **17.4%**
- Gold passages with a graph path at top 5: **23%**
- Dense latency: **15.653 ms/query**
- Dense + graph latency: **15.816 ms/query**
- Incremental graph traversal: **0.163 ms/query**
- Graph build: **3.16 seconds**
- Passage embedding: **89.83 seconds on CPU**

The graph has 4,876 nodes, 25,446 undirected edges, and 8,554 retained concepts.
Every returned graph path records its seed node, edge type, evidence string,
and separate dense and graph score contributions.

## Decision

Keep this implementation as the interpretable graph baseline, not as the new
default retriever. The measured result does not justify replacing dense search.
The graph-only rate shows that one-hop expansion is too eager. The next valid
experiment is query routing plus corroborated expansion: invoke the graph only
for relationship/multi-hop questions and require either two supporting seeds or
a structural/citation edge before a graph-only passage can displace a dense
seed. Develop that rule on FinDER's frozen development IDs or a newly created
legal development set; do not tune it against these 100 public test labels.
