# Task B: interpretable graph RAG

## Delivered design

The implementation is a local passage graph layered on the existing dense
retriever. It is intentionally deterministic and inspectable; it does not use
an LLM to invent entities or relationships.

1. Embed the query and retrieve 10 dense seed passages.
2. Expand one hop over a sparse corpus graph.
3. Score the union of seeds and neighbors with separate dense and graph score
   contributions.
4. Return the top passages, their source text, and an explanation containing
   the seed node, traversed edge, edge evidence, and score decomposition.
5. Give only the selected passages to the existing citation-constrained Ollama
   answer generator.

The Streamlit sidebar now exposes `Dense MMR baseline` and `Interpretable graph
expansion`. Graph explanations appear below the retrieved sources.

## What the graph represents

Each node is one retrievable passage or chunk. Edges are undirected and are
built from corpus text and structure only.

| Edge | Construction | Explanation shown to the user |
|---|---|---|
| `sequence` | Consecutive chunks in one FinDER reference, or consecutive Legal RAG Bench passages in one section | Source/reference and section ID |
| `same_title` | Consecutive passages with the same normalized title | Shared title |
| `shared_citation` | Exact shared Act or case citation | Citation text |
| `shared_title_term` | Rare shared title token or bigram | Shared term |
| `shared_acronym` | Rare shared uppercase acronym | Acronym |
| `shared_term` | Rare shared corpus term | Term |

Terms occurring in only one passage cannot connect nodes. Very common terms
are also excluded. Each node keeps at most 20 concepts and 12 neighbors, which
controls hubs and memory use. Structural and citation edges receive the highest
priority. No question, answer, gold passage ID, or relevance judgment is used
to build the graph.

For seed score \(s\), edge weight \(w\), and graph weight \(a=0.35\):

```text
dense contribution = (1 - a) * min-max-normalized dense seed score
graph contribution = a * normalized source-seed score * w
final score = dense contribution + strongest one-hop graph contribution
```

This strongest-path rule makes every contribution attributable to one path.
It is a transparent baseline, not a claim that one-hop propagation is the best
graph-ranking algorithm.

## Dataset choice

The graph code accepts either project dataset. The measured graph comparison
uses [Legal RAG Bench](https://huggingface.co/datasets/isaacus/legal-rag-bench)
because its corpus exposes stable hierarchical passage IDs, titles, citations,
and exact passage-level qrels.

| Property | FinDER | Legal RAG Bench |
|---|---:|---:|
| Domain | Financial reasoning | Victorian criminal law/procedure |
| Queries | 5,703 | 100 |
| Searchable passages | 5,830 deduplicated references | 4,876 benchmark passages |
| Gold evidence | Usually one or more reference passages | Exactly one most-relevant passage ID |
| Splits | Project-created frozen dev/test query split | One public `test` split only |
| Passage preparation | Project chunks references | Authors already chunked to at most 512 Kanon tokens |
| Best use here | Financial-domain validation and later entity graph | Fast, exact graph-retrieval comparison |

Legal RAG Bench questions are deliberately lexically dissimilar from the gold
passages. Its authors use retrieval at `k=5` and define binary correctness,
groundedness, and retrieval accuracy for end-to-end evaluation. The dataset card
and paper describe 4,876 passages and 100 expert-written questions. The dataset
card's YAML says CC BY-NC-SA 4.0 while its prose says CC BY-NC 4.0; confirm the
intended license before redistribution.

Important limitation: all 100 Legal RAG Bench questions are public test data.
The committed graph configuration was fixed before reading labels and evaluated
once. Do not tune it on these scores and then describe the result as an unbiased
held-out test. FinDER's frozen development/test protocol remains the right place
for parameter development.

## Metrics

### Retrieval metrics implemented

- **Precision@k:** fraction of the `k` returned passages that are gold.
- **Recall@k / Hit Rate@k:** fraction of questions whose single Legal RAG Bench
  gold passage is returned. These are identical for this single-qrel dataset.
- **MRR@k:** average reciprocal rank of the gold passage; sensitive to whether
  evidence is near the top.
- **nDCG@k:** logarithmically discounted gold-passage rank.
- **Query-level wins/losses:** graph finds a gold passage missed by dense, or
  drops one found by dense. This prevents a net score from hiding regressions.
- **Explanation coverage:** fraction of results with a relation and evidence.
- **Graph-influenced result fraction:** fraction receiving a one-hop score,
  including dense seeds that also receive graph support.
- **Graph-only result fraction:** fraction introduced outside the dense seed
  set.
- **Graph-path gold rate:** fraction of queries whose returned gold passage has
  graph support. It is diagnostic and is not the same as a graph-only win.
- **Latency:** query embedding + dense search, with graph traversal added only
  to the graph method. Index construction is reported separately.

With one gold passage, Precision@5 cannot exceed 0.20. Recall/Hit Rate, MRR, and
nDCG are more useful primary metrics than raw Precision for this dataset.

### End-to-end metrics recommended next

To match the Legal RAG Bench methodology, add answer generation at temperature
zero and judge each response for:

- **Correctness:** the response entails the reference answer.
- **Groundedness:** the response is supported by the retrieved passages,
  irrespective of whether those passages are gold.
- **Citation coverage:** supported factual claims divided by all factual claims.
- **Failure decomposition:** hallucination first; otherwise retrieval error when
  the answer is grounded but incorrect and the gold was absent; otherwise
  reasoning error when the answer is grounded but incorrect and gold was
  present.

For the finance project, also stratify metrics by lookup, numeric, table,
comparison, multi-document, long-context, and reasoning-heavy query types.

## Measured 100-question result

The comparison uses the same local MiniLM embeddings for both methods. Titles,
passage text, and footnotes are embedded. Graph construction uses no labels.

| Method | P@5 | Recall/Hit@5 | MRR@5 | nDCG@5 | Latency/query |
|---|---:|---:|---:|---:|---:|
| Dense cosine | 0.056 | 0.28 | 0.1560 | 0.1870 | 15.653 ms |
| Dense + graph | 0.054 | 0.27 | 0.1587 | 0.1869 | 15.816 ms |

At `k=5`, the graph has 3 query-level wins, 4 losses, and 93 ties. It improves
MRR slightly but decreases gold-passage coverage by one percentage point. Graph
traversal adds about 0.163 ms/query and explanation coverage is 100%.

This is a useful negative result: the current graph is interpretable and cheap,
but it does not yet outperform dense retrieval. The 85.6% graph-influenced and
17.4% graph-only result rates at `k=5` suggest expansion is too aggressive. The
next experiment should require multi-edge corroboration or route only
relationship/multi-hop questions to graph retrieval. It should be developed on
FinDER development IDs or a new Legal RAG Bench development set, not the 100
public test labels.

The graph contains 4,876 nodes, 25,446 edges, and 8,554 retained concepts. It
took 3.16 seconds to build on the measured machine; passage embedding took 89.83
seconds on CPU.

## Reproduce

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Quick pipeline check; not a reportable benchmark
python evaluate_graph.py --query-sample-size 10 --device cpu --output-dir tmp/task_b_smoke

# Fixed full public-test evaluation
python evaluate_graph.py --device cpu --output-dir results

# Run the FinDER chatbot and choose Interpretable graph expansion
streamlit run app.py
```

Outputs:

- `results/task_b_graph_metrics.csv`: dense and graph metrics at k = 1, 3, 5, 10
- `results/task_b_graph_summary.json`: protocol, graph size, timing, and wins/losses
- `results/task_b_graph_traces.jsonl`: per-query dense ranking, graph ranking, and explanations
- `results/task_b_graph_report.md`: compact measured-results report

## Sources

- [Legal RAG Bench dataset card](https://huggingface.co/datasets/isaacus/legal-rag-bench)
- [Legal RAG Bench paper](https://arxiv.org/abs/2603.01710)
- [Isaacus methodology article](https://isaacus.com/blog/legal-rag-bench)

