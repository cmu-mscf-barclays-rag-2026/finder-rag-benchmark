# Independent graph RAG

The current graph model works independently of dense retrieval. It needs no
embedding model or vector database. The core implementation is
[`standalone_graph.py`](standalone_graph.py).

## How it works

1. Create a node for each passage and each normalized keyword in the corpus.
2. Connect a passage to the keywords it contains. Connect consecutive passages
   from the same legal section or FinDER reference.
3. Match the question's keywords directly to keyword nodes.
4. Run Personalized PageRank from those nodes and return the top-ranked passages.
5. Pass those passages to the existing local Ollama generator, which cites
   its evidence with `[S1]`, `[S2]`, etc.

For example, a passage can be found through:

```text
question: "alpha"
    -> keyword: alpha
    -> passage A (contains alpha and bridge)
    -> keyword: bridge
    -> passage B (contains bridge and evidence)
```

Passage B can be retrieved even though it does not contain "alpha".
The unit tests verify this behavior. No dense scores or passage seeds enter
this process. Here, "concept" means a normalized keyword; these are simple
text relationships, not extracted factual triples.

## Scoring and interpretability

The random walk restarts at the matched query concepts with probability 0.35.
Otherwise it follows graph edges. Passages with structural neighbors allocate
15% of their outgoing probability to those neighbors; the rest goes to concepts.
Concept-to-passage edges use log-scaled mention counts; passage-to-concept
edges additionally use inverse document frequency to reduce common-term hubs.
Title mentions receive weight 2 before log scaling. Keywords occurring in more
than 75% of passages are excluded (a one-document corpus still works).

Only passage nodes are returned. Each result includes its PageRank score,
a connecting path, and score components from matched concepts, other concepts,
and adjacent passages. The displayed path is one explanation of connectivity;
the score includes all incoming contributions, not just that path. Scores are
ranking weights, not calibrated probabilities of relevance.

The tokenizer lowercases words, removes English stop words and possessives,
and applies a small plural-normalization rule. Exact normalized keyword
matching limits paraphrase and synonym handling. If no concepts match, the
system returns no passages and abstains before generation.

## Run it

From this directory, run:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

The app defaults to **Standalone graph (no embeddings)**. Embedding settings
are disabled in that mode. Ollama is needed for answer generation, but not
for graph construction or retrieval.

Programmatic FinDER usage:

```python
from rag import RAGSettings, build_rag

engine, stats = build_rag(RAGSettings(retrieval_mode="graph"))
response = engine.ask("How did revenue change?")
print(response.answer)
print(response.retrieval_trace)
```

The `graph` build path returns before constructing any embedding model or
vector store. An integration test makes both constructors fail if called and
verifies that graph retrieval and cited generation still succeed.

## Evaluation

The independent retriever was evaluated on all 4,876 passages and 100 questions
from [Legal RAG Bench](https://huggingface.co/datasets/isaacus/legal-rag-bench).
Only corpus text, titles, and document structure enter the graph. Questions,
answers, and relevance labels never create graph edges.

The control uses the exact same query-to-concept links but stops after one
concept-to-passage step. Comparing this control with the full walk isolates
the effect of further graph propagation.

| Method | Precision@5 | Recall / Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| One-step concept matching | 0.042 | 0.21 | 0.1295 | 0.1488 |
| Independent graph PageRank | 0.042 | 0.21 | 0.1192 | 0.1415 |

There are no top-5 hit wins or losses against the one-step control on these
100 queries. The graph is functional and independent; this basic configuration
does not demonstrate an advantage from further propagation on this benchmark.
This is an exploratory public-test result, not an unseen holdout or a claim
about answer correctness.

The graph has 4,876 passage nodes, 11,136 keyword nodes, 326,504 passage-keyword
edges, and 4,152 structural edges. All 100 queries converged. Explanation
coverage is 100%; this measures trace availability, not answer grounding.
Building took 0.82 seconds and graph retrieval including explanations averaged
55.25 ms per query on the measured CPU.

```powershell
.venv\Scripts\python.exe evaluate_standalone_graph.py
.venv\Scripts\python.exe -m pytest -q
```

Metrics at k = 1, 3, 5, 10, configuration, corpus fingerprints, and per-query
paths are saved to:

- [standalone_graph_metrics.csv](results/standalone_graph_metrics.csv)
- [standalone_graph_summary.json](results/standalone_graph_summary.json)
- [standalone_graph_traces.jsonl](results/standalone_graph_traces.jsonl)

Recall/Hit measures gold-passage coverage; MRR and nDCG measure its rank.
Because each question has one gold passage, Precision@5 equals Recall@5 / 5.
Generation correctness and citation grounding are not evaluated by this script.

## Previous experiment

The earlier dense-seeded graph remains available as `retrieval_mode="dense_graph"`
and is explicitly labeled **Dense + graph (previous experiment)** in the app.
Its evaluator is `evaluate_graph.py` and its results remain in
[`results/task_b_graph_report.md`](results/task_b_graph_report.md).
Those results concern a different algorithm. The independent graph does not
consume its rankings, embeddings, or scores.

The app's other mode is `dense`, the original MMR baseline.
