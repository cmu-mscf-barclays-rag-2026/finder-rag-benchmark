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
Title mentions, when a document has a title, receive weight 2 before log
scaling. The team Legal RAG evaluation passes no titles. Keywords occurring in more
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

All graph evaluations follow the team Legal RAG Bench protocol in `legal_rag/`
on `feature/person2-bm25` (Florence's BM25, dense, and hybrid runs), so the
numbers are directly comparable with the team's. The shared pieces live in
[`legal_protocol.py`](legal_protocol.py):

- **Corpus:** all 4,876 passages, passage text only. Titles and footnotes are
  not added, matching BM25 and dense.
- **Split:** questions are grouped by SHA-256 of the gold passage's text, the
  groups are shuffled with seed 42, and development is filled to 20%. The
  result, 20 development and 80 held-out questions, is checked against the
  published `split.json`. Only held-out results are reported.
- **Metrics:** Recall (= Hit), MRR, and nDCG at K = 1, 3, 5, 10, 20, credited
  at the first gold rank. `per_query.csv` and `test_metrics.csv` use
  legal_rag's exact column names.
- **Uncertainty:** a paired Hit@5 bootstrap over gold-passage groups (10,000
  resamples, seed 42).
- **Latency:** legal_rag's harness. That is 4 BLAS threads, 3 development
  warmups, and 3 shuffled, interleaved repeats of the held-out queries.

Every graph method is one fixed configuration, so nothing is selected on
development. Questions, answers, and relevance labels never create graph edges.

The control uses the exact same query-to-concept links but stops after one
concept-to-passage step. Comparing it with the full walk isolates the effect
of further graph propagation.

| Method (held-out, 80) | Recall@5 | MRR@5 | nDCG@5 | Mean ms |
|---|---:|---:|---:|---:|
| One-step concept matching | 0.1750 | 0.1142 | 0.1289 | 26.3 |
| Independent graph PageRank | 0.2000 | 0.1135 | 0.1345 | 39.5 |
| *legal_rag BM25 (reference)* | 0.3750 | 0.2540 | 0.2843 | 11.3 |
| *legal_rag hybrid RRF (reference)* | 0.4000 | 0.2712 | 0.3037 | 44.7 |

- **Propagation barely helps.** PageRank adds 2 top-5 hits with no losses
  against the one-step control (+0.025 Hit@5, interval [0.000, +0.063]).
- **The graph trails BM25 and hybrid.** It is behind BM25 by 0.175 Hit@5
  (interval [-0.291, -0.063]) and behind hybrid by 0.200. Both intervals
  exclude zero.
- **Its candidate list is deep enough to rerank.** Held-out Recall@100 is
  0.625, against 66% for BM25 in the legal_rag Top-100 diagnostic.
- **Latencies are from different machines.** Graph timings are top-20
  retrieval including explanation construction, on this laptop's CPU. The
  reference rows were timed on Florence's machine.

The graph has 4,876 passage nodes, 10,982 keyword nodes, 274,267
passage-keyword edges, and 4,152 structural edges. All 100 queries converged;
building took 0.7 s. Consecutive-passage edges use passage-ID structure, which
BM25 and dense do not use.

```powershell
git show origin/feature/person2-bm25:legal_rag/results/initial/per_query.csv | Out-File -Encoding utf8 legal_rag_per_query.csv
.venv\Scripts\python.exe evaluate_standalone_graph.py --reference-per-query legal_rag_per_query.csv
.venv\Scripts\python.exe -m pytest -q
```

Outputs in [results/legal_standalone_graph/](results/legal_standalone_graph/):

- `split.json` - the published split.
- `dev_metrics.csv` - development metrics at K = 5.
- `test_metrics.csv` and `per_query.csv` - held-out metrics in legal_rag format.
- `paired_hit5_intervals.csv` - paired Hit@5 differences with bootstrap intervals.
- `summary.json` - protocol, corpus fingerprint, graph statistics, and Recall@100.
- `traces.jsonl` - per-question interpretable paths.

Generation correctness and citation grounding are not evaluated.

## LLM-augmented graph

These are this branch's own methods (Yuchen), not a teammate's.
[`llm_graph.py`](llm_graph.py) adds the local Ollama model (`llama3.2`, 3B)
around the unchanged graph, concepts, and PageRank walk. Each piece is a
standard technique:

1. **LLM reranking.** The graph returns its top
   20 passages. For each one, the LLM answers Yes or No to "Does this passage
   help answer the question?". The passages are reordered by P(Yes), read from
   the first answer token's logprobs. This is pointwise LLM reranking, as in
   monoT5-style rerankers. Ties keep the graph's order.
2. **LLM keyword expansion.** The LLM rewrites the question as 6-10
   heading-style search phrases for its underlying legal issue, ignoring names
   and story details. The phrases are linked to corpus concepts and seed a
   second walk. The question keeps a fixed 50% of the seed mass.
3. **Keyword expansion + reranking.** This is the combination of 1 and 2. The
   top 20 of the question walk and the top 20 of the expanded walk are pooled
   (about 29 unique passages per query). The LLM reranks the pool, and
   reciprocal-rank fusion (k = 60, the same formula as the team's hybrid)
   breaks ties. It is the best variant measured and is the app's `graph_llm`
   mode (**Graph + LLM keyword expansion and rerank**; settings
   `llm_query_weight=0.5`, `llm_rerank_pool=20`).

Labels, answers, and embeddings never reach the LLM or the graph. In the app,
each trace entry keeps its graph path and adds `llm_relevance`, `graph_rank`,
and `found_by` (which walk surfaced it).

### Results

Same team protocol as above, 80 held-out questions. Rows marked *legal_rag*
are Florence's BM25, dense, and hybrid runs from `legal_rag/results/initial`
on `feature/person2-bm25`, on the same questions.

| Method | Recall@5 | MRR@5 | nDCG@5 | Mean ms |
|---|---:|---:|---:|---:|
| Independent graph PageRank | 0.2000 | 0.1135 | 0.1345 | 39 |
| LLM keywords, appended to question | 0.1125 | 0.0844 | 0.0915 | 716 |
| LLM keywords, question keeps 50% | 0.1625 | 0.1010 | 0.1164 | 735 |
| Graph top 20 + LLM rerank | 0.3250 | 0.2135 | 0.2417 | 2,573 |
| **LLM keyword expansion + LLM rerank (app `graph_llm`)** | **0.3750** | **0.2623** | **0.2906** | **4,430** |
| *legal_rag BM25* | 0.3750 | 0.2540 | 0.2843 | 11 |
| *legal_rag dense MiniLM* | 0.2875 | 0.1844 | 0.2109 | 14 |
| *legal_rag hybrid RRF* | 0.4000 | 0.2712 | 0.3037 | 45 |

### Protocol and caveats

Each method is one configuration fixed before this evaluation: the 50%
question share and the 20-passage pool were never swept, so nothing was
selected on development.

The LLM variants were designed from failure analysis on an earlier,
non-team 50-question split. That split overlaps these 80 held-out questions,
so their held-out numbers are somewhat optimistic; the graph row is clean.

Only 80 questions are held out, so treat the differences between methods as
preliminary. Only retrieval is measured; answer correctness and groundedness
are not.

All LLM outputs used here are cached in
[`llm_cache.jsonl`](results/legal_llm_graph/llm_cache.jsonl), keyed by the
exact prompt, so metrics reproduce without Ollama. `--live-latency` re-times
the LLM methods with the cache bypassed.

```powershell
git show origin/feature/person2-bm25:legal_rag/results/initial/per_query.csv | Out-File -Encoding utf8 legal_rag_per_query.csv
.venv\Scripts\python.exe evaluate_llm_graph.py --reference-per-query legal_rag_per_query.csv
.venv\Scripts\python.exe evaluate_llm_graph.py --reference-per-query legal_rag_per_query.csv --live-latency
```

The first command fetches Florence's per-question results. The second
reproduces the metrics from the cache. The third also re-times the LLM
methods live, which needs Ollama.

Outputs in [results/legal_llm_graph/](results/legal_llm_graph/) use the
same file set as the standalone evaluation: `split.json`, `dev_metrics.csv`,
`test_metrics.csv`, `per_query.csv`, `paired_hit5_intervals.csv`,
`summary.json`, and `traces.jsonl`. `traces.jsonl` holds each question's LLM
keywords, seeds, gold rank, and top 5 per method.

## Previous experiment

The earlier dense-seeded graph remains available as `retrieval_mode="dense_graph"`
and is explicitly labeled **Dense + graph (previous experiment)** in the app.
Its evaluator is `evaluate_graph.py` and its results remain in
[`results/task_b_graph_report.md`](results/task_b_graph_report.md).
Those results concern a different algorithm. The independent graph does not
consume its rankings, embeddings, or scores.

The app's other mode is `dense`, the original MMR baseline.
