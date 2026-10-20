# Independent graph RAG

The current graph model works independently of dense retrieval. It needs no
embedding model or vector database. The core implementation is
[`standalone_graph.py`](standalone_graph.py).

## How it works

1. Create a node for each passage, each normalized keyword ("concept"), and
   each section and markdown heading of the source document.
2. Connect each passage to the keywords it contains, including the words of
   the headings it sits under. Connect each passage to its heading, and each
   heading to its parent heading or section.
3. Match the question's keywords directly to keyword nodes.
4. Run Personalized PageRank from those nodes and return the top-ranked passages.
5. Pass those passages to the existing local Ollama generator, which cites
   its evidence with `[S1]`, `[S2]`, etc.

For example, development question 34 asks whether a prosecutor may treat a
suspect's "selective silence" as evidence of guilt. The gold passage
`4.15-c3-s5` never says "selective"; its heading does. The improved graph
ranks it first (the original graph ranked it 40th) and explains why:

```text
passage:4.15-c3-s5 "4.15 Silence in Response to People in Authority > Selective Silence"
  top incoming:  term:silence    -[mentions_concept]->          0.000171
                 term:selective  -[heading_mentions_concept]->  0.000148
                 term:answer     -[mentions_concept]->          0.000121
```

Structure also connects passages. A passage can be reached through a sibling
under the same heading:

```text
term:alpha -[mentions_concept]-> passage:1.2-c1-s1 -[under_heading]-> section:1.2
           -[under_heading]-> passage:1.2-c1-s2
```

Unit tests check heading inheritance and structural paths on small
fixtures. No dense scores or passage seeds enter
this process. "Concept" means a normalized keyword; these are text and
document-structure relationships, not extracted factual triples.

## Graph construction

The original graph linked passages to their keywords and to the next passage.
Failure analysis on the 20 development questions found three construction
problems, fixed below. `LEGACY_SETTINGS` in `standalone_graph.py` rebuilds the
original graph exactly (it reproduces the original held-out numbers).

1. **Query links weighted by IDF x concept volume** (`query_links="volume"`).
   A concept node splits its mass over every passage that mentions it. With
   IDF-only links, a word in one passage, such as a name like "Harry", sent
   all of its weight to that passage, while "juror" spread its weight over
   hundreds. Names and story details therefore took the largest seed weights.
   Multiplying each link by the concept's total edge weight makes the first
   walk step score every passage by the sum over matched concepts of
   IDF x log-scaled mention count, the familiar tf-idf form. A passage now
   has to match several question concepts to rank.
2. **Passages inherit their headings' words** (`heading_words=True`). In this
   corpus a heading appears only in the first split of its block, so later
   splits never repeat it (`1.5-c7-s2` sits under "Pre-trial Publicity" but
   never says "publicity"). Each passage now also counts the words of the
   section title and headings it sits under, `title_weight` = 2 times before
   the usual log scaling. This is the rule the graph already applied to
   document titles. It adds 17,522 passage-concept edges that come only from
   headings; the path labels them `heading_mentions_concept`.
3. **A section/heading tree replaces consecutive-passage edges**
   (`structure="headings"`). Each section (724, named by its `#` title) and
   markdown heading (3,536 `##`/`###` headings) becomes a node. Reading in
   order, a passage sits under the heading in effect where it starts and under
   any heading that begins inside it; each heading sits under its parent.
   Consecutive edges linked the last split of one topic to the first split of
   the next; the tree links passages through the heading they share, and the
   path names it. FinDER chunks hang from one node per reference document.

Unchanged: the tokenizer (lowercasing, English stop words, possessives, and a
small plural rule), log-scaled mention weights, the PageRank settings, and the
75% document-frequency cut-off.

Rejected on development: merging `-ing`/`-ed`/`-ly` forms into corpus words
(`word_forms="corpus"`) improved deep recall but lowered dev nDCG@5 (0.303 vs
0.372), so it is off by default. Phrase nodes for corpus collocations and
BM25 length-normalized edges also lowered dev scores in prototyping and were
not kept.

## Scoring and interpretability

The random walk restarts at the matched query concepts with probability 0.35.
Otherwise it follows graph edges:

- A passage sends 15% of its outgoing probability to the heading(s) it sits
  under and the rest to its concepts, weighted by IDF x log-scaled mentions.
- A concept splits its probability over the passages that mention it, by
  log-scaled mentions.
- A heading or section sends 15% to its parent and splits the rest over its
  passages and subheadings by the number of passages each covers, so every
  passage under a heading receives the same share.

Keywords occurring in more than 75% of passages are excluded (a one-document
corpus still works).

| Node | Example ID | Readable label |
|---|---|---|
| Passage | `passage:1.5-c7-s2` | breadcrumb: "1.5 Decide Solely on the Evidence > Pre-trial Publicity" |
| Concept | `term:publicity` | the keyword itself |
| Section | `section:1.5` | its title: "1.5 Decide Solely on the Evidence" |
| Heading | `heading:4.15#9` | its wording: "Selective Silence" |
| FinDER document | `document:<row_id>/ref1` | none |

| Relation | Meaning |
|---|---|
| `mentions_concept` | the passage's own text contains the keyword |
| `heading_mentions_concept` | only a heading above the passage contains it |
| `under_heading` | a passage under a heading, or a heading under its parent |

In a breadcrumb, `>` descends to a subheading and `|` marks a sibling heading
that begins inside the passage.

Only passage nodes are returned. Each result includes its PageRank score, a
labeled connecting path, and score components from matched concepts, other
concepts, and document structure. The path is the shortest one from a query
concept, preferring stronger edges; it is one explanation of connectivity,
and the score includes all incoming contributions, not just that path. Scores
are ranking weights, not calibrated probabilities of relevance.

Exact normalized keyword matching limits paraphrase and synonym handling. If
no concepts match, the system returns no passages and abstains before
generation.

## Run it

From this directory, run:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

The app defaults to **Standalone graph (no embeddings)**. Embedding settings
are disabled in that mode. Ollama is needed for answer generation, but not
for graph construction or retrieval. The trace panel shows each labeled path.

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

**Selection.** All construction choices were made on the 20 development
questions and frozen before the held-out run; every row below is a fixed
configuration. Development nDCG@5 was 0.150 for the original graph, 0.279
with volume links, and 0.372 for the improved graph. Heading words with
consecutive edges scored 0.376: the two differ by one rank on one question,
and the heading tree was kept as the more interpretable structure.
Questions, answers, and relevance labels never create graph edges.

The one-step control uses the improved graph's exact query links but stops
after one concept-to-passage step, isolating the effect of propagation.

| Method (held-out, 80) | Recall@5 | MRR@5 | nDCG@5 | Mean ms |
|---|---:|---:|---:|---:|
| Original graph | 0.2000 | 0.1135 | 0.1345 | 18.4 |
| + IDF x volume query links | 0.3500 | 0.2806 | 0.2980 | 17.7 |
| + heading words (consecutive edges) | 0.3750 | 0.2883 | 0.3099 | 18.7 |
| **+ heading tree = improved graph (app default)** | **0.3875** | **0.2925** | **0.3156** | **24.0** |
| Improved graph, one step only | 0.3750 | 0.2794 | 0.3030 | 3.7 |
| Heading tree without heading words | 0.3375 | 0.2727 | 0.2887 | 22.8 |
| Improved graph + corpus word forms (rejected on dev) | 0.3750 | 0.2898 | 0.3106 | 23.0 |
| *legal_rag BM25* | 0.3750 | 0.2540 | 0.2843 | 11.3 |
| *legal_rag dense MiniLM* | 0.2875 | 0.1844 | 0.2109 | 13.7 |
| *legal_rag hybrid RRF* | 0.4000 | 0.2712 | 0.3037 | 44.7 |

- **The improved graph beats the original.** +0.1875 Hit@5 (interval
  [+0.077, +0.300]), 18 wins and 3 losses.
- **Query-link weighting is the largest fix.** It alone adds +0.150 Hit@5
  ([+0.053, +0.250], 14 wins and 2 losses). Heading words and the heading
  tree add another +0.0375 ([-0.025, +0.107], 4 wins and 1 loss). That step's
  interval includes zero, but it also lifts MRR@5 and nDCG@5 and was chosen on
  development.
- **It now matches the team's lexical and hybrid baselines.** Against BM25:
  +0.0125 Hit@5 (6 wins, 5 losses), with higher MRR@5 (0.293 vs 0.254) and
  nDCG@5 (0.316 vs 0.284). Against hybrid: -0.0125 Hit@5 (11 wins, 12
  losses), with higher MRR@5 and nDCG@5. None of these differences is
  significant on 80 questions.
- **Most of the gain is in the first step.** The walk adds +0.0125 Hit@5 over
  the one-step control ([0.000, +0.038], 1 win, no losses) and +0.013 nDCG@5,
  for about 20 ms more per query.
- **Its candidate list is deeper.** Held-out Recall@100 rose from 0.625 to
  0.7625, against 66% for BM25 in the legal_rag Top-100 diagnostic.
- **Latencies are from different machines.** Graph timings are top-20
  retrieval including explanation construction, on this laptop's CPU; compare
  them within this table only. The reference rows were timed on Florence's
  machine.

The improved graph has 4,876 passage nodes, 10,985 keyword nodes, 724
section and 3,536 heading nodes, 291,789 passage-keyword edges, and 9,668
structural edges. All 100 queries converged; building took 1.0 s. The
heading tree and heading words use the passage-ID hierarchy and the headings
inside passage text, document structure that BM25 and dense do not use.

```powershell
git show origin/feature/person2-bm25:legal_rag/results/initial/per_query.csv | Out-File -Encoding utf8 legal_rag_per_query.csv
.venv\Scripts\python.exe evaluate_standalone_graph.py --reference-per-query legal_rag_per_query.csv
.venv\Scripts\python.exe -m pytest -q
```

Outputs in [results/legal_standalone_graph/](results/legal_standalone_graph/):

- `split.json` - the published split.
- `dev_metrics.csv` - development metrics at K = 5 (the selection evidence).
- `test_metrics.csv` and `per_query.csv` - held-out metrics in legal_rag format.
- `paired_hit5_intervals.csv` - paired Hit@5 differences with bootstrap intervals.
- `summary.json` - protocol, selection note, settings and statistics per graph, and Recall@100.
- `traces.jsonl` - per-question labeled paths from the improved graph.

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
   breaks ties. It is the app's `graph_llm` mode (**Graph + LLM keyword
   expansion and rerank**; settings `llm_query_weight=0.5`,
   `llm_rerank_pool=20`). It was the best variant on the original graph, but
   not on the improved one (below).

Labels, answers, and embeddings never reach the LLM or the graph. In the app,
each trace entry keeps its graph path and adds `llm_relevance`, `graph_rank`,
and `found_by` (which walk surfaced it).

### Results

Same team protocol as above, 80 held-out questions, with the improved graph
underneath every method. Rows marked *legal_rag* are Florence's BM25, dense,
and hybrid runs from `legal_rag/results/initial` on `feature/person2-bm25`,
on the same questions.

| Method | Recall@5 | MRR@5 | nDCG@5 | Mean ms |
|---|---:|---:|---:|---:|
| **Improved graph alone** | 0.3875 | 0.2925 | 0.3156 | 23 |
| LLM keywords, appended to question | 0.4000 | 0.2633 | 0.2973 | 640 |
| LLM keywords, question keeps 50% | 0.4500 | 0.3127 | 0.3467 | 660 |
| Graph top 20 + LLM rerank | 0.3125 | 0.2102 | 0.2357 | 2,469 |
| LLM keyword expansion + LLM rerank (app `graph_llm`) | 0.4125 | 0.2640 | 0.3012 | 4,340 |
| *legal_rag BM25* | 0.3750 | 0.2540 | 0.2843 | 11 |
| *legal_rag dense MiniLM* | 0.2875 | 0.1844 | 0.2109 | 14 |
| *legal_rag hybrid RRF* | 0.4000 | 0.2712 | 0.3037 | 45 |

- **The 3B reranker no longer helps.** On the improved graph, reranking the
  top 20 loses 0.075 Hit@5 (interval [-0.202, +0.051], 8 wins and 14
  losses). Expansion + rerank gains +0.025 Hit@5 ([-0.125, +0.177], 17 wins
  and 15 losses) but has lower MRR@5 and nDCG@5 than the graph alone, at
  about 185 times the latency.
- **Development favors the graph alone.** Its dev nDCG@5 is 0.372, against
  0.248-0.310 for every LLM variant. Under the team's selection rule, the
  graph alone is the configuration to use.
- **Keyword expansion has the best held-out numbers, but that is not a
  selection signal.** With the question keeping 50% of the seed mass it
  reaches Recall@5 0.450 (+0.0625 over the graph, [-0.037, +0.163], 10 wins
  and 5 losses), but it trailed the graph on development (0.302).
- **On the original graph the picture was the reverse.** There the graph
  alone scored 0.2000 Recall@5, reranking 0.3250, and expansion + rerank
  0.3750; LLM keyword expansion without reranking scored 0.1625 or less.
  `--legacy-graph` reproduces those numbers from the cache.

### Protocol and caveats

Each method is one configuration fixed before this evaluation: the 50%
question share and the 20-passage pool were never swept.

The LLM variants were designed from failure analysis on an earlier,
non-team 50-question split. That split overlaps these 80 held-out questions,
so their held-out numbers are somewhat optimistic; the graph row is clean.

Only 80 questions are held out, so treat the differences between methods as
preliminary. Only retrieval is measured; answer correctness and groundedness
are not.

All LLM outputs used here are cached in
[`llm_cache.jsonl`](results/legal_llm_graph/llm_cache.jsonl), keyed by the
exact prompt, so metrics reproduce without Ollama. `--live-latency` re-times
the LLM methods with the cache bypassed; in this run the graph alone was timed
with the legal_rag harness and the LLM methods in one live pass.

```powershell
git show origin/feature/person2-bm25:legal_rag/results/initial/per_query.csv | Out-File -Encoding utf8 legal_rag_per_query.csv
.venv\Scripts\python.exe evaluate_llm_graph.py --reference-per-query legal_rag_per_query.csv
.venv\Scripts\python.exe evaluate_llm_graph.py --reference-per-query legal_rag_per_query.csv --live-latency
.venv\Scripts\python.exe evaluate_llm_graph.py --reference-per-query legal_rag_per_query.csv --legacy-graph --output-dir results/legal_llm_graph_legacy
```

The first command fetches Florence's per-question results. The second
reproduces the metrics from the cache. The third also re-times the LLM
methods live, which needs Ollama. The fourth reruns everything on the
original graph; copy `llm_cache.jsonl` into the new output directory first so
it needs no new LLM calls.

Outputs in [results/legal_llm_graph/](results/legal_llm_graph/) use the
same file set as the standalone evaluation: `split.json`, `dev_metrics.csv`,
`test_metrics.csv`, `per_query.csv`, `paired_hit5_intervals.csv`,
`summary.json`, and `traces.jsonl`. `traces.jsonl` holds each question's LLM
keywords, seeds, gold rank, and top 5 per method.

## Selective routing and one-retry agentic search

These are this branch's own experiments (Yuchen). The question was when
extra retrieval work is worth it. Every number below is on the improved
graph, under the same team protocol.

### Routing headroom

A per-question router can gain only where the two choices disagree. The
oracle row picks the better choice for each question using the labels, so it
is the most any router could gain.

| Choice (held-out Hit@5) | Option A | Option B | Oracle |
|---|---:|---:|---:|
| Graph propagation off / on | 0.3750 | 0.3875 | 0.3875 |
| Single pass / LLM-reformulated retry | 0.3875 | 0.4500 | 0.5125 |
| Graph / *legal_rag BM25* | 0.3875 | 0.3750 | 0.4500 |

**Routing graph propagation has no headroom.** The one-step search never
finds a passage the full walk misses, on either split. The improved graph's
value lies in its construction, not in deciding per query whether to walk;
the walk costs about 20 ms, so it is simply always on. The retry decision
has the most headroom: the single pass and the retry fail on different
questions, and an oracle that retries only the 16% of questions it helps
reaches 0.5125.

### One-retry prototype

[`agentic_graph.py`](agentic_graph.py) (`OneRetryGraphRetriever`) takes at
most one extra search step:

```text
graph search  ->  confidence = top PageRank score / mean of the top 20
              ->  confident: return the first ranking (no LLM call)
              ->  otherwise: LLM keyword reformulation, second walk seeded by
                  50% question + 50% reformulation; return that ranking
```

The second walk already combines both versions of the query, so no separate
merge step is added. In exploration, reciprocal-rank fusion of the two
rankings never hurt on development but recovered less of the retry gain on
held-out (at most 0.425 Recall@5), and LLM reranking hurts on this graph
(above). Each trace entry adds `retried`,
`confidence`, `confidence_threshold`, and `reformulation` to the graph path.

Three gates were fixed before the held-out run. Each retries when its
signal falls below the signal's median over the 20 development questions; the
median needs no labels and aims at about half the queries.

- **Peakedness:** the prototype's gate, a standard score-based
  query-performance predictor.
- **Coverage:** the share of the question's link weight that the top passage
  mentions.
- **LLM judge:** CRAG-style; the highest LLM P(relevant) among the top 3
  passages, which costs 3 extra LLM calls.

| Method (held-out, 80) | Recall@5 | MRR@5 | nDCG@5 | Retried | LLM calls/query | ~ms |
|---|---:|---:|---:|---:|---:|---:|
| Single pass (improved graph) | 0.3875 | 0.2925 | 0.3156 | 0% | 0 | 23 |
| Always retry | 0.4500 | 0.3127 | 0.3467 | 100% | 1.00 | 683 |
| **Gated by peakedness (prototype)** | **0.4500** | **0.3327** | **0.3616** | 42% | 0.42 | 304 |
| Gated by coverage | 0.4125 | 0.3077 | 0.3337 | 41% | 0.41 | 295 |
| Gated by LLM judge | 0.4375 | 0.3206 | 0.3494 | 50% | 3.50 | 720 |
| *Oracle gate (uses labels; upper bound)* | 0.5125 | 0.3710 | 0.4059 | 16% | 0.16 | 131 |

Latency is the expected cost per query, built from the live step latencies
measured in the LLM-graph evaluation: the graph walk is 23 ms and the LLM
reformulation about 640 ms. The pipeline was not re-timed end to end.

- **On held-out, the gate pays for itself.** The prototype matches
  always-retry Recall@5 and has higher MRR@5 and nDCG@5, with 42% of the LLM
  calls and under half the latency. Against the single pass it gains +0.0625
  Hit@5 ([0.000, +0.139], 7 wins and 2 losses); against always-retry the
  difference is 0.000 (3 wins, 3 losses).
- **The gate itself adds little over chance.** The budget curves sweep the
  retried share against random gating. At equal budgets of 40-60%,
  peakedness beats random gating by about 0.02 Hit@5 and 0.02 nDCG@5, about
  two questions. Most of the gain comes from retrying at all.
- **No signal predicts when a retry hurts.** Questions the retry fixes lean
  toward low confidence, but the five it breaks are spread across the
  confidence range for all three signals. That is why the oracle stays far
  ahead.
- **The LLM judge costs more than it saves.** It needs three LLM calls on
  every query to avoid one reformulation on some.
- **Development does not confirm the gain.** On the 20 development questions,
  every retry variant scores at or below the single pass (nDCG@5: single pass
  0.372; LLM-judge gate 0.373; always retry 0.302; peakedness gate 0.292).
  A retry changes Hit@5 for only 3 of them, so development cannot select or
  validate a gate. The reformulation prompt was also designed on an earlier
  split overlapping held-out. Under the team rule, the single-pass graph stays
  the default, and one-retry search remains an opt-in prototype.

This matches the literature on selective query expansion: deciding per query
whether to expand from retrieval-confidence predictors gives modest,
inconsistent gains (Cronen-Townsend, Zhou and Croft, 2004; Amati, Carpineto
and Romano, 2004). The useful next step is more labeled development
questions, not a more elaborate gate.

```powershell
.venv\Scripts\python.exe evaluate_agentic_graph.py --reference-per-query legal_rag_per_query.csv
```

This reads cached LLM outputs and live latencies from
[results/legal_llm_graph/](results/legal_llm_graph/), so it needs no Ollama.
It also checks that `OneRetryGraphRetriever` reproduces the evaluated ranking
on every question. Outputs in
[results/legal_agentic_graph/](results/legal_agentic_graph/) follow the
usual file set plus:

- `budget_curves.csv` - Hit@5, MRR@5, nDCG@5, cost, and LLM calls per query
  for each gate, random gating, and the oracle, at retried shares from 0% to
  100%, on both splits.
- `gate_costs.csv` - thresholds, retried shares, and expected cost per method.
- `traces.jsonl` - each question's signals, reformulation, both top 5s, and
  whether the retry helps.
- `summary.json` - routing headroom, thresholds, measured step costs, and
  protocol notes.

## Previous experiment

The earlier dense-seeded graph remains available as `retrieval_mode="dense_graph"`
and is explicitly labeled **Dense + graph (previous experiment)** in the app.
Its evaluator is `evaluate_graph.py` and its results remain in
[`results/task_b_graph_report.md`](results/task_b_graph_report.md).
Those results concern a different algorithm. The independent graph does not
consume its rankings, embeddings, or scores.

The app's other mode is `dense`, the original MMR baseline.
