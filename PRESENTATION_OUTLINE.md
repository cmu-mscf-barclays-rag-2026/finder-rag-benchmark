# Presentation outline: Yuchen's part (~7 min)

Target: 10 slides, about 6.5 min of talk plus buffer. Times are per slide.
Numbers are test or held-out results, with dev-selected configurations.
Full metric tables (nDCG@10, MRR@10, Recall@1/5/10/100 with intervals) are in
`GRAPH_RAG_README.md`; use them as backup slides.

---

## 1. Title and roadmap (0:15)

- Retrieval for RAG: keyword, semantic, graph, and LLM-assisted search
- Roadmap: background → metrics → methods → results → takeaways

---

## 2. Keyword vs semantic search (0:45)

- **Keyword (sparse):** TF-IDF → BM25. Exact term overlap; strong on names,
  numbers, identifiers.
- **Semantic (dense):** word2vec → BERT → Sentence-BERT. Text becomes a
  vector; similarity = cosine. Catches paraphrase, misses exact terms.
- **Vector search backends:** FAISS (flat, IVF, product quantization) and
  HNSW (navigable small-world graph). Approximate nearest neighbours at scale.
- Our scale: 178k chunks, so exact cosine on a laptop GPU (6.5 min) is enough.

> Say: the rest of the talk asks when each kind of search wins.

---

## 3. Metrics, protocol, hyperparameters (0:45)

- **Retrieval:** Recall@k (is the gold in the top k?), MRR@10 (how high is
  the first hit?), nDCG@10 (rank-discounted gain). Recall@100 = the ceiling
  a reranker could reach.
- **End-to-end (LongBench v2):** multiple-choice QA accuracy with a
  gpt-4o-mini reader. Chance 0.25, most-common letter 0.30.
- **Protocol:** select on dev, run test once; paired bootstrap 95% intervals.
- **Hyperparameters:** top k (chunks given to the reader), chunk size
  (300 words), evidence budget (2,400 words), BM25 k1/b, RRF k = 60,
  gate threshold for retries.

---

## 4. Methods covered (0:30)

| Mine (this talk) | Teammates' / not covered here |
|---|---|
| Graph RAG (keyword + heading graph, PageRank) | Multi-vector (MVR), subchunking |
| Selective one-retry agentic search | Multimodal embeddings, ViDoRE |
| Multi-query retrieval (LangChain) | |
| DSPy prompt optimization | |
| Rank fusion (RRF, CombSUM/MNZ, ISR, Borda) | |
| Top-k budget sweep | |

- Baselines throughout: team BM25 and team MiniLM dense.
- Hierarchical and sibling retrieval appear inside the graph (heading tree).

---

## 5. Graph RAG: no embeddings, interpretable (1:00)

- Nodes: passages, keywords, section/heading tree. Question keywords seed
  Personalized PageRank; top-ranked passages go to the reader.
- Every hit comes with a path: `term:selective → heading → passage`.
- Three construction fixes, found by dev error analysis:
  1. Query links weighted IDF × keyword volume (stops names dominating)
  2. Passages inherit their headings' words
  3. Heading tree links siblings (hierarchical retrieval)
- **Legal RAG Bench (80 held-out):**

| | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| Original graph | 0.200 | 0.114 | 0.135 |
| **Improved graph** | **0.388** | **0.293** | **0.316** |
| BM25 | 0.375 | 0.254 | 0.284 |
| Dense MiniLM | 0.288 | 0.184 | 0.211 |
| Hybrid (BM25 + dense RRF) | 0.400 | 0.271 | 0.304 |

> Say: +0.19 Recall@5 over the original graph (interval excludes 0); matches
> BM25 and hybrid, best on rank quality.

---

## 6. Does the graph transfer? Plus agentic retry (0:50)

| nDCG@10 (test) | Legal | FiQA | FinQA | FinanceBench |
|---|---:|---:|---:|---:|
| Graph (best mode) | **0.331** | 0.161 | 0.725 | 0.488 |
| BM25 | 0.288 | 0.234 | **0.759** | 0.349 |
| Dense MiniLM | 0.246 | **0.369** | 0.374 | **0.547** |

- Graph propagation helps only where documents have structure (legal headings).
- Graph misses are length bias (prefers long passages) and wrong-year near misses.
- **Selective one-retry:** if confidence (top score / top-20 mean) is low,
  gpt-4o-mini writes one follow-up query. FinanceBench nDCG@10 0.488 → 0.528
  (+0.040, interval excludes 0) for $0.06 total.

---

## 7. LongBench v2: baselines, multi-query, DSPy (0:50)

503 long-document multiple-choice questions; 402 test; gpt-4o-mini reader; 8 chunks.

| Method | Accuracy | vs BM25 |
|---|---:|---:|
| No context | 0.266 | −0.090 |
| Dense MiniLM | 0.303 | −0.052 |
| **BM25** | **0.356** | |
| Graph | 0.328 | −0.027 |
| Multi-query (LangChain) | 0.308 | −0.047 |
| DSPy chain-of-thought (MIPROv2) | 0.346 | −0.010 |

- Multi-query rewrites drop the answer choices (keep 20% of choice words).
- DSPy's optimizer kept the default prompt; ties BM25.
- Dense trails because MiniLM reads only 256 tokens and questions need exact matches.

---

## 8. Top k vs budget (0:30)

| Accuracy, k = | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| BM25 | 0.323 | 0.323 | **0.356** | 0.341 |
| Graph | 0.321 | 0.323 | 0.328 | 0.336 |
| Multi-query | 0.321 | 0.316 | 0.308 | 0.341 |
| DSPy | 0.318 | 0.308 | 0.346 | **0.366** |

- More context is not monotonically better; dev picks k = 8.
- Multi-query's loss shows up only at k = 8, so it is fragile.

---

## 9. Rank fusion (0:40)

- Fuse BM25 and dense top 64, keep 8. Rules: RRF, CombSUM, CombMNZ
  (min-max and z-score), ISR, BordaFuse (França et al. 2025).

| Accuracy (test) | Score | vs dense | vs BM25 |
|---|---:|---:|---:|
| BM25 + dense, 6 rules | 0.333–0.343 | +0.03 to +0.04 | −0.01 to −0.02 |
| **Graph + dense, RRF** | **0.348** | **+0.045** (sig.) | −0.007 |

- Fusion fixes most of dense's gap but never beats BM25; the rule barely matters.
- Graph complements dense at least as well as BM25 does.

---

## 10. Takeaways (0:25)

- No single retriever wins: BM25 for exact-match text, dense for short
  passages, graph where documents have structure.
- LLM add-ons help only when targeted: a gated retry helps; generic query
  rewrites hurt.
- Retrieval choice barely changes LLM cost (~$0.75–0.86 per 1,000 questions).
- Could be promising: a stronger embedder that reads whole chunks;
  choice-preserving rewrite prompts.
