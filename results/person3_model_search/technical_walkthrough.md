# Technical walkthrough: question-preserving HyDE

## Three separate model roles

| Role | Model / method | Purpose |
|---|---|---|
| Hypothetical-passage generator | HuggingFaceTB/SmolLM3-3B | Generate one hypothetical evidence paragraph from the question. The best observed dense configuration used the focused prompt and a 160-new-token cap. |
| Embedding encoder | sentence-transformers/all-MiniLM-L6-v2 | Encode the original question, generated paragraph and each corpus passage into vectors. Maximum 256 tokens; normalized vectors. |
| Sparse retriever | BM25, k1=1.2, b=0.75 | Search the corpus using the original question's words. BM25 is a scoring method, not another generative model. |

LFM2.5-1.2B-Instruct was the second generator tested. SmolLM3 is the correct name; “SOMA3” is not the model name.

## 1. Generate a hypothetical paragraph

Input: only the original question, not its reference answer or gold evidence.

Use the model's official chat template, non-thinking mode, greedy generation, repetition penalty1.2, no-repeat3-gram. The focused prompt asks for a short Victorian-law textbook paragraph, preserving the legal issue and avoiding invented citations or facts. The 160-token limit is a maximum, not a guaranteed paragraph length or quality. Generated passages can still contain mistakes.

This paragraph is a search aid, not the final answer to the user.

## 2. Mix question and paragraph embeddings

Let q be the original question, h the generated paragraph and E the fixed MiniLM encoder:

```python
q_vec = normalize(E(q))
h_vec = normalize(E(h))
search_vec = normalize(0.5 * q_vec + 0.5 * h_vec)
passage_vecs = normalize(E(corpus_passages))  # precomputed
scores = passage_vecs @ search_vec
dense_ranking = stable_argsort_descending(scores)[:100]
```

Equal weights apply to **vectors**, not concatenated text, LLM parameters, or final-answer votes. Normalizing q_vec + h_vec gives the same direction as the equal-weight expression.

The best observed dense Hit@5 was37%; question-only dense was29%. The 37% configuration's MRR was0.1975; the original-prompt mixture reached35% with a higher MRR0.2250. More questions recovered does not imply better ranking on every metric.

## 3. Add BM25 using weighted Reciprocal Rank Fusion (RRF)

Run BM25 on the **original question** to get another top100 list. Combine that list with the mixed-vector dense list:

```python
rrf_score[passage] = (
    0.5 / (60 + bm25_rank[passage])
    + 0.5 / (60 + dense_rank[passage])
)
```

Ranks are one-based. A passage outside a retriever's top100 contributes zero for that retriever; merge the candidate union and keep the highest-scoring five passages. This fuses **ranks**, not raw BM25 and cosine scores.

The best observed SmolLM3 hybrid used50% BM25 /50% dense and reached40% Hit@5, with MRR0.2555 and nDCG0.2918. Standard hybrid used75% BM25 /25% question-only dense and reached36%. Therefore the new maximum changes generator/input construction and fusion weight together; it is not a generator-only improvement.

## Two different 50/50 choices

| Stage | What is split50/50? |
|---|---|
| Dense input | Original-question embedding and generated-paragraph embedding |
| Hybrid fusion | BM25 ranking and the resulting dense ranking |

## Evidence status

The 37% and40% are exploratory maxima on the same100 previously inspected questions. Exact McNemar p-values versus their controls were approximately0.0574 and0.4545; neither is significant at5%. Holm correction across32 comparisons gave1.0 for both selected tests. These are retrieval results, not answer accuracy.

Only rounded aggregate observations and the recorded protocol were recovered from execution logs after workspace cleanup. Original generated passages, per-query rankings and final raw-output validation are unavailable. This walkthrough explains the executed algorithm; it is not a replacement for the missing end-to-end experiment package.
