# What Q, H and Q+H change

The generation stage receives the question and fixed instructions, without source passages, reference answers or relevance labels. H is a hypothetical passage, not verified evidence.

| Route | Steps |
|---|---|
| Q | Original question → BM25 → real passages |
| H | Question → generator → H → BM25 → real passages |
| Q+H | Question → generator → H; concatenate Q and H → BM25 → real passages |
| Dense HyDE | Question → generator → H → encoder → vector search → real passages |

## Why the BM25 scores can improve

Legal questions often describe people and situations; the source corpus uses legal categories and rules. H can introduce useful matching vocabulary. In saved question 13, the question describes a judge asking undecided jurors to continue deliberating. H supplies **Black direction**, matching the labeled title **Perseverance (Black) Directions**. Q misses the passage at Top 5; the original H retrieves it.

Generated expansion can also hurt. In question 16, a direct question about good-character evidence expands into credibility, reputation, admissibility and bad character; the labeled passage leaves the Top 5.

Question 25 illustrates the correctness boundary. H calls a CCTV identification scenario **recognition evidence**, while the dataset reference says **comparison evidence**. The relevant passage discusses several identification categories, so the extra vocabulary still helps retrieve it. Retrieval success does not validate H's proposed answer.

These cases support a vocabulary-matching hypothesis. No term-removal ablation or independent human mechanism annotation was performed.

## BM25 details

All routes use the full title+text corpus, k1=1.2, b=0.75, the same tokenizer and stable corpus-index ties. Query-term counts are binary: repeating a word does not directly increase its query weight. The model has no query-length normalization, so a longer query does not automatically dilute every useful term. Added terms may instead increase competing documents' scores.

Q+H preserves original terms as well as generated terms. Its higher score in one round does not by itself establish superiority over H; the fresh trial did not include a direct fresh-Q+H versus fresh-H significance test.

The source case records remain in the retained local archive under `results/person3_sparse_hyde/sparse_query_v1/full/legal/{13,16,25}.json`, with original questions and H in `results/person3_openai_hyde/`. The published statistical tables contain no raw API ledger or credentials.

[Mechanism comparison and results](README.md) · [HyDE paper](https://aclanthology.org/2023.acl-long.99/)
