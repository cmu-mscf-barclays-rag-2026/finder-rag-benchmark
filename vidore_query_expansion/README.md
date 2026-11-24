# ViDoRe Finance fusion follow-up

This follow-up tests where score fusion helps in the [ViDoRe query expansion experiment](../README.md). Both comparisons use the existing 2,942-page corpus, 31 English validation questions and 247 English test questions held out from the original optimizer. The tables below report test results. The CSV files include validation and test metrics at k = 5, 8, and 10.

## 1. Merge OCR and image results for a single question

We used each original question and fixed the top 100 candidates from each modality. The previous reciprocal rank fusion (RRF) used a constant of 60 and weights of 25% OCR and 75% image. The new methods standardize scores within each list. CombSUM adds the standardized scores. CombMNZ multiplies that sum by the number of lists containing the page. A missing page makes no contribution to either the sum or the count. [Sponsor's fusion paper](https://arxiv.org/html/2507.03761v1#S3)

| Method | Recall@10 | nDCG@10 |
|---|---:|---:|
| OCR only | 57.41% | 0.5165 |
| Image only | 62.81% | 0.5791 |
| Equal-weight RRF | 61.91% | 0.5620 |
| Previous weighted RRF | 63.61% | 0.5760 |
| Z-score CombMNZ | 63.38% | 0.5829 |
| Z-score CombSUM | **64.29%** | **0.5860** |

CombSUM led both metrics on this split. Compared with image-only retrieval, it recovered 1.47 percentage points more labeled pages on average at top 10 and gained 0.0068 nDCG@10.

## 2. Merge results across MultiQuery rewrites

We replayed the saved LangChain MultiQuery rewrites, candidate depths and original-question weights for each backend. The OCR/image backend produced each input list using the original first-stage weighted RRF. The comparison then changed the rule that merged those lists. The main table preserves the prior query weights. The CSV also includes an equal-weight comparison between standard CombMNZ and RRF.

| Backend | Original query nDCG@10 | MultiQuery RRF | MultiQuery CombSUM | MultiQuery CombMNZ |
|---|---:|---:|---:|---:|
| BM25 | 0.5037 | **0.5191** | 0.5097 | 0.5083 |
| Dense OCR | **0.5165** | 0.5071 | 0.5057 | 0.5050 |
| Dense image | 0.5791 | 0.5719 | **0.5806** | 0.5791 |
| OCR/image fusion | **0.5760** | 0.5540 | 0.5582 | 0.5590 |

BM25 benefited most from the existing RRF merge. For dense image retrieval, CombSUM recovered the drop from MultiQuery RRF and edged past the original query in both nDCG@10 and Recall@10 (62.98% versus 62.81%). The change was small. The two experiments estimate effects at separate stages. Running both new merge rules together would answer a different question. The reported results measure page retrieval on this split.

## Reproduce

Keep this folder inside `vidore_query_expansion/`. Use the parent project's Python environment, cached ViDoRe dataset, cached page and query embeddings, and experiment cache. These scripts reuse the cache locally and make no API calls. See the [parent benchmark instructions](../README.md) for building the prerequisite cache.

From `vidore_query_expansion/`, substitute the three local cache paths below.

```text
python fusion_followup/compare_fusion.py --data DATA_DIR --vectors VECTOR_DIR --cache EXPERIMENT_CACHE_DIR
python fusion_followup/compare_multiquery_fusion.py --data DATA_DIR --vectors VECTOR_DIR --cache EXPERIMENT_CACHE_DIR
```

Each script writes a summary CSV and a protocol JSON. Pass `--per-query` to also export question-level results. The first script verified 834 original-query baseline rankings. The second verified all 1,112 prior MultiQuery rankings across validation and test. Both checks compare the saved top 20 pages.
