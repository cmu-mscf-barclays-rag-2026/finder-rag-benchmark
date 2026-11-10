# Feasibility assessment: ViDoRe V3 Finance–EN

ViDoRe supports a matched comparison of page images and OCR text for financial documents. The public corpus pairs both representations for each page and provides graded relevance and annotated regions. The run covers all 2,942 pages in six bank 10-Ks and all 1,854 query variants. The 309 original English questions form the primary result. Translations are summarized by language.

## Measured English results

| Method | Hit@5 | Recall@5 | nDCG@10 | Complete-answer-page Hit@5 | Estimated median query ms |
|---|---:|---:|---:|---:|---:|
| BM25 OCR | 0.7896 | 0.4727 | 0.5302 | 0.7523 | 0.45 |
| Qwen OCR | 0.7929 | 0.4715 | 0.5275 | 0.7207 | 48.57 |
| Qwen image | 0.8673 | 0.5373 | 0.5931 | 0.8153 | 48.59 |
| Text + image RRF | 0.8544 | 0.5146 | 0.5760 | 0.7883 | 49.65 |

**Qwen image** had the highest English nDCG@10 in this run. The image-minus-OCR difference using the same Qwen checkpoint was +0.0656. Complete-answer-page Hit@5 uses the 222 English questions with a grade-2 page. Image retrieval added 36 top-five page hits missed by OCR and lost 13 hits that OCR found. Among the 23 English questions tagged for charts, images or infographics, OCR and image nDCG@10 were 0.6157 and 0.6661, respectively. 

## Compute

- OCR: 733.5 s encoding, 0.93 s matrix assembly, 23.0 MiB of float32 vectors.
- IMAGE: 1226.9 s encoding, 1.47 s matrix assembly, 23.0 MiB of float32 vectors.
- BM25: 1.40 s to build postings, 9.7 MiB in numeric postings before counting the vocabulary and Python objects.

The run retained 19 pages with empty OCR in the shared candidate pool. The encoding time starts with prepared input records and includes model preprocessing and inference. The query total sums measured encoder and ranking components. It is an estimate for local sequential execution.

## Evidence evaluation and recommendation

Use page metrics for the full benchmark and add a separate fact-coverage layer for a reviewed set of questions. The two worked examples link specific facts to OCR offsets and preserve the publisher's query/page regions. For example, a question comparing two segments can score a page hit when only one of the two mapped fact pages is retrieved. Grade-1 and grade-2 labels score page relevance. They do not enumerate all facts required across pages or verify what text or pixels an answer model ultimately used.

For a first English ranking baseline under this configuration, use Qwen image as the measured reference by nDCG@10. Before generalizing to Barclays documents, check additional report types and inspect where each input format adds evidence. The corpus contains six reports from one sector and year. The visual path caps page images at 786,432 pixels and OCR encoding uses a 4,096-token input limit. Both input limits can remove detail before ranking. Timing describes one laptop setup, and the comparisons do not test answer generation.
