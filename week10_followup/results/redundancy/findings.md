# Neighbor-page overlap diagnostic

For Qwen256 Top-10, each delivered neighbor page was compared with the
delivered seed pages in the same context. This uses cosine similarity between
saved Qwen page vectors, which encode only the first 256 tokens of each page.
It is an exploratory overlap signal, not a measure of factual or full-page
semantic redundancy.

| Budget | Neighbor policy | Mean delivered neighbors | Compared neighbors | Cosine ≥ .8 | Cosine ≥ .9 | Exact OCR-text duplicates |
|---:|---|---:|---:|---:|---:|---:|
| 6k | 1/0 | 3.18 | 984 | 2.5% | 0.3% | 0 |
| 6k | 0/1 | 2.79 | 863 | 3.0% | 0.8% | 0 |
| 6k | 1/1 | 4.18 | 1,292 | 2.4% | 0.4% | 0 |
| 12k | 1/0 | 5.85 | 1,807 | 3.6% | 0.9% | 0 |
| 12k | 0/1 | 5.43 | 1,678 | 4.0% | 0.9% | 0 |
| 12k | 1/1 | 7.85 | 2,425 | 3.5% | 0.8% | 0 |

Under this strict short-window proxy, near-duplicate neighbors are uncommon.
That does not establish that the other neighbors contribute distinct facts:
the vector ignores the rest of each page, and high-level semantic overlap
can occur below either threshold. The per-question counts are in
`per_query.csv` for later manual review.
