# Source attribution and distribution

The processed child/parent corpora and question records derive from [Isaacus Legal RAG Bench](https://huggingface.co/datasets/isaacus/legal-rag-bench), revision `db0b31dc6d195ce9916897e1ac5e4e6209736c8a`, based on the Judicial College of Victoria's Criminal Charge Book.

The source metadata declares **CC BY-NC-SA 4.0**; its prose license section instead links to **CC BY-NC 4.0**. Both include a noncommercial restriction. We preserve this discrepancy rather than claiming a new license for the source data. The included derived data is attributed to the original authors and must not be treated as MIT-licensed simply because surrounding project code uses that license. Clarify the publisher's applicable terms before commercial sponsor use or redistribution incompatible with those terms.

Modifications: numeric ID grouping; ordered concatenation of original body text; added child/parent mappings, source character spans, split metadata, and measured length fields. The original child bodies and reference answers are preserved. Footnotes remain metadata. Generated parent text does not introduce new legal content or annotations.

References requested by the dataset card:

- Abdur-Rahman Butler and Umar Butler. *Legal RAG Bench: an end-to-end benchmark for legal RAG*. 2026. https://arxiv.org/abs/2603.01710
- Umar Butler, Abdur-Rahman Butler, and Adrian Lucas Malec. *The Massive Legal Embedding Benchmark (MLEB)*. 2025. https://arxiv.org/abs/2510.19365

Token counts use the tokenizer from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Tokenizer files are downloaded to a local cache, not redistributed in this ZIP.
