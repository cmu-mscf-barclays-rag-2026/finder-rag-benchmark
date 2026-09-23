# Data provenance and attribution

Dataset: [Legal RAG Bench, Isaacus](https://huggingface.co/datasets/isaacus/legal-rag-bench).

Pinned revision: `db0b31dc6d195ce9916897e1ac5e4e6209736c8a`.

Source material: the Judicial College of Victoria's Criminal Charge Book. Benchmark authors: Abdur-Rahman Butler and Umar Butler.

At the inspected revision, the Hugging Face metadata says `cc-by-nc-sa-4.0`, while the README license paragraph links to CC BY-NC 4.0. Both state a noncommercial restriction; the share-alike wording is inconsistent. This package records that discrepancy rather than resolving it. The upstream code repository's MIT license is a separate statement about its code, not permission to relicense the dataset.

Raw corpus text and reference answers are downloaded locally and excluded from the upload ZIP. Included audit tables contain counts, lengths, identifiers, and integrity hashes. Before commercial use or redistribution of source/derived textual content, clarify the applicable terms with the data publisher. Sponsor involvement by itself does not settle the use classification.

References:

- Butler, Abdur-Rahman and Butler, Umar. *Legal RAG Bench: an end-to-end benchmark for legal RAG*. 2026. https://arxiv.org/abs/2603.01710
- Butler, Umar, Butler, Abdur-Rahman, and Malec, Adrian Lucas. *The Massive Legal Embedding Benchmark (MLEB)*. 2025. https://arxiv.org/abs/2510.19365

The dataset card requests citation of both works. This package implements independent team utilities and preserves the prior FinDER metric conventions; it does not copy the authors' evaluation implementation.
