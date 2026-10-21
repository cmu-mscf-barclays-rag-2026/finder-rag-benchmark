# Legal RAG: source-span evaluation pilot

Kevin's October 20–26 workstream: evaluate retrieved evidence independently of parent, child or grandchild boundaries.

## Deliverable

- 20 inherited development questions with 27 proposed supporting spans.
- Exact Unicode offsets, source hashes, verbatim evidence and annotation rationales.
- A shared evaluator for complete evidence inclusion and partial character/token coverage.
- Actual BM25 retrieval on the full parent, child and grandchild corpora, at k=1/5/10 with unlimited and 2,048-token contexts.
- A presentation notebook, results, source-mapping tests and an independent score audit.

**Annotation status:** source offsets are verified, but the evidence selections are assistant-reviewed drafts. No human reviewer has approved them. These are not official additional dataset labels or a finalized benchmark. Review `REVIEW_SHEET.md` before using them as validated gold evidence.

## Showcase

Open **`Span_Evaluation.ipynb`**. It includes saved outputs: motivation, source mapping, annotation example, boundary-invariance controls, actual retrieval results, and remaining review work. `RESULTS.md` provides a short written summary.

## Run locally or in Colab

Use Python 3.10+ and run from this directory:

```bash
pip install -r requirements.txt
python span_bench.py validate
python -m unittest discover -s tests -v
python span_bench.py run
python audit_results.py
```

No GPU, model download or API key is needed. The pinned tokenizer is used for counting and slicing, not for embedding. BM25 uses lowercase Unicode word tokens, positive Robertson IDF, k1=1.5, b=0.75 and stable corpus-order ties. These fixed settings demonstrate the evaluator; they are not a tuned retrieval system. Query text alone enters retrieval.

Grandchildren target 128 WordPiece tokens with 32-token overlap within each original child; boundary slices are re-counted to enforce the cap. All 2,785 reconstructed parents are indexed, not only sources containing the pilot's gold evidence. Canonical source text is compressed in one file and loads automatically.

## Main files

| File | Purpose |
|---|---|
| `Span_Evaluation.ipynb` | Meeting showcase |
| `span_bench.py` | Source loader, chunking, evaluator, BM25 and external scoring CLI |
| `data/annotations.json` | Question/answer, exact spans, source mapping, review status |
| `annotation_selections.json` | Original assistant-selected text anchors and rationales; provenance, not the live scoring input |
| `data/sources.jsonl.gz` | Unchanged canonical reconstructed parents with child mappings |
| `ANNOTATION_PROTOCOL.md`, `REVIEW_SHEET.md` | Evidence-selection rules and per-question review material |
| `METRICS.md`, `TEAM_HANDOFF.md` | Metric definitions and teammate integration |
| `results/` | Actual pilot results, rankings, profiles, controls and verification records |
| `tests/test_spans.py`, `audit_results.py` | Deterministic checks and independent saved-score audit |

## Limitations

This is a small development-only pilot with draft evidence labels, not an 80-question held-out comparison. Multiple spans currently come from one gold child per question; no genuinely multi-document labels have been validated. Offsets refer to reconstructed source text, not the original PDF/HTML. Alternative acceptable evidence is not exhaustively labeled. Smaller chunks and larger parents return different amounts of text, so read budget and returned-token columns alongside top-k scores. No answer-generation quality or new dense-model result is claimed.

Upload the entire `legal_rag_spans` folder beside `legal_rag_hierarchy` and `qwen_embeddings`; do not overwrite those folders. Source-data licensing restrictions remain applicable; see `DATA_LICENSE.md`.
