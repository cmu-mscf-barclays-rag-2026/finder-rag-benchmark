# ViDoRe Finance–EN: text and page-image retrieval

This experiment compares BM25 on OCR text, dense retrieval on OCR text, dense retrieval on page images, and equal-weight reciprocal rank fusion of the two dense rankings. Every method searches the same 2,942 pages from six bank 10-Ks for all 1,854 query variants. Results are reported for the 309 English questions and the five translated versions separately.

Open `ViDoRe_Finance.ipynb` for the walkthrough and saved outputs. `FEASIBILITY.md` summarizes the benchmark assessment. The run manifest records the exact dataset and model revisions, timing protocol, and input limits.

## Reproduce

Use Python 3.12 with a CUDA-enabled PyTorch installation and matching torchvision. The recorded run used PyTorch 2.7.0 and torchvision 0.22.0 with CUDA 12.8. Run the commands from this folder.

```console
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python run_benchmark.py download
python run_benchmark.py validate
python run_benchmark.py encode --smoke
python run_benchmark.py encode
python run_benchmark.py evaluate
python audit_results.py
python build_evidence_examples.py
```

The public data and model download into `.cache/`. The downloads are about 1.3 GB and 4.3 GB respectively. Completed embedding batches are checkpointed. Run `encode` again to resume with the same batch sizes, device, and runtime for comparable timings. The program accepts `--data`, `--model`, and `--cache` for assets stored elsewhere. The benchmark uses no OpenAI API key.

To match the recorded search runtime, set `OPENBLAS_NUM_THREADS=4` before running `evaluate`. To regenerate the saved notebook, brief, and archive after scoring, install `nbformat` and `IPython` and run `python make_report.py`.

## Evaluation

- **Page retrieval.** Report Hit, Recall, Precision, MRR and graded nDCG at ranks 1, 5, and 10. Grades 1 and 2 count as relevant. Graded nDCG uses linear gains. The benchmark also reports all-judged-page inclusion.
- **Complete-answer-page hit.** For questions with a publisher-labeled grade-2 page, report whether at least one such page appears in the top k. Grade-1 pages contribute partial evidence and may collectively support a complete answer.
- **Worked evidence examples.** `results/evidence_examples.json` records checked OCR anchors with page offsets, the publisher's query/page regions, and coverage from the actual rankings. These demonstrate a page-to-fact mapping for two queries.
- **Timing.** The manifest separates corpus encoding and matrix assembly. Query timing separates single-query GPU encoding from CPU ranking. Fusion totals add the shared encoding time once, both search times, and the fusion time.

Sources: [ViDoRe Finance–EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en), [ViDoRe V3 paper](https://aclanthology.org/2026.acl-long.755/), [MTEB task](https://docs.mteb.org/overview/available_tasks/retrieval/), and [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).

The run reads the original ViDoRe release to retain page IDs and bounding boxes. Scores follow the protocol recorded in the run manifest. `results/qrels.json` reformats published relevance judgments by the ViDoRe V3 authors under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The ZIP does not redistribute the page images, corpus OCR files, or model weights.
