# Legal RAG: embedding models and input windows

Dense-retrieval comparison of MiniLM and Qwen3-Embedding-0.6B on Legal RAG Bench, using original child passages and merged parents.

## Findings

Development evaluation selected Qwen's 512-token window for both corpora. On 80 held-out questions, Hit@5 increased from 28.75% to 46.25% for children and from 33.75% to 47.50% for parents. Larger Qwen windows did not improve the development selection objective. This does not isolate long context as the cause of Qwen's advantage over MiniLM.

## Start here

Open `Qwen_Window_Experiment.ipynb` for the presentation, or read `RESULTS.md` for results and limitations. Saved outputs are included; displaying them does not require rerunning GPU embeddings.

Install dependencies in a separate environment:

```bash
pip install -r requirements.txt
python run_experiment.py audit
python run_experiment.py report
python -m unittest discover -s tests -v
```

Run these commands from this folder. See `GPU_SETUP.md` for GPU setup. The notebook's run flags are off by default. `python run_gpu.py` checks completed runs and resumes missing work; it does not recompute already completed runs. Use a separate clean experiment copy without saved results for a fresh benchmark.

## Contents

- `experiment.py`, `_hierarchy/`: experiment and evidence-mapping implementation.
- `config.json`: fixed model revisions, windows and evaluation settings.
- `data/`, `assets/`: prepared corpora, queries and fixed context-budget tokenizer.
- `results/comparison.csv`: consolidated development/test metrics and timings.
- `results/selection.json`: frozen development-based configuration selection.
- `results/paired_examples.json`: observed window-comparison examples.
- `results/final_audit.json`: saved independent metric audit.
- `results/run_records.zip`: all 84 detailed GPU records from 10 development and four test runs, compressed into one file.
- `compact_results.py`, `run_experiment.py`: restore archived records and run the original implementation without changing its saved experiment signature.
- `tests/`: deterministic metric and data checks.

The notebook and command-line wrapper automatically restore detailed records when needed. Expanded copies are ignored by Git. Old CPU experiment results, model weights and embedding caches are not included.

## Evaluation scope

The split is 20 development / 80 test questions. Selection uses development evidence coverage@5, then Hit@5, then the shorter window. Returned text is capped at 2,048 MiniLM tokens for every model. An embedding input window and a returned-context budget are different controls. Labels provide one supporting passage per question, so these scores do not establish multi-evidence completeness or generated-answer correctness.

Keep this directory beside the existing hierarchy folder in the repository. See `DATA_LICENSE.md` before reusing the supplied data.
