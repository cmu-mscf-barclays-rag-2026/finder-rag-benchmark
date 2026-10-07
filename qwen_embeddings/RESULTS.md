# MiniLM and Qwen: completed GPU experiment

## Scope and execution

The registered development sweep is complete: MiniLM at 256 tokens and Qwen at 512, 2,048, 4,096 and 8,192 tokens, each on the full child and parent corpora. All ten configurations used the same 20 development questions. Selected windows were frozen before the 80 held-out test questions were evaluated.

The test comparison includes MiniLM, Qwen-512, and the selected Qwen window per corpus (duplicates removed). All active results were measured on the same NVIDIA GeForce RTX 5070 Laptop GPU with PyTorch 2.7.0+cu128. Earlier CPU results are excluded from this package and from the comparison tables.

The 2,048-token returned-body budget and its MiniLM tokenizer are fixed across all configurations. Qwen document windows use the Qwen tokenizer; Qwen query instruction and query window remain fixed. Model changes also change architecture, training, tokenizer, and pooling, so model gains cannot be attributed solely to longer context.

## Selected settings

Selection maximized development gold-body token coverage@5, breaking ties by unit Hit@5 and then shorter windows. Qwen-512 remains a preregistered test anchor.

- Child: Qwen window **512**; dev coverage@5 **40.00%**, unit Hit@5 **40.00%**.
- Parent: Qwen window **512**; dev coverage@5 **30.00%**, unit Hit@5 **40.00%**.

## Held-out results (80 questions)

| Model | Corpus | Window | Unit Hit@5 | Unit Hit@10 | Full gold-child inclusion@5 | Gold-token coverage@5 | Mean query ms |
|---|---|---:|---:|---:|---:|---:|---:|
| minilm | child | 256 | 28.75% | 35.00% | 28.75% | 28.75% | 12.38 |
| minilm | parent | 256 | 33.75% | 46.25% | 25.00% | 26.80% | 14.56 |
| qwen | child | 512 | 46.25% | 57.50% | 45.00% | 46.17% | 89.72 |
| qwen | parent | 512 | 47.50% | 58.75% | 38.75% | 40.43% | 96.93 |

Unit Hit refers to the original child or mapped parent as appropriate. Full gold-child inclusion requires the entire annotated passage to survive context truncation. Gold-token coverage permits partial labelled-text inclusion; neither is answer correctness.

## What changed with a longer Qwen window?

- **Child:** development selected the 512-token anchor, so there is no separately selected longer-window test row. Longer windows were still evaluated on development.
- **Parent:** development selected the 512-token anchor, so there is no separately selected longer-window test row. Longer windows were still evaluated on development.

## Model comparison

- **Child, Qwen-512 versus MiniLM-256:** test unit Hit@5 changed by +17.50 percentage points; gold-token coverage@5 changed by +17.42 points.
- **Parent, Qwen-512 versus MiniLM-256:** test unit Hit@5 changed by +13.75 percentage points; gold-token coverage@5 changed by +13.63 points.

## Development window sweep

| Corpus | Qwen window | Hit@5 | Gold-token coverage@5 | Input truncation rate | Corpus encoding seconds |
|---|---:|---:|---:|---:|---:|
| child | 512 | 40.00% | 40.00% | 10.81% | 156.4 |
| child | 2048 | 40.00% | 40.00% | 0.00% | 163.9 |
| child | 4096 | 40.00% | 40.00% | 0.00% | 165.8 |
| child | 8192 | 40.00% | 40.00% | 0.00% | 164.3 |
| parent | 512 | 40.00% | 30.00% | 29.34% | 94.5 |
| parent | 2048 | 30.00% | 30.00% | 3.91% | 231.4 |
| parent | 4096 | 40.00% | 25.00% | 0.83% | 297.3 |
| parent | 8192 | 40.00% | 25.00% | 0.07% | 338.0 |

## Timing interpretation

Corpus encoding is measured separately from matrix assembly. Query latency includes single-query encoding and exact CPU cosine search, excludes context assembly and file I/O, and uses one warmup plus three repeats. All Qwen windows used corpus batch size 4; MiniLM used 16. Timings are sequential measurements on a laptop, not randomized performance trials; temperature, background activity and power settings may affect them. Resumed batches retain their originally measured encoding times. Model loading and downloading are excluded.

## Validation and limitations

- Eight unit/integration tests passed. An independent audit checked 1,560 query/k metric rows, all run signatures, split membership, timing arithmetic and context-budget bounds.
- Both models passed GPU smoke and batch-padding consistency checks. These checks are separate from the benchmark results above.
- Only 20 questions were available for development and 80 for test. Report point estimates as exploratory rather than statistically established superiority.
- Three gold parents occur in both splits; this is question-held-out evaluation, not unseen-parent generalization.
- Gold labels identify one supporting child per question; body coverage does not establish multi-evidence completeness or answer correctness.
- Optional 16k/32k windows were not preregistered and were not run. The completed scope is 512/2k/4k/8k.
- No LLM generation, HyDE, hybrid fusion or graph retrieval was added to this dense-model comparison.

## Files to present

`Qwen_Window_Experiment.ipynb` displays the design, development sweep, frozen selection, held-out metrics, timing and helped/hurt examples. `results/comparison.csv` contains the full table; `results/paired_examples.json` contains real paired differences; `results/final_audit.json` records verification. The run_records.zip archive preserves predictions, per-query metrics, lengths, raw timing and manifests; the notebook restores it automatically.
