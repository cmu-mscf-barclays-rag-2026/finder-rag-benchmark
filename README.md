# FinDER local LangChain RAG baseline

A small retrieval-augmented generation project built around the
[FinDER financial reasoning dataset](https://huggingface.co/datasets/Linq-AI-Research/FinDER).
It uses a fully local, zero-API-cost stack:

- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
- LangChain's in-memory vector store and MMR retrieval
- Ollama with the local `llama3.2` 3B model for answer generation
- Streamlit for the chat interface

No OpenAI API key, API account, or API credits are required. The initial model
and dataset downloads need internet access; inference runs on your computer.

## Included files

- `app.py` — Streamlit chat interface with chat history and source excerpts.
- `rag.py` — shared FinDER loading, chunking, retrieval, and local answer chain.
- `finder_rag_baseline.ipynb` — step-by-step notebook using the same pipeline.
- `evaluate_dense.py` — Task A dense-retrieval Precision@k/Recall@k sweep.
- `results/task_a_report.md` — measured results and interpretation for Task A.
- `tests/test_rag.py` — offline tests for document preparation and retrieval.

- `graph_rag.py` - interpretable passage graph and dense-seed expansion.
- `legal_data.py` - leakage-safe Legal RAG Bench loaders.
- `evaluate_graph.py` - dense-versus-graph retrieval evaluation.
- `GRAPH_RAG_README.md` - graph design, dataset comparison, metrics, and results.

The benchmark answers are deliberately not indexed. Only FinDER's reference
passages are searchable, avoiding answer leakage into the retrieval context.

## One-time setup on Windows

Python 3.10–3.12 is recommended.

1. Install [Ollama for Windows](https://ollama.com/download/windows), open it,
   and leave it running.
2. Open PowerShell in this project folder and run:

```powershell
ollama pull llama3.2
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements-dev.txt
Copy-Item .env.example .env
```

The Ollama model is about 2 GB. If `ollama` is not recognized immediately after
installation, close and reopen PowerShell or restart the notebook kernel.

## Run the notebook

```powershell
jupyter lab finder_rag_baseline.ipynb
```

Choose the `.venv` Python kernel and use **Run All**. The notebook checks Ollama,
pulls `llama3.2` if needed, downloads a deterministic FinDER sample, creates the
local embedding index, and runs an example grounded question.

On this computer, the verified standalone runtime is installed at
`C:\Users\20122\Documents\Codex\Ollama`. The notebook detects and starts it
automatically. You can also double-click `start_ollama.cmd` in that folder.

## Run the Streamlit chatbot

With the same environment active and Ollama open:

```powershell
streamlit run app.py
```

The sidebar shows whether Ollama and the selected model are ready. The first
question takes longer while the FinDER sample and Hugging Face embedding model
are downloaded and indexed. The index is then cached for the app process.

## Configuration

The defaults in `.env.example` work without secrets:

```dotenv
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_EMBEDDING_DEVICE=cpu
```

Use `HF_EMBEDDING_DEVICE=cuda` only when your PyTorch installation can access an
NVIDIA GPU. For a smaller language model on low-memory computers, first run
`ollama pull llama3.2:1b`, then set `OLLAMA_MODEL=llama3.2:1b`.

## How it works

1. Download FinDER's `train` split and select a shuffled, repeatable sample.
2. Convert each `references` passage into a LangChain document with metadata.
3. Split long references into overlapping chunks.
4. Embed the chunks locally with the MiniLM sentence transformer.
5. Store vectors in memory and retrieve diverse passages with MMR.
6. Ask local Llama 3.2 to answer only from those passages and cite `[S1]`,
   `[S2]`, and so on.

## Troubleshooting

- **Ollama is not reachable:** open the Ollama desktop application, then run
  `ollama list` in a new PowerShell window.
- **Model is not installed:** run `ollama pull llama3.2`.
- **`aiohttp` has no `SocketTimeoutError`:** activate the intended environment,
  run `python -m pip install --upgrade -r requirements-dev.txt`, restart the
  notebook kernel, and run all cells from the top.
- **Windows `WinError 206`:** use a short cache location before starting Jupyter
  or Streamlit: `$env:HF_HOME = "$env:USERPROFILE\hf-cache"`.
- **Slow first run:** model downloads and CPU embedding are expected to take
  time once. Start with the default 300 dataset rows.

This is an intentionally simple baseline. Natural next steps are persistent
vector storage, retrieval evaluation against held-out questions, hybrid search,
reranking, and streaming output. FinDER is licensed CC BY-NC 4.0; review its
dataset card before commercial use.

## Task B: interpretable graph RAG

The app now offers an **Interpretable graph expansion** retrieval strategy. It
starts from dense passage seeds, traverses one hop over explicit structural,
citation, title-term, acronym, and rare-term edges, and shows why each source
was selected together with its dense and graph score contributions.

See [`GRAPH_RAG_README.md`](GRAPH_RAG_README.md) for the approach, FinDER versus
Legal RAG Bench comparison, metric definitions, measured 100-question results,
limitations, and reproduction commands. The full retrieval benchmark is:

```powershell
python evaluate_graph.py --device cpu --output-dir results
```

The first measured graph baseline is deliberately not the default: on Legal
RAG Bench it slightly improves MRR@5 but reduces Hit/Recall@5 from 0.28 to 0.27.
It is retained as an auditable baseline for safer query routing and graph-edge
experiments.

## Task A: dense retrieval evaluation

See [`TASK_A_README.md`](TASK_A_README.md) for the parameters, metric
definitions, held-out evaluation protocol, and final numbers. The full run is:

```powershell
python evaluate_dense.py --device cpu
```

The evaluator uses all 5,703 questions, selects chunking only on the frozen
1,128-query development split, and reports the chosen configuration on 4,575
held-out test questions. It writes machine-readable CSV/JSON files to
`results/`; the detailed interpretation is in
[`results/task_a_report.md`](results/task_a_report.md).
