# Legal MultiQueryRetriever and DSPy

Two framework comparisons over the same 4,876 Legal passages:

| Method | Implementation | What changes |
|---|---|---|
| MultiQuery | Native LangChain default prompt; three rewrites plus original; BM25 and RRF60 | Retrieval queries |
| DSPy | Native ChainOfThought + MIPROv2 light; no demonstrations at inference | Reader instructions over fixed BM25 evidence |

[Main findings and charts](../presentation/meeting10_legal_20261117/README.md) · [Team comparison](../presentation/meeting10_team_20261117/README.md)

## Offline reproduction

The published CSVs reproduce the summary scores, primary McNemar test, three paired sign-flip tests, Holm correction and bootstrap intervals. No key, corpus or model download is needed:

```bash
cd presentation/meeting10_legal_20261117
python -m pip install numpy pandas scipy
OPENBLAS_NUM_THREADS=1 python reproduce.py
```

## Full experiment with local checkpoints

From the repository root, create an isolated Python 3.12 environment and install `presentation/meeting10_legal_20261117/requirements.lock.txt`.

Restore the pinned Legal corpus and questions under `data/legal_rag_bench/` and the earlier Legal baseline rankings under `results/person3_openai_hyde/`. Their expected hashes are in the published `source_hashes.json`; prior baseline files are also in the existing Legal raw-results archive. `prepare` verifies all 100 historical BM25 rankings before new calls.

Configure your own key locally in an ignored `.env.local` with permissions `600`; never commit it. `OPENAI_ORG_ID` may be supplied there or through the environment. The proxy caches exact successful requests, records usage before continuing and reserves uncertain attempts. Historical budget assumptions are documented in `api.py` and the frozen protocol.

```bash
python person3_meeting10_legal/run.py prepare
# These three stages can make paid calls:
python person3_meeting10_legal/run.py multiquery
python person3_meeting10_legal/run.py optimize
python person3_meeting10_legal/run.py readers
python person3_meeting10_legal/run.py evaluate
python person3_meeting10_legal/verify.py
python person3_meeting10_legal/report.py
```

The full validator requires retained raw responses and the local API ledger. Those execution caches, credentials and optimizer datasets are excluded from publication. The included audit records describe the completed run; the offline script checks published statistics separately.

`combine_team.py` rebuilds the team table from Legal aggregates and the bundled, pinned LongBench source snapshot. Publication adds a path fallback for that snapshot; the executed Legal runner, framework adapter and API proxy are unchanged.
