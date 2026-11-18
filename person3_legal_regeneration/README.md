# Legal fresh-H replication

One fresh generation for each of the original 100 Legal questions. The original model, exact prompt, temperature, output cap, corpus and BM25 settings are retained. This is a repeatability study, not a term-ablation experiment.

[Mechanisms, HyDE comparison and p-value tables](../presentation/legal_qh_replication_20261118/README.md) · [Exact frozen prompt and protocol](../presentation/legal_qh_replication_20261118/data/replication_protocol.json)

## Offline table checks

```bash
cd presentation/legal_qh_replication_20261118
python -m pip install numpy pandas scipy
OPENBLAS_NUM_THREADS=1 python reproduce.py
```

This checks all three correction families, their exact tests, published rates and the eight-row comparison table. No API key, corpus or generation cache is needed.

## Full replay with retained local data

Use the isolated dependencies from `presentation/meeting10_legal_20261117/requirements.lock.txt`. Restore the original Legal files, saved sparse rankings and generation records at the paths in `data/source_hashes.json` in the report folder. The BM25 helper is in `person3_meeting10_legal/run.py`.

Run from the repository root:

```bash
python person3_legal_regeneration/run.py prepare
# Paid only for missing calls in this replication's own ledger:
python person3_legal_regeneration/run.py generate
python person3_legal_regeneration/run.py evaluate
python person3_legal_regeneration/verify_and_report.py
```

`prepare` requires exact replay of all 300 old rankings before any fresh generation. `generate` uses a dedicated SQLite checkpoint, per-request reservations and no automatic retries. Its recorded budget prior is $4.8181403, fresh-round limit $0.75, and cumulative guard $7; these are this study's historical settings, not an organization billing limit. Existing successful calls from this repeat are reused on resume; original generations are not reused as fresh calls.

Keep credentials in an ignored local `.env.local` with permissions 600. Raw generations, the API ledger and execution caches remain local. Full validation needs those checkpoints; the public offline check verifies published statistics separately.

An export-only missing-import fix was recorded after generation. All inference and numerical function ASTs and numerical result hashes were preserved; details are in the frozen receipt. No paid calls were repeated for that fix.
