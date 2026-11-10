# Meeting 9: prompt length x constraint preservation

This dense-only exploratory experiment uses the historical frozen Legal20 and FiQA20 query subsets. It is not a fresh holdout. Four common-base prompts vary requested length (30–50 / 70–100 words) and one explicit constraint-preservation sentence. All other generation and retrieval settings remain fixed. No source passages, answers, or labels enter generation inputs.

This runner never calls an API. The shared workflow handles credentials, read-only model access checks, a cumulative budget guard, pre-request reservations, latency, durable checkpoints, and no automatic retry of unresolved calls.

From the repository root:

```bash
.venv/bin/python person3_meeting9_ablation/run.py prepare
# Shared workflow consumes results/.../factorial20_v1/pending_calls.jsonl.
# It writes successful_api_outputs.json (a list preserving call_id, text, model_returned, status, usage, and latency_seconds).
.venv/bin/python person3_meeting9_ablation/run.py evaluate
.venv/bin/python person3_meeting9_ablation/run.py validate
```

Evaluation refuses incomplete generations. It reuses the full 4,876-document Legal and 57,638-document FiQA corpus vector caches, verifies question-embedding/ranking compatibility, encodes only the 160 new passages, and saves top100 rankings. Hit, Recall, MRR, and nDCG are calculated at 5/10/15/20/25. Legal Hit@5 and FiQA nDCG@10 are the predeclared primary metrics; one Holm family covers 12 exact paired comparisons (four factorial edges plus two averaged main effects, mixed50 only, across two datasets); pure HyDE is secondary descriptive.

Long and short requests have the same 160-token output allowance. Word-count noncompliance and length truncation are retained and reported, not silently regenerated. This tests requested length, not a perfectly enforced word-count intervention. Historical full experiments remain unchanged.
