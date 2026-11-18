# Meeting 10 follow-up and FinDER query expansion

Start with the [meeting brief](MEETING_BRIEF.md) for the measured findings,
cross-dataset interpretation, and teammate handoff status.

The [presentation table](../outputs/meeting10_presentation_20261118/combined_results.md)
and [editable workbook](../outputs/meeting10_presentation_20261118/combined_results.xlsx)
combine 41 selected rows from the current Florence, Kevin, Legal, and
LongBench results, including Graph RAG from Yuchen's branch. Percentage-point
changes use a matched control within each study and setting.

`context_grid.py` re-scores the saved Week 9 ViDoRe rankings at K = 5, 8,
10, 20; 6k/10k/12k context tokens; and 0/0, 1/0, 0/1, 1/1 physical-page
expansion. It reports actual delivered seed and neighboring pages, skipped
pages, unused budget and paired relevance changes. See the
[measured findings](results/context_grid/findings.md). `pilot_grid.py` applies
the draft ten-question fact rubric to the same delivered-page traces. The
[FiQA grid](results/fiqa_grid/findings.md) repeats K and budget settings on
shorter financial answer passages, without sibling expansion. A separate
[neighbor-overlap diagnostic](results/redundancy/findings.md) checks how often
added ViDoRe pages are near-duplicates of delivered seeds under a short-window
embedding proxy. The [controlled Qwen input-window result](results/qwen_window_compare/findings.md)
compares 256 and 1,024 document tokens with identical saved query vectors.

`finder_query_expansion.py` compares a single BM25 query with
[LangChain MultiQueryRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever)
and a [DSPy](https://dspy.ai/current/api/optimizers/BootstrapFewShot/)
rewrite program. It uses FinDER's verified 5,830 pooled reference passages,
the fixed Week 3 dev/test split, and the BM25 settings previously selected
on dev (k1=1.6, b=1.0). Each generated method requests two alternatives,
includes the original query, searches the same index to depth 20 per query,
and applies equal-weight RRF before reporting Top-5 and Top-10. DSPy's
BootstrapFewShot optimizer sees only development questions and scores
retrieval Recall@5. Its test questions are hash-selected from the fixed
test partition; the manifest records their fingerprint. Retrieval labels
are exact reference membership, not answer accuracy.
The [paired FinDER findings](results/finder_comparison/findings.md) include
both generators, retrieval metrics, model calls, token use and latency.

The query generators are the pinned local `Qwen/Qwen3-0.6B` checkpoint and
`gpt-4o-mini`. The local Qwen checkpoint uses non-thinking generation and
the same sampling settings in the LangChain and DSPy paths. GPT calls use
the OpenAI API through the `gpt-4o-mini` alias, which is not a frozen model
snapshot. The `.env` file is git-ignored; set `OPENAI_API_KEY` in the
environment or in `week10_followup/.env` to run GPT. Do not commit keys.
The local Qwen runs shown here use float32 on MPS, temperature 0.7 and a
160-token generation cap; the GPT calls use the same temperature and cap.

## Reproduce

From the repository root, with the Week 9 environment and cached dataset:

```sh
.venv-week6/bin/python -m pip install -r week10_followup/requirements.txt
.venv-week6/bin/hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --cache-dir .cache/models
.venv-week6/bin/python -m week10_followup.context_grid --output week10_followup/results_new/context_grid
.venv-week6/bin/python -m week10_followup.pilot_grid --grid week10_followup/results_new/context_grid/per_query.csv --output week10_followup/results_new/pilot_grid
.venv-week6/bin/python -m week10_followup.fiqa_grid --output week10_followup/results_new/fiqa_grid
.venv-week6/bin/python -m week10_followup.redundancy --grid week10_followup/results_new/context_grid/per_query.csv --output week10_followup/results_new/redundancy
.venv-week6/bin/python -m week9_evidence.run --dataset .cache/vidore_v3_finance_en --method qwen --qwen-window 1024 --device mps --output week10_followup/results_new/qwen1024 --ks 5 8 10 20 --budgets 6000 10000 12000
.venv-week6/bin/python -m week9_evidence.check_qwen --dataset .cache/vidore_v3_finance_en --vectors week10_followup/results_new/qwen1024/vectors.npz --page-window 1024 --output week10_followup/results_new/qwen1024_cpu_check.json
.venv-week6/bin/python -m week10_followup.qwen_window_compare --long week10_followup/results_new/qwen1024 --cpu-check week10_followup/results_new/qwen1024_cpu_check.json --output week10_followup/results_new/qwen_window_compare
.venv-week6/bin/python -m week10_followup.pilot_grid --grid week10_followup/results_new/qwen_window_compare/qwen1024_per_query.csv --output week10_followup/results_new/pilot_qwen1024
.venv-week6/bin/python -m week10_followup.finder_query_expansion --generator gpt --test-limit 200 --dev-limit 16 --output week10_followup/results_new/finder_gpt200
.venv-week6/bin/python -m week10_followup.finder_query_expansion --generator qwen --device mps --test-limit 200 --dev-limit 16 --output week10_followup/results_new/finder_qwen200
.venv-week6/bin/python -m week10_followup.report_finder --source week10_followup/results_new --output week10_followup/results_new/finder_comparison
.venv-week6/bin/python -m week10_followup.aggregate --source-results week10_followup/results_new --output week10_followup/results_new/comparison
.venv-week6/bin/python -m unittest discover -s week10_followup/tests -v
```

The aggregation command reads two teammates' committed Git objects. On a
clone that lacks those branch objects, first fetch `feature/person3-hybrid`
and `kevinzhang` from `origin`.

The `--device cpu` alternative works when MPS is unavailable. Outputs must
be written to new or empty folders. Generation calls are sequential, so
latency includes query generation, all retrieval calls and fusion; index
construction is excluded. Token counts are from the generation models and
do not include embedding or document indexing. API charges depend on the
account's rates; the manifest records actual tokens, not an estimated price.

The [comparison aggregator](aggregate.py) includes the current FinDER,
ViDoRe and FiQA results. It also reads the committed Legal and LongBench
method comparison from Git commit `18bf882` and Kevin's 247-English-question
ViDoRe query-expansion study from commit `8df8598`, without merging another
branch. The latter is preserved in a separate 64-row CSV because its test
subset and retrieval backends differ from the 309-question context grid.
It can ingest later teammate commits using the
[handoff schema](team_results/README.md). Different datasets have different
retrieval units and relevance definitions. The report retains separate rows
and only computes deltas for matched query sets and settings.
