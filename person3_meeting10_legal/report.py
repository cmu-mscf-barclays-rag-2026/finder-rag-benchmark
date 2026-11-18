"""Publish the independently validated Meeting 10 Legal results without API calls.

Only saved metrics, a read-only API ledger and explicit public-file allowlists
are used. Raw prompts, answers, optimizer datasets and the ledger stay local.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/person3_meeting10_legal/v1'
REPORT = OUT / 'report'
DOC = ROOT / 'presentation/meeting10_legal_20261117'
EXPORT = ROOT.parent / 'outputs/meeting10_20261117/legal'
KS = [1, 3, 5, 8, 10, 15, 20, 25]
METHODS = ['bm25_reader', 'multiquery_reader', 'dspy_unoptimized', 'dspy_optimized']
LABELS = dict(bm25_reader='BM25 + fixed reader', multiquery_reader='MultiQuery + fixed reader',
              dspy_unoptimized='BM25 + DSPy CoT, unoptimized',
              dspy_optimized='BM25 + DSPy CoT, optimized')
GITHUB = 'https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark'


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv(path):
    return pd.read_csv(path, dtype={'query_id': str}, float_precision='round_trip')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(value):
    return json.loads(value) if isinstance(value, str) else value


def ci_text(value):
    low, high = interval(value)
    return f'[{low:+.4f}, {high:+.4f}]'


def costs(protocol, split, framework):
    """Map all100 MQR requests to exact question prompts; export aggregates only."""
    summary = read(OUT / 'api/cost_summary.json')
    qa = [json.loads(line) for line in (ROOT / 'data/legal_rag_bench/qa.jsonl').read_text().splitlines() if line.strip()]
    prompt_to_id = {framework['multiquery_default_prompt'].format(question=q['question']): str(q['id']) for q in qa}
    assert len(prompt_to_id) == 100
    connection = sqlite3.connect(f'file:{OUT / "api/api_checkpoint.sqlite"}?mode=ro', uri=True)
    try:
        calls = connection.execute('SELECT key,state,cost,latency_seconds,request_json,response_json,created_stage FROM calls').fetchall()
        cursor = connection.execute('SELECT * FROM events ORDER BY event_id')
        columns = [column[0] for column in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor]
    finally:
        connection.close()
    assert calls and all(row[1] == 'saved' and row[2] is not None for row in calls), 'Report requires all API attempts resolved'
    assert abs(sum(row[2] for row in calls) - summary['known_actual_cost_usd']) < 1e-12
    assert len(calls) == summary['unique_attempts']
    mapped = {}
    models = Counter()
    finishes = Counter()
    for key, state, cost, latency, request_text, response_text, stage in calls:
        body, response = json.loads(request_text), json.loads(response_text)
        models[response['model']] += 1
        finishes[response['choices'][0]['finish_reason']] += 1
        users = [m['content'] for m in body['messages'] if m['role'] == 'user']
        if len(users) == 1 and isinstance(users[0], str) and users[0] in prompt_to_id:
            qid = prompt_to_id[users[0]]
            assert qid not in mapped and stage == 'multiquery_generation'
            mapped[qid] = dict(key=key, cost=float(cost), latency=float(latency))
    assert len(mapped) == 100 and set(mapped) == set(split['dev_ids']) | set(split['test_ids'])
    test_generation = sum(mapped[qid]['cost'] for qid in split['test_ids'])
    dev_generation = sum(mapped[qid]['cost'] for qid in split['dev_ids'])
    all_generation = sum(row['cost'] for row in mapped.values())
    stages = summary['by_stage']
    assert abs(all_generation - stages['multiquery_generation']['actual_new_spend_usd']) < 1e-12
    rows, latency_rows = [], []
    for method in METHODS:
        stage_events = [e for e in events if e['stage'] == method and e['http_status'] == 200]
        assert stage_events, method
        reader_cost = sum(e['retained_upstream_cost_usd'] for e in stage_events)
        assert abs(reader_cost - stages[method]['logical_retained_cost_usd']) < 1e-12
        generation = test_generation if method == 'multiquery_reader' else 0.
        fresh = reader_cost + generation
        rows.append(dict(method=method, test_n=80, reader_logical_fresh_usd=reader_cost,
             query_generation_test80_usd=generation, modeled_fresh_test_inference_usd=fresh,
             test_inference_usd_per1000=fresh / 80 * 1000,
             reader_new_call_spend_usd=stages[method]['actual_new_spend_usd'],
             reader_unique_new_calls=stages[method]['actual_unique_attempts'],
             successful_reader_http_requests=len(stage_events)))
        fresh_latencies = np.array([e['retained_upstream_latency_seconds'] for e in stage_events], dtype=float)
        current_latencies = np.array([e['request_elapsed_seconds'] for e in stage_events], dtype=float)
        latency_rows.append(dict(stage=method, successful_http_requests=len(stage_events),
             cache_hits=sum(bool(e['cache_hit']) for e in stage_events),
             retained_upstream_p50_seconds=float(np.median(fresh_latencies)),
             retained_upstream_total_seconds=float(fresh_latencies.sum()),
             current_request_p50_seconds=float(np.median(current_latencies)),
             current_request_total_seconds=float(current_latencies.sum())))
    generation_latency = np.array([mapped[qid]['latency'] for qid in split['test_ids']])
    latency_rows.append(dict(stage='multiquery_generation_test80', successful_http_requests=80,
         cache_hits=None, retained_upstream_p50_seconds=float(np.median(generation_latency)),
         retained_upstream_total_seconds=float(generation_latency.sum()),
         current_request_p50_seconds=None, current_request_total_seconds=None))
    detail = dict(model_requested=protocol['model'], upstream_model_counts=dict(models),
         matched_query_generation_requests=100, exact_prompt_matching=True,
         multiquery_generation_all100_usd=all_generation,
         multiquery_generation_test80_usd=test_generation,
         multiquery_generation_development20_usd=dev_generation,
         multiquery_generation_test80_usd_per1000=test_generation / 80 * 1000,
         dspy_optimizer_new_call_setup_usd=stages['dspy_optimizer_setup']['actual_new_spend_usd'],
         dspy_optimizer_unique_calls=stages['dspy_optimizer_setup']['actual_unique_attempts'],
         dspy_optimizer_logical_retained_setup_usd=stages['dspy_optimizer_setup']['logical_retained_cost_usd'],
         total_new_call_estimate_usd=summary['known_actual_cost_usd'],
         unique_upstream_calls=len(calls), finish_reason_counts=dict(finishes),
         cost_basis='Saved usage at fixed prices; cached-input discount ignored. Not a billing statement.',
         method_cost_basis='Modeled fresh test80 query generation plus reader calls, including exact-cache reuse at retained source cost; optimization/dev generation excluded.',
         latency_basis='Saved API request durations and current proxy request wall time only; not measured end-to-end method latency.')
    result = pd.DataFrame(rows)
    result.to_csv(REPORT / 'cost_by_method.csv', index=False)
    pd.DataFrame(latency_rows).to_csv(REPORT / 'api_latency_by_stage.csv', index=False)
    dump(REPORT / 'cost_breakdown.json', detail)
    dump(REPORT / 'api_summary.json', summary)
    return result.set_index('method'), pd.DataFrame(latency_rows), detail


def team_table(retrieval, readers, tests, primary, cost, breakdown):
    """Preserve metric meanings so other datasets can append their own rows."""
    base = readers.loc['bm25_reader']
    rows = []
    changes = dict(bm25_reader='baseline', multiquery_reader='retrieval',
                   dspy_unoptimized='reader_prompt_and_CoT', dspy_optimized='reader_instruction_optimization_and_CoT')
    for method in METHODS:
        for metric in ['reference_token_f1_lexical_proxy', 'normalized_answer_exact_match']:
            field = 'token_f1' if metric.startswith('reference_token') else 'em'
            low, high = None, None
            if field == 'token_f1':
                matched = tests[(tests.candidate == method) & (tests.control == 'bm25_reader')]
                if len(matched):
                    low, high = interval(matched.iloc[0].paired_bootstrap95)
            rows.append(dict(dataset='Legal RAG Bench', method=LABELS[method], stage_changed=changes[method],
                 n=80, metric=metric, score=float(readers.loc[method, field]),
                 baseline=float(base[field]), delta=float(readers.loc[method, field] - base[field]),
                 ci_low=low, ci_high=high,
                 test_inference_usd_per1000=float(cost.loc[method, 'test_inference_usd_per1000']),
                 setup_usd=breakdown['dspy_optimizer_new_call_setup_usd'] if method == 'dspy_optimized' else 0.,
                 notes='Top 8 original passages;2400 document words. Cost includes test-only MQR generation where applicable; optimizer/dev generation excluded. Lexical answer proxy, not MC/legal accuracy. CI, when present, is an unadjusted marginal95% paired-delta interval, not simultaneous. Previously inspected benchmark.'))
        route = 'multiquery' if method == 'multiquery_reader' else 'bm25'
        for k in KS:
            row = retrieval.loc[(route, k)]
            baseline = float(retrieval.loc[('bm25', k), 'hit'])
            low, high = primary['paired_bootstrap95'] if method == 'multiquery_reader' and k == 5 else (None, None)
            rows.append(dict(dataset='Legal RAG Bench', method=LABELS[method], stage_changed=changes[method],
                 n=80, metric=f'passage_hit_at_{k}', score=float(row['hit']), baseline=baseline,
                 delta=float(row['hit'] - baseline), ci_low=low, ci_high=high,
                 test_inference_usd_per1000=breakdown['multiquery_generation_test80_usd_per1000'] if method == 'multiquery_reader' else 0.,
                 setup_usd=0., notes='Retrieval-only cost. DSPy rows reuse identical BM25 ranks. MQR re-truncates each branch at each k; these are not prefixes of one ranking. CI at primary Hit@5 is an unadjusted marginal95% paired-delta interval, not simultaneous.'))
    pd.DataFrame(rows).to_csv(REPORT / 'team_results_long.csv', index=False)


def charts(retrieval, readers):
    os.environ.setdefault('MPLCONFIGDIR', str(REPORT / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, label in zip(axes, ['hit', 'ndcg'], ['Passage Hit@k', 'nDCG@k']):
        for method, name, color in [('bm25', 'BM25', '#788593'), ('multiquery', 'MultiQuery + BM25', '#137b8a')]:
            vals = [retrieval.loc[(method, k), metric] for k in KS]
            ax.plot(KS, vals, marker='o', markersize=4, label=name, color=color)
        ax.set(xlabel='Per-route depth and final k', ylabel=label, xticks=KS)
        ax.set_ylim(0, 1 if metric == 'hit' else max(.5, float(retrieval.ndcg.max()) * 1.25))
        ax.grid(alpha=.15)
        if metric == 'hit':
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Legal retrieval: same 80 test questions and 4,876 passages')
    fig.tight_layout(rect=[0, .12, 1, .94])
    fig.text(.5, .025, 'Each k re-truncates the query branches before RRF; points are separate configurations, not nested ranking prefixes.', ha='center', fontsize=8)
    fig.savefig(REPORT / 'retrieval_depth.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    values = [float(readers.loc[m, 'token_f1']) for m in METHODS]
    short = ['BM25 + fixed reader', 'MultiQuery + fixed reader', 'BM25 + DSPy CoT, unoptimized', 'BM25 + DSPy CoT, optimized']
    bars = ax.barh(short, values, color=['#788593', '#137b8a', '#83aec6', '#c58b38'])
    ax.invert_yaxis()
    ax.set(xlabel='Reference-answer token-F1 (lexical proxy)', xlim=(0, max(.5, max(values) * 1.25)))
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=10)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    fig.suptitle('Legal answering: same Top 8 / 2,400-word evidence allowance')
    fig.subplots_adjust(left=.37, right=.97, top=.85, bottom=.20)
    fig.text(.02, .045, '80 test questions; gpt-4o-mini. DSPy changes the reader, while reusing BM25 evidence.\nToken-F1 measures lexical overlap; it is not validated legal correctness. Paired tests and intervals are in the report.', fontsize=8.5)
    fig.savefig(REPORT / 'reader_f1.png', dpi=180)
    plt.close(fig)


def write_report(protocol, split, validation, retrieval, readers, tests, primary, cost, latency, breakdown):
    delta, p = primary['mean_delta'], primary['p_exact_mcnemar']
    r0, r1 = retrieval.loc[('bm25', 5), 'hit'], retrieval.loc[('multiquery', 5), 'hit']
    direction = 'higher' if delta > 0 else 'lower' if delta < 0 else 'unchanged'
    result = f'MultiQuery Hit@5 is {r1:.1%}, versus {r0:.1%} for BM25 ({delta*100:+.1f} percentage points; exact paired McNemar p={p:.5f}).'
    if p >= .05:
        result += ' The primary comparison does not establish a difference at p<.05.'
    else:
        result += f' It is {direction} in this exploratory primary comparison.'
    versions = protocol['versions']
    rows = ['# Legal MultiQueryRetriever and DSPy', '', 'Meeting 10 · 17 November 2026 · Cheryl', '',
        '**Primary retrieval result:** ' + result, '',
        'This study separates changing retrieval queries from changing the answer program. All 100 questions were previously inspected; the test split isolates only this optimization round.', '',
        '## Dataset and fixed comparison', '',
        '- Legal RAG Bench: 100 questions, 4,876 original title+text passages. Exactly one labeled relevant passage per question; 95 distinct labeled passages.',
        '- Fixed seed 42 split by labeled passage: 10 optimizer-training questions, 10 optimizer-validation questions, 80 test questions. No labeled passage group crosses these splits.',
        f'- Model: `gpt-4o-mini`; returned snapshots: {", ".join("`"+m+"`" for m in breakdown["upstream_model_counts"])}. Reader temperature 0, output cap 512 tokens.',
        f'- Framework versions: LangChain Classic `{versions["langchain-classic"]}`, DSPy `{versions["dspy"]}`, LangChain OpenAI `{versions["langchain-openai"]}`.',
        '- Same full corpus and BM25 algebra as the prior work: k1=1.2,b=.75,binary query counts,stable original-corpus-index ties. All 100 saved baseline Top100 rankings replay exactly.', '',
        '## Methods', '',
        '**MultiQueryRetriever:** The native LangChain default prompt requests three alternative questions; `include_original=True` retains the original question. No prompt tuning or custom rewrite cleanup is added. Each branch retrieves 25 original passages. The native unique union is retained, then an explicit equal-weight RRF adaptation (constant 60) ranks a common evidence allowance.', '',
        'For each reported k, every branch is first truncated to k and RRF returns k passages. Thus the Top 5/Top 8/Top 25 runs are separate depth configurations, not prefixes of one master ranking; their Hit values need not be monotonic.', '',
        '**DSPy:** A native ChainOfThought answer program reads the same original-question BM25 Top 8 passages. MIPROv2 `auto="light"` optimizes instructions on the 10+10 development questions. It may internally bootstrap development traces to propose instructions, but the final test program contains zero labeled or bootstrapped demonstrations. The unoptimized CoT control separates instruction optimization from merely changing to a CoT reader. DSPy does not change retrieval in this experiment.', '',
        'Every reader receives at most 8 original passages and 2,400 document words in ranking order; the final passage is prefix-truncated and delivery then stops. The allowance excludes question/instruction/label overhead. Unlike the teammate LongBench experiment, the Legal passages are not rechunked into 300-word chunks.', '',
        '## Retrieval on all 80 test questions', '',
        '| k | BM25 Hit@k | MultiQuery Hit@k | BM25 nDCG@k | MultiQuery nDCG@k |',
        '|---:|---:|---:|---:|---:|']
    for k in KS:
        a, b = retrieval.loc[('bm25', k)], retrieval.loc[('multiquery', k)]
        rows.append(f'| {k} | {a.hit:.1%} | {b.hit:.1%} | {a.ndcg:.4f} | {b.ndcg:.4f} |')
    rows += ['', '![Retrieval depth configurations](retrieval_depth.png)', '',
        f'Primary paired Hit@5 difference: **{delta:+.4f}**, unadjusted marginal 95% paired question-bootstrap interval **{ci_text(primary["paired_bootstrap95"])}** (not simultaneous); wins/losses/ties={primary["wins"]}/{primary["losses"]}/{primary["ties"]}. Exact McNemar p={p:.5f}. Other depths are descriptive. Each question has one labeled passage, so labeled Recall@k equals Hit@k.', '',
        '| Reader method | Retrieval route | Hit@5 | Hit@8 |', '|---|---|---:|---:|']
    for method in METHODS:
        route = 'multiquery' if method == 'multiquery_reader' else 'bm25'
        rows.append(f'| {LABELS[method]} | {"MultiQuery+BM25" if route=="multiquery" else "Same BM25 ranks"} | {retrieval.loc[(route,5),"hit"]:.1%} | {retrieval.loc[(route,8),"hit"]:.1%} |')
    rows += ['', '## Answering on the same 80 test questions', '',
        '| Method | Token-F1 | Normalized exact match | n |', '|---|---:|---:|---:|']
    for method in METHODS:
        v = readers.loc[method]
        rows.append(f'| {LABELS[method]} | {v.token_f1:.4f} | {v.em:.1%} | {int(v.n)} |')
    rows += ['', '![Reader token F1](reader_f1.png)', '',
        '| Paired token-F1 comparison | Mean difference | Unadjusted marginal 95% paired interval | Raw p | Holm3 p |', '|---|---:|---|---:|---:|']
    for row in tests.itertuples(index=False):
        rows.append(f'| {LABELS[row.candidate]} minus {LABELS[row.control]} | {row.mean_delta:+.4f} | {ci_text(row.paired_bootstrap95)} | {row.p_raw:.5f} | {row.p_holm3:.5f} |')
    assert not (tests.p_holm3 < .05).any(), 'Update the report interpretation if independently validated reader conclusions change'
    rows += ['', '**None of the three reader comparisons passes Holm3 correction.** MultiQuery has higher mean token-F1, but this study does not establish a corrected reader improvement. Its marginal interval excludes zero while Holm3 p is above .05 because the displayed intervals are unadjusted individual intervals, not simultaneous intervals for the three comparisons. The interval does not override the multiplicity-adjusted test.', '',
        'For optimized DSPy versus BM25, neither an improvement nor inferiority is established. Optimization also does not establish an improvement over the unoptimized CoT control. These statements concern lexical token-F1, not legal answer accuracy.', '',
        'Tests use 100,000 two-sided paired sign flips and Holm correction across the three listed reader contrasts. Intervals use 20,000 paired question-bootstrap draws; every displayed bootstrap interval is unadjusted/marginal, not simultaneous. They do not account for shared-source dependence or all historical project comparisons.', '',
        '**Token-F1 and exact match are lexical reference-answer proxies, not validated legal correctness or multiple-choice accuracy.** A correct paraphrase may score poorly and overlapping language may score well. No LLM judge score is presented as ground truth.', '',
        '## Evidence actually delivered', '',
        '| Method | Mean full passages | Mean partial passages | Mean words | Gold passage full | Gold passage touched |', '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        v = readers.loc[method]
        rows.append(f'| {LABELS[method]} | {v.full_passages:.2f} | {v.partial_passage:.2f} | {v.used_words:.1f} | {v.gold_full:.1%} | {v.gold_touched:.1%} |')
    truncation = validation.get('checks', {}).get('api', {}).get('reader_truncated_outputs', [])
    rows += ['', '“Touched” means at least a prefix of the labeled passage was delivered; this does not establish that its answer-bearing evidence was included.', '',
        f'Reader outputs ending with a length limit: **{len(truncation)}**; these remain in the scores and are identified in the independent validation record.', '',
        '## API cost and latency', '',
        '| Method | Test reader USD | Test query-generation USD | Modeled fresh USD/1,000 test queries | New reader-call USD |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        v = cost.loc[method]
        rows.append(f'| {LABELS[method]} | {v.reader_logical_fresh_usd:.6f} | {v.query_generation_test80_usd:.6f} | {v.test_inference_usd_per1000:.4f} | {v.reader_new_call_spend_usd:.6f} |')
    rows += ['', f'MultiQuery generation across all 100 questions costs **${breakdown["multiquery_generation_all100_usd"]:.7f}**: test 80 **${breakdown["multiquery_generation_test80_usd"]:.7f}**, development 20 **${breakdown["multiquery_generation_development20_usd"]:.7f}**. These are matched by exact native query prompt, not prorated. The per 1,000 method cost includes test query generation plus reader calls; development generation is excluded.', '',
        f'DSPy one-time optimization new-call estimate: **${breakdown["dspy_optimizer_new_call_setup_usd"]:.8f}** from **{breakdown["dspy_optimizer_unique_calls"]}** unique calls. This is excluded from per-query test inference. Total new-call estimate for this study: **${breakdown["total_new_call_estimate_usd"]:.7f}**, from **{breakdown["unique_upstream_calls"]}** unique saved upstream calls; finish reasons: `{breakdown["finish_reason_counts"]}`, with no unresolved reservations. Estimates use saved token usage and fixed prices, ignoring cached-input discounts; they are not billing statements.', '',
        'Modeled fresh costs count every successful logical reader HTTP request at its retained upstream cost, including requests served from the local exact-payload cache. The separate new-call column records what this execution newly requested. DSPy parser recovery, if any, is included.', '',
        '| API stage | Successful requests | Retained upstream p50 seconds | Current request p50 seconds |', '|---|---:|---:|---:|']
    for row in latency.itertuples(index=False):
        current = '—' if pd.isna(row.current_request_p50_seconds) else f'{row.current_request_p50_seconds:.4f}'
        label = LABELS.get(row.stage, 'MultiQuery generation, test80')
        rows.append(f'| {label} | {int(row.successful_http_requests)} | {row.retained_upstream_p50_seconds:.4f} | {current} |')
    rows += ['', 'Retained latency is the source API-call duration reused by exact cache hits. Current request time includes proxy/cache handling. Neither column is end-to-end RAG latency; generation and reader medians must not be summed into a claimed measured method latency.', '',
        '## Reproduce and merge', '',
        'Run from the repository root with the isolated environment. The first three lines rebuild and verify saved results without paid calls:', '',
        '```bash', 'OPENBLAS_NUM_THREADS=1 ../work/meeting10_venv/bin/python person3_meeting10_legal/run.py evaluate',
        'OPENBLAS_NUM_THREADS=1 ../work/meeting10_venv/bin/python person3_meeting10_legal/verify.py',
        'OPENBLAS_NUM_THREADS=1 ../work/meeting10_venv/bin/python person3_meeting10_legal/report.py', '```', '',
        'The full experiment stages are `prepare`, `multiquery`, `optimize`, `readers`, then `evaluate`. Paid stages use the durable local budget proxy and reuse exact successful requests. Full replay requires the pinned original dataset and retained local checkpoints; the public package contains metrics/protocols rather than prompts, raw reader answers or optimizer development datasets.', '',
        '[Source code](../../person3_meeting10_legal/) · [Frozen protocol](protocol.json) · [Exact split](selection.json) · [Independent validation](independent_validation.json) · [Dependency lock](requirements.lock.txt) · [API summary](api_summary.json) · [Cost matching](cost_breakdown.json)', '',
        '[Cross-dataset long-form rows](team_results_long.csv) retain dataset, stage changed, n, metric, score, matched baseline, delta, paired interval, cost and setup. Only Legal results are filled. Legal lexical F1 must remain distinct from the teammate LongBench multiple-choice accuracy; no teammate numbers are inferred or pooled.', '',
        'Limitations: historical benchmark exposure; 80 evaluation questions; source dependence unadjusted; one labeled passage per question may miss other useful passages; fixed lexical answer metrics; page/answer correctness not manually validated.']
    (REPORT / 'README.md').write_text('\n'.join(rows) + '\n')


def publish_safe_files():
    originals = ['protocol.json', 'selection.json', 'framework_metadata.json', 'independent_validation.json',
        'primary_retrieval_test.json', 'baseline_replay.json', 'source_hashes.json', 'implementation_hashes.json',
        'freeze_receipt.json', 'requirements.lock.txt', 'retrieval_per_query.csv', 'retrieval_summary.csv',
        'reader_per_query.csv', 'reader_summary.csv', 'reader_paired_tests.csv']
    generated = ['README.md', 'cost_by_method.csv', 'api_latency_by_stage.csv', 'cost_breakdown.json',
                 'api_summary.json', 'team_results_long.csv', 'retrieval_depth.png', 'reader_f1.png']
    sources = {name: OUT / name for name in originals} | {name: REPORT / name for name in generated}
    # Never traverse arbitrary result folders or copy databases/raw answer files.
    assert len(sources) == len(originals) + len(generated)
    for name, path in sources.items():
        assert path.is_file() and not path.is_symlink(), name
        if path.suffix in ['.json', '.csv', '.md', '.txt']:
            assert not re.search(rb'(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})', path.read_bytes()), f'Credential-like token found: {name}'
    for destination in [DOC, EXPORT]:
        destination.mkdir(parents=True, exist_ok=True)
        for name, path in sources.items():
            shutil.copy2(path, destination / name)
        if destination == EXPORT:
            readme = destination / 'README.md'
            text = readme.read_text()
            original = '[Source code](../../person3_meeting10_legal/)'
            assert text.count(original) == 1
            target = os.path.relpath(ROOT / 'person3_meeting10_legal', destination)
            readme.write_text(text.replace(original, f'[Source code]({target}/)'))
        manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), independent_validation_passed=True,
             files={name: dict(sha256=sha(destination / name), bytes=(destination / name).stat().st_size) for name in sources},
             raw_ledger_or_optimizer_datasets_exported=False,
             source_code_directory='person3_meeting10_legal',
             report_generator_sha256=sha(Path(__file__)), chart_layout_requires_visual_review=True)
        review_path = destination / 'VISUAL_REVIEW.json'
        if review_path.is_file():
            review = read(review_path)
            if review.get('passed') is True and all(
                    (destination / name).is_file() and sha(destination / name) == digest
                    for name, digest in review.get('figures', {}).items()) and set(
                    review.get('figures', {})) == {'retrieval_depth.png', 'reader_f1.png'}:
                manifest.update(chart_layout_requires_visual_review=False, visual_review='VISUAL_REVIEW.json')
                manifest['files']['VISUAL_REVIEW.json'] = dict(sha256=sha(review_path), bytes=review_path.stat().st_size)
        dump(destination / 'PUBLICATION_MANIFEST.json', manifest)
        for name, item in manifest['files'].items():
            assert sha(destination / name) == item['sha256'], name


def main():
    validation = read(OUT / 'independent_validation.json')
    assert validation.get('passed') is True and validation.get('status') == 'complete', 'Finish evaluation and independent validation before reporting'
    protocol, split, framework = [read(OUT / name) for name in ['protocol.json', 'selection.json', 'framework_metadata.json']]
    assert len(split['test_ids']) == 80
    REPORT.mkdir(parents=True, exist_ok=True)
    retrieval = csv(OUT / 'retrieval_summary.csv')
    retrieval = retrieval[retrieval.cohort == 'test80'].set_index(['method', 'k'])
    assert len(retrieval) == 16 and (retrieval.n == 80).all()
    readers = csv(OUT / 'reader_summary.csv').set_index('method')
    assert set(readers.index) == set(METHODS) and (readers.n == 80).all()
    tests, primary = csv(OUT / 'reader_paired_tests.csv'), read(OUT / 'primary_retrieval_test.json')
    assert len(tests) == 3
    cost, latency, breakdown = costs(protocol, split, framework)
    team_table(retrieval, readers, tests, primary, cost, breakdown)
    charts(retrieval, readers)
    write_report(protocol, split, validation, retrieval, readers, tests, primary, cost, latency, breakdown)
    publish_safe_files()
    print('Published validated local report:', DOC)
    print('Review figures before external publication. No API calls were made.')


if __name__ == '__main__':
    main()
