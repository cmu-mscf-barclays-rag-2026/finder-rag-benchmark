"""Summarize paired FinDER query-expansion results with fixed-seed intervals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics

from week9_evidence.vidore import digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'week10_followup/results'
OUTPUT = SOURCE / 'finder_comparison'
METRICS = ('hit', 'recall', 'mrr', 'ndcg')
GPT_INPUT_PER_M = 0.15
GPT_OUTPUT_PER_M = 0.60


def save_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_method(folder, k):
    evaluation = json.loads((folder / f'evaluation_k{k}.json').read_text())
    summary = evaluation['summary']
    rows = {}
    for row in evaluation['per_query']:
        rows[row['query_id']] = {'hit': int(row['first_relevant_rank'] is not None),
                                 'recall': row['recall'], 'mrr': row['reciprocal_rank'],
                                 'ndcg': row['ndcg'], 'latency_ms': row['latency_ms']}
    if len(rows) != summary['query_count']:
        raise ValueError('Duplicate/missing FinDER query rows')
    return rows, summary


def paired(a, b, label):
    if set(a) != set(b):
        raise ValueError('Paired methods have different questions')
    qids = sorted(a)
    result = {'comparison': label, 'n': len(qids),
              'hit_gained': sum(a[q]['hit'] == 0 and b[q]['hit'] == 1 for q in qids),
              'hit_lost': sum(a[q]['hit'] == 1 and b[q]['hit'] == 0 for q in qids)}
    rng = random.Random(42)
    samples = [[rng.randrange(len(qids)) for _ in qids] for _ in range(10000)]
    for metric in METRICS:
        differences = [b[q][metric] - a[q][metric] for q in qids]
        boot = sorted(statistics.fmean(differences[i] for i in sample) for sample in samples)
        result[f'delta_{metric}'] = statistics.fmean(differences)
        result[f'ci95_{metric}_low'] = boot[249]
        result[f'ci95_{metric}_high'] = boot[9749]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.source = args.source.resolve()
    summaries, pairs, paths, identity = [], [], [], None
    data = {}
    for generator in ('gpt', 'qwen'):
        folder = args.source / f'finder_{generator}200'
        if not (folder / 'manifest.json').exists():
            continue
        manifest = json.loads((folder / 'manifest.json').read_text())
        run_identity = (manifest['bundle_fingerprint'], manifest['test_query_ids_sha256'],
                        manifest['test_count'])
        if identity is not None and identity != run_identity:
            raise ValueError('GPT and Qwen FinDER experiments have different test questions')
        identity = run_identity
        paths.append(folder / 'manifest.json')
        for k in (5, 10):
            base, base_summary = load_method(folder / 'single_query', k)
            data[(generator, 'single_query', k)] = base
            if generator == 'qwen' and ('gpt', 'single_query', k) in data:
                if any(base[q][m] != data[('gpt', 'single_query', k)][q][m]
                       for q in base for m in METRICS):
                    raise ValueError('FinDER baseline differs between generator runs')
            for method in ('single_query', 'multiquery', 'dspy'):
                rows, summary = load_method(folder / method, k)
                data[(generator, method, k)] = rows
                usage = {'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0} if method == 'single_query' else \
                        manifest['multiquery_usage' if method == 'multiquery' else 'dspy_test_usage']
                if method == 'single_query':
                    alternatives, candidates = 0.0, ''
                else:
                    traces = [json.loads(line) for line in (folder / method / 'traces.jsonl').open()]
                    if len(traces) != len(rows) or {t['query_id'] for t in traces} != set(rows):
                        raise ValueError('FinDER query traces do not match evaluation rows')
                    alternatives = statistics.fmean(len(t['generated_queries']) for t in traces)
                    candidates = statistics.fmean(len({cid for search in t['searches']
                                                       for cid in search['doc_ids']}) for t in traces)
                summaries.append({'generator': generator, 'method': method, 'k': k,
                                  'n': len(rows), 'hit': statistics.fmean(r['hit'] for r in rows.values()),
                                  'recall': summary['recall'], 'mrr': summary['mrr'],
                                  'ndcg': summary['ndcg'], 'latency_ms': summary['latency_ms'],
                                  'mean_alternatives': alternatives,
                                  'mean_unique_candidates': candidates,
                                  'estimated_api_usd': ((usage['input_tokens'] * GPT_INPUT_PER_M +
                                                         usage['output_tokens'] * GPT_OUTPUT_PER_M) / 1_000_000
                                                        if generator == 'gpt' else ''),
                                  **usage})
                if method != 'single_query':
                    pairs.append({'generator': generator, 'k': k,
                                  **paired(base, rows, f'{method} minus single_query')})
        print(f'Loaded {generator}: {manifest["test_count"]} test questions')
    if not summaries:
        raise ValueError('No completed FinDER generator runs')
    if all((g, m, 5) in data for g in ('gpt', 'qwen') for m in ('multiquery', 'dspy')):
        for method in ('multiquery', 'dspy'):
            for k in (5, 10):
                pairs.append({'generator': 'gpt_vs_qwen', 'k': k,
                              **paired(data[('qwen', method, k)], data[('gpt', method, k)],
                                       f'{method} gpt minus qwen')})
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / 'summary.csv', summaries)
    save_csv(args.output / 'paired.csv', pairs)
    historical_path = ROOT / 'week3_bm25/results/b_bm25_common/b_bm25_tuned_per_query_k5.csv'
    historical = list(csv.DictReader(historical_path.open()))
    historical_hit = sum(bool(r['first_relevant_rank']) for r in historical) / len(historical)
    lines = ['# FinDER query-expansion findings', '',
             f'Held-out hash-selected test questions: **{identity[2]}** from the fixed 4,575-question '
             'FinDER test partition. All methods search the same 5,830 pooled reference passages '
             'with the tuned BM25 index. The index contains benchmark reference passages, not '
             'full financial filings. DSPy receives 16 development examples and no test labels; '
             'its actual optimization model-call count is in each manifest. '
             f'The historical tuned-BM25 full-test Hit@5 is {historical_hit:.3f} over '
             f'{len(historical)} questions, versus {next(r["hit"] for r in summaries if r["method"] == "single_query" and r["k"] == 5):.3f} '
             'on this 200-question sample, so sample scores should not be extrapolated to the full test set.', '',
             '| Generator | Method | K | Hit | Recall | MRR | nDCG | Alternatives | Unique candidates | Mean latency ms | Model calls | Input / output tokens | API USD estimate |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        candidate_text = '—' if row['mean_unique_candidates'] == '' else f"{row['mean_unique_candidates']:.1f}"
        cost_text = 'local' if row['estimated_api_usd'] == '' else f"${row['estimated_api_usd']:.5f}"
        lines.append(f"| {row['generator']} | {row['method']} | {row['k']} | "
                     f"{row['hit']:.3f} | {row['recall']:.3f} | {row['mrr']:.3f} | "
                     f"{row['ndcg']:.3f} | {row['mean_alternatives']:.2f} | "
                     f"{candidate_text} | {row['latency_ms']:.1f} | {row['model_calls']} | "
                     f"{row['input_tokens']} / {row['output_tokens']} | {cost_text} |")
    lines.extend(['', 'Intervals are 10,000 fixed-seed paired question-bootstrap draws, reported '
                  'as descriptive marginal 95% intervals. A single run per generator was made.'])
    for k in (5, 10):
        lines.extend(['', f'## Paired Top-{k} changes', '',
                      '| Comparison | Δ Hit | 95% interval | Wins / losses | Δ Recall | Δ nDCG |',
                      '|---|---:|---|---:|---:|---:|'])
        for row in pairs:
            if row['k'] != k:
                continue
            lines.append(f"| {row['generator']} {row['comparison']} | {row['delta_hit']:+.3f} | "
                         f"[{row['ci95_hit_low']:+.3f}, {row['ci95_hit_high']:+.3f}] | "
                         f"{row['hit_gained']} / {row['hit_lost']} | {row['delta_recall']:+.3f} | "
                         f"{row['delta_ndcg']:+.3f} |")
    if {r['generator'] for r in summaries} == {'gpt', 'qwen'}:
        lookup = {(r['generator'], r['method'], r['k']): r for r in summaries}
        lines.extend(['', '## Interpretation', '',
                      f"At Top-10, the paired BM25 baseline hits {lookup['gpt', 'single_query', 10]['hit']:.3f} "
                      f"of questions. GPT MultiQuery and DSPy both reach "
                      f"{lookup['gpt', 'multiquery', 10]['hit']:.3f}; Qwen MultiQuery reaches "
                      f"{lookup['qwen', 'multiquery', 10]['hit']:.3f} and Qwen DSPy "
                      f"{lookup['qwen', 'dspy', 10]['hit']:.3f}. The GPT DSPy-versus-Qwen "
                      'DSPy paired Hit interval is above zero on this sample; the GPT-versus-Qwen '
                      'MultiQuery interval crosses zero. GPT DSPy uses more input tokens and '
                      'takes longer per query than GPT MultiQuery, while the local Qwen DSPy '
                      'path is slower than local Qwen MultiQuery. These are single-run pilot '
                      'findings, not full-test or answer-quality estimates.'])
    lines.extend(['', 'Model calls and input/output tokens cover the test query-generation stage; '
                  'Top-5 and Top-10 are scored from the same generated queries, so their displayed '
                  'model calls and costs are not additive. '
                  'The separate DSPy development optimization cost is in each manifest. '
                  'Qwen runs locally; GPT-4o mini API estimates use $0.15/M input and $0.60/M '
                  'output tokens from the [model pricing page](https://developers.openai.com/api/docs/models/gpt-4o-mini), '
                  'without cached-input discounts or taxes. They are not billing records. '
                  'Qwen and GPT token counts come from different tokenizers. '
                  'Local Qwen and remote GPT latencies have different execution paths and are '
                  'descriptive, not a hardware-normalized speed ranking. Retrieval metrics do '
                  'not score generated answer correctness.'])
    (args.output / 'findings.md').write_text('\n'.join(lines) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps({
        'test_query_count': identity[2], 'bundle_fingerprint': identity[0],
        'test_query_ids_sha256': identity[1],
        'historical_full_baseline_sha256': digest(historical_path),
        'source_manifest_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
        'source_sha256': digest(Path(__file__)),
        'bootstrap_seed': 42, 'bootstrap_draws': 10000,
        'output_sha256': {p.name: digest(p) for p in args.output.glob('*.csv')},
    }, indent=2) + '\n')
    print(f'Wrote FinDER comparison -> {args.output}')


if __name__ == '__main__':
    main()
