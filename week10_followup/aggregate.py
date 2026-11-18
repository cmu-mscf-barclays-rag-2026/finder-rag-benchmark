"""Combine current FinDER, ViDoRe, FiQA and later team retrieval summaries.

Rows retain dataset, corpus, query-set, metric-scope and budget identities.
Only rows with identical identities receive a within-dataset baseline delta.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

from week3_bm25.finder_bm25.data import digest
from week9_evidence.vidore import load_dataset

ROOT = Path(__file__).resolve().parents[1]
FINDER_DIR = ROOT / 'week10_followup/results'
TEAM_DIR = ROOT / 'week10_followup/team_results'
OUTPUT = FINDER_DIR / 'comparison'
TEAM_COMMIT = '18bf882b612b5fa3c0d83a4b861603fd8b424446'
KEVIN_COMMIT = '8df85985ea1ea3f3b7e7508d7390f2970f4dab6c'
TEAM_PRIMARY_PATH = 'presentation/meeting10_team_20261117/team_primary_comparison.csv'
TEAM_ALL_PATH = 'presentation/meeting10_team_20261117/team_all_metrics_long.csv'
KEVIN_DIR = 'vidore_query_expansion/results'
FIELDS = ('dataset', 'corpus', 'split', 'query_set_fingerprint', 'query_count',
          'retrieval_unit', 'method', 'scope', 'k', 'budget', 'hit', 'recall',
          'mrr', 'ndcg', 'mean_delivered_units', 'mean_tokens',
          'latency_ms', 'model_calls', 'input_tokens', 'output_tokens', 'source')


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finder_rows(source_root=FINDER_DIR):
    rows, common = [], None
    for generator in ('gpt', 'qwen'):
        folder = source_root / f'finder_{generator}200'
        path = folder / 'manifest.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        identity = (manifest['bundle_fingerprint'], manifest['test_query_ids_sha256'],
                    manifest['test_count'])
        if common is not None and identity != common:
            raise ValueError('FinDER generators used different corpus or test queries')
        common = identity
        for method in ('single_query', 'multiquery', 'dspy'):
            if method == 'single_query' and generator == 'qwen' and rows:
                continue
            for k in (5, 10):
                result_path = folder / method / f'evaluation_k{k}.json'
                if not result_path.exists():
                    raise ValueError(f'Missing FinDER evaluation: {result_path}')
                result = json.loads(result_path.read_text())
                summary = result['summary']
                if (summary['query_count'] != manifest['test_count']
                    or summary['bundle_fingerprint'] != manifest['bundle_fingerprint']):
                    raise ValueError('FinDER evaluation/manifest mismatch')
                usage = (None if method == 'single_query' else
                         manifest['multiquery_usage' if method == 'multiquery' else 'dspy_test_usage'])
                rows.append({'dataset': 'FinDER', 'corpus': manifest['bundle_fingerprint'],
                             'split': 'test_hash_sample', 'query_set_fingerprint': summary['query_set_fingerprint'],
                             'query_count': summary['query_count'], 'retrieval_unit': 'reference_passage',
                             'method': 'single_query_bm25' if method == 'single_query' else f'{method}_{generator}',
                             'scope': 'ranked', 'k': k, 'budget': '',
                             'hit': sum(r['first_relevant_rank'] is not None for r in result['per_query']) /
                                    len(result['per_query']),
                             'recall': summary['recall'], 'mrr': summary['mrr'],
                             'ndcg': summary['ndcg'], 'mean_delivered_units': summary['retrieved_count'],
                             'mean_tokens': '', 'latency_ms': summary['latency_ms'],
                             'model_calls': '' if usage is None else usage['model_calls'],
                             'input_tokens': '' if usage is None else usage['input_tokens'],
                             'output_tokens': '' if usage is None else usage['output_tokens'],
                             'source': str(result_path.relative_to(ROOT))})
    return rows


def vidore_rows(source_root=FINDER_DIR):
    source_grid = source_root / 'context_grid'
    path = source_grid / 'summary.csv'
    if not path.exists():
        return []
    source = json.loads((source_grid / 'manifest.json').read_text())
    pages, queries, qrels, dataset_manifest = load_dataset(ROOT / '.cache/vidore_v3_finance_en')
    if source['dataset_sha256'] != dataset_manifest['sha256']:
        raise ValueError('ViDoRe grid is from a different dataset')
    query_fp = digest(sorted(queries))
    rows = []
    paths = [path]
    window_path = source_root / 'qwen_window_compare/qwen1024_summary.csv'
    if window_path.exists():
        window_manifest = json.loads((window_path.parent / 'manifest.json').read_text())
        if window_manifest['dataset_revision'] != dataset_manifest['revision'] or window_manifest['queries'] != len(queries):
            raise ValueError('Qwen1024 is from a different ViDoRe dataset or query set')
        paths.append(window_path)
    for summary_path in paths:
        for r in csv.DictReader(summary_path.open()):
            if int(r['k']) not in (10, 20) or int(r['budget']) not in (6000, 12000):
                continue
            if (int(r['before']), int(r['after'])) not in ((0, 0), (1, 0), (0, 1)):
                continue
            policy = f"{r['before']}/{r['after']}"
            rows.append({'dataset': 'ViDoRe_finance_en', 'corpus': dataset_manifest['sha256']['corpus.jsonl'],
                         'split': 'test_english', 'query_set_fingerprint': query_fp,
                         'query_count': int(r['n']), 'retrieval_unit': 'OCR_page',
                         'method': f"{r['method']}_neighbors_{policy}", 'scope': 'delivered',
                         'k': int(r['k']), 'budget': int(r['budget']),
                         'hit': float(r['mean_delivered_hit']),
                         'recall': float(r['mean_delivered_recall']),
                         'mrr': float(r['mean_delivered_mrr']),
                         'ndcg': float(r['mean_delivered_ndcg']),
                         'mean_delivered_units': float(r['mean_delivered_pages']),
                         'mean_tokens': float(r['mean_tokens']), 'latency_ms': '',
                         'model_calls': '', 'input_tokens': '', 'output_tokens': '',
                         'source': str(summary_path.relative_to(ROOT))})
    return rows


def fiqa_rows(source_root=FINDER_DIR):
    path = source_root / 'fiqa_grid/summary.csv'
    if not path.exists():
        return []
    corpus_fp = file_sha(ROOT / '.cache/fiqa/corpus.jsonl')
    from week8_retrieval.beir_eval import load_beir
    _, fiqa_queries, _ = load_beir(ROOT / '.cache/fiqa', 'test')
    query_fp = digest(sorted(fiqa_queries))
    rows = []
    for r in csv.DictReader(path.open()):
        if int(r['k']) not in (10, 20) or int(r['budget']) not in (6000, 12000):
            continue
        rows.append({'dataset': 'FiQA', 'corpus': corpus_fp, 'split': 'test',
                     'query_set_fingerprint': query_fp,
                     'query_count': int(r['n_queries']), 'retrieval_unit': 'answer_passage',
                     'method': 'bm25', 'scope': 'delivered', 'k': int(r['k']),
                     'budget': int(r['budget']),
                     'hit': float(r['delivered_hit']), 'recall': float(r['delivered_recall']),
                     'mrr': float(r['delivered_mrr']), 'ndcg': float(r['delivered_ndcg']),
                     'mean_delivered_units': float(r['returned_docs']),
                     'mean_tokens': float(r['tokens']), 'latency_ms': '',
                     'model_calls': '', 'input_tokens': '', 'output_tokens': '',
                     'source': str(path.relative_to(ROOT))})
    return rows


def team_rows():
    rows = []
    for path in sorted(TEAM_DIR.glob('*.json')):
        data = json.loads(path.read_text())
        if data.get('schema') != 'week10_retrieval_summary_v1' or not isinstance(data.get('rows'), list):
            raise ValueError(f'Invalid team result schema: {path}')
        for row in data['rows']:
            missing = set(FIELDS) - {'source'} - set(row)
            if missing:
                raise ValueError(f'Missing team result fields in {path}: {sorted(missing)}')
            if not all(0 <= float(row[m]) <= 1 for m in ('hit', 'recall', 'mrr', 'ndcg')):
                raise ValueError(f'Team metrics outside [0,1]: {path}')
            rows.append({**row, 'source': str(path.relative_to(ROOT))})
    return rows


def committed_team_rows():
    """Read Legal/LongBench results from their committed Git objects."""
    primary_bytes, all_bytes = git_blob(TEAM_COMMIT, TEAM_PRIMARY_PATH), git_blob(TEAM_COMMIT, TEAM_ALL_PATH)
    primary = list(csv.DictReader(io.StringIO(primary_bytes.decode())))
    all_rows = list(csv.DictReader(io.StringIO(all_bytes.decode())))
    if len(primary) != 6 or len(all_rows) != 43:
        raise ValueError('Unexpected committed Legal/LongBench row counts')
    for row in primary:
        if row['dataset'] not in ('Legal RAG Bench', 'LongBench v2'):
            raise ValueError('Unexpected teammate dataset')
        if not any(all(row[key] == candidate[key] for key in
                       ('dataset', 'method', 'metric', 'score')) for candidate in all_rows):
            raise ValueError('Primary team result absent from complete metrics')
    return primary, all_rows, {TEAM_PRIMARY_PATH: hashlib.sha256(primary_bytes).hexdigest(),
                               TEAM_ALL_PATH: hashlib.sha256(all_bytes).hexdigest()}


def git_blob(commit, path):
    result = subprocess.run(['git', 'show', f'{commit}:{path}'],
                            cwd=ROOT, capture_output=True, check=True)
    return result.stdout


def committed_vidore_query_rows():
    """Keep Kevin's optimizer-held-out 247-English-question study separate."""
    paths = {name: f'{KEVIN_DIR}/{name}' for name in
             ('summary.csv', 'splits.csv', 'independent_audit.json', 'selection.json')}
    blobs = {name: git_blob(KEVIN_COMMIT, path) for name, path in paths.items()}
    audit = json.loads(blobs['independent_audit.json'])
    if audit['status'] != 'passed' or audit['test_english_queries'] != 247:
        raise ValueError('Kevin ViDoRe independent audit is not complete')
    splits = list(csv.DictReader(io.StringIO(blobs['splits.csv'].decode())))
    test_ids = sorted(row['query_id'] for row in splits
                      if row['language'] == 'english' and row['split'] == 'test')
    if len(test_ids) != 247 or len(set(test_ids)) != 247:
        raise ValueError('Kevin ViDoRe test split mismatch')
    source_rows = list(csv.DictReader(io.StringIO(blobs['summary.csv'].decode())))
    source_rows = [r for r in source_rows if r['scope'] == 'split_language' and
                   r['group'] == 'test/english']
    if len(source_rows) != 64:
        raise ValueError('Expected 16 methods at four cutoffs for Kevin ViDoRe')
    baselines = {}
    for row in source_rows:
        if int(row['queries']) != 247 or int(row['k']) not in (1, 5, 10, 20):
            raise ValueError('Kevin ViDoRe query count or cutoff mismatch')
        backend, method = row['method'].split('/')
        if backend not in ('bm25', 'ocr', 'image', 'combined') or method not in (
                'original', 'multiquery', 'dspy_initial', 'dspy_optimized'):
            raise ValueError('Unexpected Kevin ViDoRe method')
        if method == 'original':
            baselines[backend, int(row['k'])] = row
    if len(baselines) != 16:
        raise ValueError('Kevin ViDoRe baseline grid incomplete')
    rows = []
    for row in source_rows:
        backend, method = row['method'].split('/')
        baseline = baselines[backend, int(row['k'])]
        if any(not 0 <= float(row[key]) <= 1 for key in ('hit', 'recall', 'ndcg')):
            raise ValueError('Kevin ViDoRe retrieval metric outside [0,1]')
        rows.append({'backend': backend, 'method': method, 'k': int(row['k']),
                     'queries': int(row['queries']), 'hit': row['hit'],
                     'recall': row['recall'], 'mrr': row['mrr'], 'ndcg': row['ndcg'],
                     'delta_hit': float(row['hit']) - float(baseline['hit']),
                     'delta_recall': float(row['recall']) - float(baseline['recall']),
                     'delta_ndcg': float(row['ndcg']) - float(baseline['ndcg'])})
    selection = json.loads(blobs['selection.json'])
    if selection['baseline'] != {'k1': 0.8, 'b': 0.75, 'ocr_weight': 0.25}:
        raise ValueError('Kevin ViDoRe baseline settings differ from checked report')
    return rows, {'commit': KEVIN_COMMIT,
                  'test_query_ids_sha256': digest(test_ids),
                  'blob_sha256': {paths[name]: hashlib.sha256(data).hexdigest()
                                  for name, data in blobs.items()}}


def with_baseline_deltas(rows):
    identity_keys = ('dataset', 'corpus', 'split', 'query_set_fingerprint',
                     'query_count', 'scope', 'k', 'budget')
    def identity(row):
        return tuple(row[key] for key in identity_keys)

    baselines = {}
    for row in rows:
        baseline = {'FinDER': 'single_query_bm25', 'ViDoRe_finance_en': 'bm25_neighbors_0/0',
                    'FiQA': 'bm25'}.get(row['dataset'])
        if row['method'] == baseline:
            baselines[identity(row)] = row
    result = []
    for row in rows:
        base = baselines.get(identity(row))
        result.append({**row, **{f'delta_{metric}': float(row[metric]) - float(base[metric])
                                if base is not None else ''
                                for metric in ('hit', 'recall', 'mrr', 'ndcg')}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results', type=Path, default=FINDER_DIR)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    source_root = args.source_results.resolve()
    rows = with_baseline_deltas(finder_rows(source_root) + vidore_rows(source_root)
                                + fiqa_rows(source_root) + team_rows())
    primary_team, all_team, team_hashes = committed_team_rows()
    kevin_rows, kevin_source = committed_vidore_query_rows()
    if not rows:
        raise ValueError('No completed results to aggregate')
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'comparison.csv'
    with csv_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[*FIELDS, 'delta_hit', 'delta_recall',
                                                 'delta_mrr', 'delta_ndcg'])
        writer.writeheader()
        writer.writerows(rows)
    stage_path = args.output / 'team_method_comparison.csv'
    with stage_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_team[0]))
        writer.writeheader()
        writer.writerows(all_team)
    kevin_path = args.output / 'team_vidore_query_expansion.csv'
    with kevin_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(kevin_rows[0]))
        writer.writeheader()
        writer.writerows(kevin_rows)
    lines = ['# Cross-dataset retrieval comparison', '',
             'Each row keeps its corpus, test-query set, retrieval unit, cutoff and context budget. '
             'Deltas compare only within an identical setting; ViDoRe deltas use BM25 OCR as a '
             'descriptive reference and mix retriever/window differences. Raw scores across FinDER, '
             'FiQA and ViDoRe do not share relevance definitions.', '',
             '| Dataset | Method | K | Budget | Queries | Hit | Recall | nDCG | Δ Hit |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        delta = row['delta_hit']
        lines.append(f"| {row['dataset']} | {row['method']} | {row['k']} | {row['budget'] or '—'} | "
                     f"{row['query_count']} | {float(row['hit']):.3f} | {float(row['recall']):.3f} | "
                     f"{float(row['ndcg']):.3f} | {float(delta):+.3f} |" if delta != '' else
                     f"| {row['dataset']} | {row['method']} | {row['k']} | {row['budget'] or '—'} | "
                     f"{row['query_count']} | {float(row['hit']):.3f} | {float(row['recall']):.3f} | "
                     f"{float(row['ndcg']):.3f} | — |")
    lines.extend(['', '## Completed teammate methods', '',
                  'The committed Legal and LongBench studies change different pipeline stages and '
                  'score different outcomes. Their full 43 long-form metric rows are preserved in '
                  '`team_method_comparison.csv`. FinDER fuses branch Top-20 lists once and scores '
                  'their Top-5/Top-10 prefixes; the Legal study truncates each branch at each K '
                  'before fusion and generates three alternatives, versus two on FinDER. '
                  'Legal DSPy optimizes the answer reader; FinDER DSPy optimizes queries. '
                  'LongBench measures multiple-choice answer accuracy, while '
                  'FinDER and ViDoRe rows here measure retrieval.', '',
                  '| Dataset | Method | Stage changed | Metric | Score | Δ from matched baseline |',
                  '|---|---|---|---|---:|---:|'])
    for row in primary_team:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['stage_changed']} | "
                     f"{row['metric']} | {float(row['score']):.3f} | "
                     f"{float(row['delta']):+.3f} |")
    lines.extend(['', '## New teammate ViDoRe query-expansion study', '',
                  'Kevin’s committed study uses 247 optimizer-held-out English questions from '
                  'the same 2,942-page ViDoRe corpus. It uses a different query subset, tuned '
                  'BM25 settings, Qwen3-VL-Embedding-2B OCR/image backends, and GPT-4o mini '
                  'query generation. Deltas below are paired within each backend; they are not '
                  'paired with Florence’s 309-question context grid. DSPy uses MIPROv2 query '
                  'instruction optimization, unlike Florence’s FinDER BootstrapFewShot program. '
                  'The full 64 English test rows at K=1/5/10/20 are in '
                  '`team_vidore_query_expansion.csv`.', '',
                  '| Backend | Method | K | Hit | Recall | nDCG | Δ Hit | Δ nDCG |',
                  '|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in kevin_rows:
        if row['k'] != 10:
            continue
        lines.append(f"| {row['backend']} | {row['method']} | 10 | "
                     f"{float(row['hit']):.3f} | {float(row['recall']):.3f} | "
                     f"{float(row['ndcg']):.3f} | {row['delta_hit']:+.3f} | "
                     f"{row['delta_ndcg']:+.3f} |")
    lines.extend(['', f'Team contribution files loaded: {len(list(TEAM_DIR.glob("*.json")))}.',
                  'Additional teammates’ results can be added under `week10_followup/team_results/` '
                  'with the schema in its README, then this script can be rerun.',
                  f'Legal/LongBench source commit: `{TEAM_COMMIT}`. '
                  f'ViDoRe query-expansion source commit: `{KEVIN_COMMIT}`.'])
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps({
        'rows': len(rows), 'team_files': [str(p.relative_to(ROOT)) for p in sorted(TEAM_DIR.glob('*.json'))],
        'input_sha256': {p: file_sha(ROOT / p) for p in sorted({r['source'] for r in rows})},
        'committed_team_source': {'commit': TEAM_COMMIT, 'blob_sha256': team_hashes},
        'committed_vidore_source': kevin_source,
        'output_sha256': {csv_path.name: file_sha(csv_path), stage_path.name: file_sha(stage_path),
                          kevin_path.name: file_sha(kevin_path)}},
        indent=2) + '\n')
    print(f'Aggregated {len(rows)} rows -> {args.output}')


if __name__ == '__main__':
    main()
