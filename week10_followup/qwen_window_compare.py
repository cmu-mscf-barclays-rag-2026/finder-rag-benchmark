"""Controlled ViDoRe Qwen256 versus Qwen1024 document-window comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from week8_retrieval.beir_eval import rank_metrics
from week9_evidence.vidore import digest, load_dataset, read_jsonl
from week9_evidence.run import TOKENIZER, neighbors
from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from week10_followup.context_grid import evaluate_method, summarize, save_csv

ROOT = Path(__file__).resolve().parents[1]


def load_run(path, queries, pages):
    manifest = json.loads((path / 'manifest.json').read_text())
    saved = read_jsonl(path / 'rankings.jsonl')
    rankings = {row['query_id']: row['doc_ids'] for row in saved}
    if len(rankings) != len(saved) or set(rankings) != set(queries):
        raise ValueError('Ranking/query mismatch')
    if any(len(ids) != len(set(ids)) or not set(ids) <= pages.keys() for ids in rankings.values()):
        raise ValueError('Invalid ranking page IDs')
    return rankings, manifest


def compare_rankings(short, long, qrels):
    rows = []
    for k in (1, 3, 5, 8, 10, 20):
        a = {qid: rank_metrics(short[qid], qrels[qid], k) for qid in qrels}
        b = {qid: rank_metrics(long[qid], qrels[qid], k) for qid in qrels}
        rows.append({'k': k, 'n': len(qrels),
                     'hit256': statistics.fmean(x['hit'] for x in a.values()),
                     'hit1024': statistics.fmean(x['hit'] for x in b.values()),
                     'hit_gained': sum(not a[q]['hit'] and b[q]['hit'] for q in qrels),
                     'hit_lost': sum(a[q]['hit'] and not b[q]['hit'] for q in qrels),
                     'recall256': statistics.fmean(x['recall'] for x in a.values()),
                     'recall1024': statistics.fmean(x['recall'] for x in b.values()),
                     'ndcg256': statistics.fmean(x['ndcg'] for x in a.values()),
                     'ndcg1024': statistics.fmean(x['ndcg'] for x in b.values())})
    return rows


def compare_delivered(short_rows, long_rows):
    """Pair the same question and packing settings across document windows."""
    def key(row):
        return (row['query_id'], int(row['k']), int(row['before']),
                int(row['after']), int(row['budget']))
    short = {key(r): r for r in short_rows if r['method'] == 'qwen256'}
    long = {key(r): r for r in long_rows}
    if len(short) != len(long) or set(short) != set(long):
        raise ValueError('Qwen window delivered rows do not pair exactly')
    results = []
    for k in (5, 8, 10, 20):
        for before, after in ((0, 0), (1, 0), (0, 1), (1, 1)):
            for budget in (6000, 10000, 12000):
                ids = [qid for qid in sorted({r[0] for r in short})
                       if (qid, k, before, after, budget) in short]
                a = [short[(qid, k, before, after, budget)] for qid in ids]
                b = [long[(qid, k, before, after, budget)] for qid in ids]
                results.append({'k': k, 'before': before, 'after': after,
                                'budget': budget, 'n': len(ids),
                                'hit_gained': sum(float(x['delivered_hit']) == 0 and
                                                  float(y['delivered_hit']) == 1
                                                  for x, y in zip(a, b)),
                                'hit_lost': sum(float(x['delivered_hit']) == 1 and
                                                float(y['delivered_hit']) == 0
                                                for x, y in zip(a, b)),
                                **{f'mean_delta_{metric}': statistics.fmean(
                                    float(y[f'delivered_{metric}']) - float(x[f'delivered_{metric}'])
                                    for x, y in zip(a, b))
                                   for metric in ('hit', 'recall', 'mrr', 'ndcg')},
                                'mean_delta_pages': statistics.fmean(
                                    int(y['delivered_pages']) - int(x['delivered_pages'])
                                    for x, y in zip(a, b))})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / '.cache/vidore_v3_finance_en')
    parser.add_argument('--short', type=Path, default=ROOT / 'week9_evidence/results/qwen256')
    parser.add_argument('--long', type=Path, default=ROOT / 'week10_followup/results/qwen1024')
    parser.add_argument('--cpu-check', type=Path,
                        default=ROOT / 'week10_followup/results/qwen1024_cpu_check.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'week10_followup/results/qwen_window_compare')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    pages, queries, qrels, dataset = load_dataset(args.dataset)
    short, short_manifest = load_run(args.short, queries, pages)
    long, long_manifest = load_run(args.long, queries, pages)
    sa, la = short_manifest['retrieval'], long_manifest['retrieval']
    for key in ('encoder', 'query_window', 'query_instruction', 'search'):
        if sa[key] != la[key]:
            raise ValueError(f'Qwen window controls differ: {key}')
    if (sa['embedding_window'], la['embedding_window']) != (256, 1024):
        raise ValueError('Expected Qwen256 and Qwen1024 runs')
    if short_manifest['dataset'] != long_manifest['dataset'] or short_manifest['dataset'] != dataset:
        raise ValueError('Dataset revision/checksums differ')
    check = json.loads(args.cpu_check.read_text())
    if (not check['passed'] or check['page_window'] != 1024 or
        check['vectors_sha256'] != digest(args.long / 'vectors.npz')):
        raise ValueError('Qwen1024 CPU validation is missing or does not match saved vectors')
    with np.load(args.short / 'vectors.npz') as sv, np.load(args.long / 'vectors.npz') as lv:
        if (not np.array_equal(sv['query_ids'], lv['query_ids']) or
            not np.array_equal(sv['page_ids'], lv['page_ids']) or
            not np.array_equal(sv['queries'], lv['queries'])):
            raise ValueError('Query embeddings or page/query ID ordering differ')
    rank_rows = compare_rankings(short, long, qrels)
    tokenizer, tokenizer_sha = evaluation_tokenizer({'evaluation_tokenizer': TOKENIZER},
                                                    ROOT / '.cache/models', True)
    if tokenizer_sha != short_manifest['tokenizer_sha256'] or tokenizer_sha != long_manifest['tokenizer_sha256']:
        raise ValueError('Context measurement tokenizer differs')
    costs = {cid: len(tokenizer.encode(p['text']).ids) for cid, p in pages.items()}
    long_rows = [{'method': 'qwen1024', **r} for r in evaluate_method(long, pages, qrels,
                                                                     costs, neighbors(pages))]
    long_summary = summarize(long_rows)
    short_grid = ROOT / 'week10_followup/results/context_grid'
    short_summary = list(csv.DictReader((short_grid / 'summary.csv').open()))
    paired_delivery = compare_delivered(list(csv.DictReader((short_grid / 'per_query.csv').open())),
                                         long_rows)
    delivery = []
    for sr in short_summary:
        if sr['method'] != 'qwen256' or int(sr['k']) not in (10, 20):
            continue
        lr = next(r for r in long_summary if r['k'] == int(sr['k']) and
                  r['budget'] == int(sr['budget']) and r['before'] == int(sr['before']) and
                  r['after'] == int(sr['after']))
        delivery.append({'k': lr['k'], 'budget': lr['budget'], 'before': lr['before'],
                         'after': lr['after'], 'hit256': float(sr['mean_delivered_hit']),
                         'hit1024': lr['mean_delivered_hit'],
                         'recall256': float(sr['mean_delivered_recall']),
                         'recall1024': lr['mean_delivered_recall'],
                         'pages256': float(sr['mean_delivered_pages']),
                         'pages1024': lr['mean_delivered_pages']})
    args.output.mkdir(parents=True)
    save_csv(args.output / 'ranked_comparison.csv', rank_rows)
    save_csv(args.output / 'delivered_comparison.csv', delivery)
    save_csv(args.output / 'paired_delivered.csv', paired_delivery)
    save_csv(args.output / 'qwen1024_summary.csv', long_summary)
    save_csv(args.output / 'qwen1024_per_query.csv', long_rows)
    (args.output / 'manifest.json').write_text(json.dumps({
        'dataset_revision': dataset['revision'], 'queries': len(queries), 'pages': len(pages),
        'short_window': 256, 'long_window': 1024, 'same_query_embeddings_exact': True,
        'short_ranking_sha256': digest(args.short / 'rankings.jsonl'),
        'long_ranking_sha256': digest(args.long / 'rankings.jsonl'),
        'tokenizer_sha256': tokenizer_sha,
        'cpu_check_sha256': digest(args.cpu_check),
        'source_sha256': digest(Path(__file__)),
        'output_sha256': {p.name: digest(p) for p in sorted(args.output.glob('*.csv'))},
    }, indent=2) + '\n')
    print(f'Compared Qwen256 and Qwen1024 on {len(queries)} questions -> {args.output}')


if __name__ == '__main__':
    main()
