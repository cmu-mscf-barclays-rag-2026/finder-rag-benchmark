"""Exploratory draft-fact coverage for the Meeting 10 context grid."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from week9_evidence.evidence import fact_metrics, validate_annotations
from week9_evidence.vidore import digest, load_dataset, read_jsonl
from week10_followup.context_grid import save_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / '.cache/vidore_v3_finance_en')
    parser.add_argument('--annotations', type=Path,
                        default=ROOT / 'week9_evidence/annotations/finance_pilot.jsonl')
    parser.add_argument('--grid', type=Path,
                        default=ROOT / 'week10_followup/results/context_grid/per_query.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'week10_followup/results/pilot_grid')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    pages, queries, qrels, dataset = load_dataset(args.dataset)
    annotations = read_jsonl(args.annotations)
    validate_annotations(annotations, pages, queries, allow_draft=True)
    by_id = {a['query_id']: a for a in annotations}
    rows = []
    for row in csv.DictReader(args.grid.open()):
        qid = row['query_id']
        if qid not in by_id:
            continue
        kept = json.loads(row['delivered_ids'])
        visible = {cid: [(0, len(pages[cid]['text']))] for cid in kept}
        facts = fact_metrics(by_id[qid], visible, int(row['tokens']))
        rows.append({'method': row['method'], 'query_id': qid,
                     'k': int(row['k']), 'before': int(row['before']),
                     'after': int(row['after']), 'budget': int(row['budget']),
                     'fact_recall': facts['fact_recall'], 'sufficient': facts['sufficient'],
                     'covered_fact_count': facts['covered_fact_count'],
                     'fact_count': facts['fact_count'],
                     'facts_per_1k_tokens': facts['facts_per_1k_tokens'],
                     'delivered_hit': int(row['delivered_hit']),
                     'delivered_pages': int(row['delivered_pages']),
                     'tokens': int(row['tokens'])})
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ('method', 'k', 'before', 'after', 'budget'))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(groups.items()):
        summary.append({**dict(zip(('method', 'k', 'before', 'after', 'budget'), key)),
                        'n': len(group),
                        **{f'mean_{metric}': statistics.fmean(r[metric] for r in group)
                           for metric in ('fact_recall', 'sufficient', 'facts_per_1k_tokens',
                                          'delivered_hit', 'delivered_pages', 'tokens')}})
    methods = {row['method'] for row in rows}
    if len(rows) != len(annotations) * len(methods) * 4 * 4 * 3 or any(r['n'] != len(annotations) for r in summary):
        raise ValueError('Missing grid configuration or draft annotation')
    args.output.mkdir(parents=True)
    save_csv(args.output / 'per_question.csv', rows)
    save_csv(args.output / 'summary.csv', summary)
    (args.output / 'manifest.json').write_text(json.dumps({
        'dataset_revision': dataset['revision'], 'annotation_sha256': digest(args.annotations),
        'grid_sha256': digest(args.grid), 'query_count': len(annotations),
        'annotation_status': 'draft_unreviewed', 'source_sha256': digest(Path(__file__)),
        'output_sha256': {p.name: digest(p) for p in args.output.glob('*.csv')},
    }, indent=2) + '\n')
    print(f'Scored {len(rows)} draft-fact grid rows -> {args.output}')


if __name__ == '__main__':
    main()
