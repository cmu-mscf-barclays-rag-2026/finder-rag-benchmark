"""Exploratory Qwen256 cosine-overlap diagnostic for delivered neighbor pages.

The page vectors encode only the first 256 Qwen tokens, so high cosine is a
possible redundancy signal, not proof that whole pages repeat the same facts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import numpy as np

from week9_evidence.vidore import digest, load_dataset, read_jsonl
from week10_followup.context_grid import save_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path,
                        default=ROOT / 'week10_followup/results/context_grid/per_query.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'week10_followup/results/redundancy')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    source = ROOT / 'week9_evidence/results/qwen256'
    grid = args.grid
    pages, queries, _, dataset = load_dataset(ROOT / '.cache/vidore_v3_finance_en')
    rankings = {row['query_id']: row['doc_ids'] for row in read_jsonl(source / 'rankings.jsonl')}
    if set(rankings) != set(queries):
        raise ValueError('Qwen rankings and ViDoRe questions differ')
    with np.load(source / 'vectors.npz') as vectors:
        ids = [str(cid) for cid in vectors['page_ids']]
        embeddings = vectors['pages'].copy()
    if set(ids) != set(pages) or embeddings.shape[0] != len(ids):
        raise ValueError('Qwen vectors and ViDoRe pages differ')
    index = {cid: i for i, cid in enumerate(ids)}
    normalized = {cid: re.sub(r'\s+', ' ', page['text']).strip().casefold()
                  for cid, page in pages.items()}
    rows = []
    for row in csv.DictReader(grid.open()):
        if (row['method'] != 'qwen256' or int(row['k']) != 10 or
            (int(row['before']), int(row['after'])) not in ((1, 0), (0, 1), (1, 1)) or
            int(row['budget']) not in (6000, 12000)):
            continue
        delivered = json.loads(row['delivered_ids'])
        seeds = set(rankings[row['query_id']][:10])
        delivered_seeds = [cid for cid in delivered if cid in seeds]
        neighbors = [cid for cid in delivered if cid not in seeds]
        similarities = [float(np.max(embeddings[[index[seed] for seed in delivered_seeds]] @
                                      embeddings[index[cid]]))
                        for cid in neighbors] if delivered_seeds else []
        exact = sum(any(normalized[cid] == normalized[seed] for seed in delivered_seeds)
                    for cid in neighbors)
        rows.append({'query_id': row['query_id'], 'budget': int(row['budget']),
                     'before': int(row['before']), 'after': int(row['after']),
                     'delivered_seeds': len(delivered_seeds),
                     'delivered_neighbors': len(neighbors),
                     'neighbors_compared': len(similarities),
                     'exact_text_duplicates': exact,
                     'cosine_ge_0_8': sum(value >= .8 for value in similarities),
                     'cosine_ge_0_9': sum(value >= .9 for value in similarities),
                     'mean_max_cosine': statistics.fmean(similarities) if similarities else ''})
    if len(rows) != 309 * 3 * 2:
        raise ValueError('Unexpected diagnostic grid size')
    summary = []
    for budget in (6000, 12000):
        for before, after in ((1, 0), (0, 1), (1, 1)):
            group = [row for row in rows if row['budget'] == budget and
                     (row['before'], row['after']) == (before, after)]
            compared = sum(row['neighbors_compared'] for row in group)
            summary.append({'budget': budget, 'before': before, 'after': after,
                            'queries': len(group),
                            'mean_delivered_neighbors': statistics.fmean(
                                row['delivered_neighbors'] for row in group),
                            'neighbors_compared': compared,
                            'exact_text_duplicates': sum(row['exact_text_duplicates'] for row in group),
                            'fraction_cosine_ge_0_8': sum(row['cosine_ge_0_8'] for row in group) / compared,
                            'fraction_cosine_ge_0_9': sum(row['cosine_ge_0_9'] for row in group) / compared,
                            'mean_max_cosine': statistics.fmean(
                                row['mean_max_cosine'] for row in group if row['mean_max_cosine'] != '')})
    args.output.mkdir(parents=True)
    save_csv(args.output / 'per_query.csv', rows)
    save_csv(args.output / 'summary.csv', summary)
    (args.output / 'manifest.json').write_text(json.dumps({
        'dataset_sha256': dataset['sha256'],
        'vectors_sha256': digest(source / 'vectors.npz'),
        'rankings_sha256': digest(source / 'rankings.jsonl'),
        'grid_sha256': digest(grid), 'source_sha256': digest(Path(__file__)),
        'scope': 'Qwen256 Top-10 delivered neighbors versus delivered seeds, cosine over first 256 Qwen tokens',
        'rows': len(rows)}, indent=2) + '\n')
    print(f'Wrote {len(rows)} exploratory overlap traces -> {args.output}')


if __name__ == '__main__':
    main()
