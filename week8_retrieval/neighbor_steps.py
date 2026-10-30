"""Measure first and second directional sibling increments at frozen Top-10."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from week5_hierarchy.hierarchy_experiment import ROOT, load_data
from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from week8_retrieval.run import BUDGETS, SOURCE, evaluate_one, get_rankings, write_csv

POLICIES = ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2),
            (1, 1), (1, 2), (2, 1), (2, 2))
STEPS = (((0, 0), (1, 0)), ((0, 0), (0, 1)),
         ((1, 0), (2, 0)), ((0, 1), (0, 2)),
         ((1, 1), (1, 2)), ((1, 1), (2, 1)),
         ((1, 2), (2, 2)), ((2, 1), (2, 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'week8_retrieval/results')
    args = parser.parse_args()
    if not (args.output / 'test_summary.csv').exists():
        parser.error('Run the main Week 8 grid first')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    children, parents, queries, _ = load_data(ROOT / 'week5_hierarchy/kevin_hierarchy')
    test = [q for q in queries if q['split'] == 'test']
    tokenizer, checksum = evaluation_tokenizer(manifest['config'], ROOT / '.cache/models', True)
    if checksum != manifest['measurement_tokenizer_sha256']:
        raise ValueError('Measurement tokenizer changed')
    ranks = get_rankings('test', test, children)
    rows = []
    for q in test:
        for budget in BUDGETS:
            for before, after in POLICIES:
                metrics = evaluate_one(q, ranks[q['id']], children, parents, tokenizer,
                    10, budget, before, after, whole_chunks=True)
                rows.append({'query_id': q['id'], 'budget': budget, 'before': before, 'after': after,
                    'hit': metrics['hit'], 'coverage': metrics['coverage'], 'tokens': metrics['tokens']})
    write_csv(args.output / 'neighbor_policy_per_query.csv', rows)
    idx = {(r['query_id'], r['budget'], r['before'], r['after']): r for r in rows}
    summary = []
    for budget in BUDGETS:
        for source, target in STEPS:
            pairs = [(idx[q['id'], budget, *source], idx[q['id'], budget, *target]) for q in test]
            summary.append({'budget': budget, 'from_before': source[0], 'from_after': source[1],
                'to_before': target[0], 'to_after': target[1],
                'hit_wins': sum(b['hit'] > a['hit'] for a, b in pairs),
                'hit_losses': sum(b['hit'] < a['hit'] for a, b in pairs),
                'mean_coverage_gain': statistics.mean(b['coverage']-a['coverage'] for a,b in pairs),
                'mean_token_cost': statistics.mean(b['tokens']-a['tokens'] for a,b in pairs)})
    write_csv(args.output / 'neighbor_steps.csv', summary)
    print(f'Wrote {args.output / "neighbor_steps.csv"}', flush=True)


if __name__ == '__main__':
    main()
