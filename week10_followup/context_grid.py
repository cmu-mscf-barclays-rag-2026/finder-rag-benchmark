"""Re-score saved ViDoRe rankings with the Meeting 10 context grid.

The ranking is frozen for each method. Only seed depth, physical-neighbor
expansion, and whole-page context budget change. No model inference occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from week8_retrieval.beir_eval import rank_metrics
from week9_evidence.run import TOKENIZER, neighbors, expand
from week9_evidence.vidore import digest, load_dataset, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('bm25', 'minilm', 'qwen256')
KS = (5, 8, 10, 20)
BUDGETS = (6000, 10000, 12000)
POLICIES = ((0, 0), (1, 0), (0, 1), (1, 1))


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pack_trace(candidates, seed_ids, costs, budget, next_seed=None):
    """Pack in candidate order, matching Week 9's whole-page skip-to-fit rule."""
    delivered, skipped, zero_text, tokens = [], [], [], 0
    for cid in dict.fromkeys(candidates):
        cost = costs[cid]
        if cost == 0:
            zero_text.append(cid)
        elif tokens + cost <= budget:
            delivered.append(cid)
            tokens += cost
        else:
            skipped.append(cid)
    seed_set = set(seed_ids)
    delivered_set = set(delivered)
    return {
        'delivered_ids': delivered,
        'candidate_pages': len(set(candidates)),
        'seed_pages_considered': len(seed_ids),
        'neighbor_pages_considered': len(set(candidates) - seed_set),
        'delivered_pages': len(delivered),
        'delivered_seeds': len(delivered_set & seed_set),
        'delivered_neighbors': len(delivered_set - seed_set),
        'excluded_seeds_budget': len(set(skipped) & seed_set),
        'skipped_pages_budget': len(skipped),
        'skipped_neighbors_budget': len(set(skipped) - seed_set),
        'zero_text_pages': len(zero_text),
        'tokens': tokens,
        'unused_tokens': budget - tokens,
        'next_ranked_seed_fits': int(next_seed is not None and next_seed not in candidates
                                     and costs[next_seed] > 0
                                     and costs[next_seed] <= budget - tokens),
    }


def evaluate_method(rankings, pages, qrels, costs, positions):
    rows = []
    for qid in sorted(qrels):
        ranking = rankings[qid]
        for k in KS:
            ranked = rank_metrics(ranking, qrels[qid], k)
            seeds = ranking[:k]
            next_seed = ranking[k] if len(ranking) > k else None
            for before, after in POLICIES:
                candidates = expand(ranking, pages, positions, k, before, after)
                for budget in BUDGETS:
                    trace = pack_trace(candidates, seeds, costs, budget, next_seed)
                    delivered = rank_metrics(trace['delivered_ids'], qrels[qid], trace['delivered_pages'])
                    rows.append({'query_id': qid, 'k': k, 'before': before, 'after': after,
                                 'budget': budget, 'relevant_pages': len(qrels[qid]),
                                 **{f'ranked_{m}': v for m, v in ranked.items()},
                                 **{f'delivered_{m}': v for m, v in delivered.items()},
                                 **{key: val for key, val in trace.items() if key != 'delivered_ids'},
                                 'delivered_ids': json.dumps(trace['delivered_ids'])})
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['method'], row['k'], row['before'], row['after'], row['budget'])
        groups.setdefault(key, []).append(row)
    metrics = ('ranked_hit', 'ranked_recall', 'ranked_mrr', 'ranked_ndcg',
               'delivered_hit', 'delivered_recall', 'delivered_mrr', 'delivered_ndcg',
               'candidate_pages', 'neighbor_pages_considered', 'delivered_pages',
               'delivered_seeds', 'delivered_neighbors', 'excluded_seeds_budget',
               'skipped_pages_budget', 'skipped_neighbors_budget', 'zero_text_pages',
               'tokens', 'unused_tokens', 'next_ranked_seed_fits')
    result = []
    for key, group in sorted(groups.items()):
        counts = sorted(r['delivered_pages'] for r in group)
        result.append({'method': key[0], 'k': key[1], 'before': key[2],
                       'after': key[3], 'budget': key[4], 'n': len(group),
                       **{f'mean_{metric}': statistics.fmean(r[metric] for r in group)
                          for metric in metrics},
                       'median_delivered_pages': statistics.median(counts),
                       'p10_delivered_pages': counts[int((len(counts)-1)*0.1)],
                       'p90_delivered_pages': counts[int((len(counts)-1)*0.9)]})
    return result


def paired(rows):
    by_key = {(r['method'], r['query_id'], r['k'], r['budget'], r['before'], r['after']): r
              for r in rows}
    output = []
    for method in METHODS:
        for k in KS:
            for budget in BUDGETS:
                base = [by_key[(method, qid, k, budget, 0, 0)]
                        for qid in sorted({r['query_id'] for r in rows if r['method'] == method})]
                for before, after in POLICIES[1:]:
                    comp = [by_key[(method, r['query_id'], k, budget, before, after)] for r in base]
                    output.append({'method': method, 'k': k, 'budget': budget,
                                   'before': before, 'after': after, 'n': len(base),
                                   'hit_gained': sum(a['delivered_hit'] == 0 and b['delivered_hit'] == 1
                                                     for a, b in zip(base, comp)),
                                   'hit_lost': sum(a['delivered_hit'] == 1 and b['delivered_hit'] == 0
                                                   for a, b in zip(base, comp)),
                                   'mean_recall_delta': statistics.fmean(
                                       b['delivered_recall'] - a['delivered_recall'] for a, b in zip(base, comp)),
                                   'mean_delivered_pages_delta': statistics.fmean(
                                       b['delivered_pages'] - a['delivered_pages'] for a, b in zip(base, comp)),
                                   'mean_seeds_delta': statistics.fmean(
                                       b['delivered_seeds'] - a['delivered_seeds'] for a, b in zip(base, comp))})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / '.cache/vidore_v3_finance_en')
    parser.add_argument('--source-results', type=Path, default=ROOT / 'week9_evidence/results')
    parser.add_argument('--output', type=Path, default=ROOT / 'week10_followup/results/context_grid')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    pages, queries, qrels, dataset_manifest = load_dataset(args.dataset)
    tokenizer, tokenizer_sha = evaluation_tokenizer({'evaluation_tokenizer': TOKENIZER},
                                                    ROOT / '.cache/models', True)
    costs = {cid: len(tokenizer.encode(p['text']).ids) for cid, p in pages.items()}
    positions = neighbors(pages)
    all_rows, sources = [], {}
    for method in METHODS:
        path = args.source_results / method / 'rankings.jsonl'
        saved = read_jsonl(path)
        rankings = {r['query_id']: r['doc_ids'] for r in saved}
        if len(rankings) != len(saved) or set(rankings) != set(queries):
            raise ValueError(f'Ranking/query mismatch for {method}')
        if any(len(v) < max(KS)+1 or len(v) != len(set(v)) or not set(v) <= pages.keys()
               for v in rankings.values()):
            raise ValueError(f'Invalid ranking for {method}')
        all_rows.extend({'method': method, **r} for r in evaluate_method(
            rankings, pages, qrels, costs, positions))
        sources[method] = digest(path)
    args.output.mkdir(parents=True)
    save_csv(args.output / 'per_query.csv', all_rows)
    save_csv(args.output / 'summary.csv', summarize(all_rows))
    save_csv(args.output / 'paired_neighbors.csv', paired(all_rows))
    (args.output / 'manifest.json').write_text(json.dumps({
        'dataset_revision': dataset_manifest['revision'], 'dataset_sha256': dataset_manifest['sha256'],
        'ranking_sha256': sources, 'tokenizer_sha256': tokenizer_sha,
        'ks': KS, 'budgets': BUDGETS, 'policies_before_after': POLICIES,
        'packing': 'Week 9 whole-page skip-to-fit, physical PDF neighbors, no seed backfill',
        'query_count': len(queries), 'source_code_sha256': digest(Path(__file__)),
        'output_sha256': {p.name: digest(p) for p in sorted(args.output.glob('*.csv'))},
    }, indent=2) + '\n')
    print(f'Wrote {len(all_rows)} query/configuration traces to {args.output}')


if __name__ == '__main__':
    main()
