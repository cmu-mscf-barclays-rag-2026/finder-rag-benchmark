"""Frozen Qwen depth, directional sibling, and MMR experiment.

Uses Week 6's saved rankings and cached vectors. No model inference is allowed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np

from week5_hierarchy.hierarchy_experiment import ROOT, load_data
from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from hierarchy.handoff import assemble_context
from hierarchy.io import digest, read_jsonl

SOURCE = ROOT / 'week6_qwen_hierarchy/results/qwen'
BUDGETS = (4000, 6000, 8000)
DEPTHS = (10, 15, 20)
POLICIES = ((0, 0), (1, 1), (0, 2), (1, 2), (2, 1), (2, 2))
LAMBDAS = (1.0, 0.7, 0.5)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_rankings(split, queries, children):
    rows = read_jsonl(SOURCE / 'rankings' / f'{split}_qwen_4096_dense.jsonl')
    ranks = {r['query_id']: r['child'] for r in rows}
    if set(ranks) != {q['id'] for q in queries}:
        raise ValueError('Frozen ranking/query mismatch')
    if any(len(r) != 100 or len(set(r)) != 100 or not set(r) <= children.keys() for r in ranks.values()):
        raise ValueError('Invalid frozen Top-100 child ranking')
    return ranks


def cached_vectors(children, queries, config, manifest):
    """Recover the exact Qwen vectors by Week 6 cache key; fail on a cache miss."""
    from transformers import AutoTokenizer
    spec = manifest['encoder']
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'],
        cache_dir=str(ROOT / '.cache/models'), local_files_only=True, padding_side='left')
    tokenizer.truncation_side = 'right'
    cache = ROOT / '.cache/week6_embeddings'
    salt = json.dumps(spec, sort_keys=True)

    def load(texts, window, prompt=''):
        inputs = [prompt + t for t in texts]
        ids = tokenizer(inputs, add_special_tokens=True, truncation=False)['input_ids']
        vectors = []
        for tokens in ids:
            key = hashlib.sha256((salt + json.dumps(tokens[:window])).encode()).hexdigest()
            path = cache / f'{key}.npy'
            if not path.exists():
                raise FileNotFoundError(
                    f'Qwen vector missing from frozen Week 6 cache: {key}. '
                    'See week8_retrieval/README.md for cache reproduction requirements.')
            vector = np.load(path, allow_pickle=False)
            if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                raise ValueError(f'Invalid cached Qwen vector: {key}')
            vectors.append(vector)
        return np.stack(vectors)

    records = sorted(children.values(), key=lambda r: r['id'])
    doc_vectors = load([r['text'] for r in records], 4096)
    prompt = f'Instruct: {config["query_instruction"]}\nQuery: '
    query_vectors = load([q['question'] for q in queries], config['query_window'], prompt)
    return {r['id']: v for r, v in zip(records, doc_vectors)}, {q['id']: v for q, v in zip(queries, query_vectors)}


def mmr_ranking(candidates, vectors, query, lam, depth=20):
    matrix = np.stack([vectors[c] for c in candidates])
    relevance = matrix @ query
    similarity = matrix @ matrix.T
    remaining = list(range(len(candidates)))
    chosen = []
    while remaining and len(chosen) < depth:
        best = max(remaining, key=lambda i: (
            lam * float(relevance[i]) - (1-lam) * max((float(similarity[i, j]) for j in chosen), default=0),
            -i))
        chosen.append(best)
        remaining.remove(best)
    return [candidates[i] for i in chosen]


def sibling_candidates(ranking, children, parents, k, before, after):
    selected = []
    seen = set()
    for seed in ranking[:k]:
        siblings = [s['child_id'] for s in parents[children[seed]['parent_id']]['child_spans']]
        at = siblings.index(seed)
        # Nearest neighbors first, alternating backward/forward when both exist.
        additions = [seed]
        for distance in range(1, max(before, after) + 1):
            if distance <= before and at - distance >= 0:
                additions.append(siblings[at-distance])
            if distance <= after and at + distance < len(siblings):
                additions.append(siblings[at+distance])
        for cid in additions:
            if cid not in seen:
                selected.append(cid)
                seen.add(cid)
    return selected


def sibling_context(ranking, children, parents, tokenizer, k, budget, before, after):
    candidates = sibling_candidates(ranking, children, parents, k, before, after)
    kept, used = [], 0
    for cid in candidates:
        # The measurement tokenizer treats the whitespace separator as a
        # boundary, so chunk token counts are additive. Assert below on the
        # exact assembled text to guard that property.
        cost = len(tokenizer.encode(children[cid]['text']).ids)
        if used + cost <= budget:
            kept.append(cid)
            used += cost
    context = assemble_context(kept, 'child', children, parents, tokenizer, max(1, len(kept)))
    if context['returned_content_tokens'] != used:
        raise ValueError('Tokenizer chunk additivity assumption failed')
    context['seed_ids'] = ranking[:k]
    return context


_TERM_CACHE = {}
_OVERLAP_CACHE = {}


def overlap(ids, children):
    key = tuple(ids)
    if key in _OVERLAP_CACHE:
        return _OVERLAP_CACHE[key]
    for cid in ids:
        if cid not in _TERM_CACHE:
            _TERM_CACHE[cid] = set(re.findall(r'\w+', children[cid]['text'].lower()))
    terms = [_TERM_CACHE[cid] for cid in ids]
    values = [len(a & b) / len(a | b) for i, a in enumerate(terms) for b in terms[i+1:] if a | b]
    result = statistics.mean(values) if values else 0.0
    _OVERLAP_CACHE[key] = result
    return result


def evaluate_one(q, ranking, children, parents, tokenizer, k, budget, before, after, whole_chunks=False):
    if before == after == 0 and not whole_chunks:
        context = assemble_context(ranking, 'child', children, parents, tokenizer, k, budget)
    else:
        context = sibling_context(ranking, children, parents, tokenizer, k, budget, before, after)
    gold = set(q['gold_child_ids'])
    full = gold & set(context['fully_included_child_ids'])
    seen = total = 0
    for cid in gold:
        offsets = tokenizer.encode(children[cid]['text']).offsets
        visible = context['visible_child_characters'].get(cid, 0)
        seen += sum(end <= visible for _, end in offsets)
        total += len(offsets)
    tokens = context['returned_content_tokens']
    if tokens > budget:
        raise AssertionError('Context budget exceeded')
    ids = context['unit_ids']
    return {'hit': int(bool(full)), 'coverage': seen / total if total else 0,
            'tokens': tokens, 'coverage_per_1k': 1000 * seen / total / tokens if total and tokens else 0,
            'returned_chunks': len(ids), 'overlap_jaccard': overlap(ids, children),
            'seed_ids': json.dumps(context['seed_ids']), 'returned_ids': json.dumps(ids),
            'gold_ids_in_context': json.dumps(sorted(full))}


def summarize(rows, split):
    groups = {}
    for row in rows:
        key = tuple(row[x] for x in ('method', 'lambda', 'k', 'budget', 'before', 'after'))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        summary.append({'split': split, **dict(zip(('method', 'lambda', 'k', 'budget', 'before', 'after'), key)),
            'n': len(group), 'hits': sum(r['hit'] for r in group),
            'mean_coverage': statistics.mean(r['coverage'] for r in group),
            'mean_tokens': statistics.mean(r['tokens'] for r in group),
            'mean_coverage_per_1k': statistics.mean(r['coverage_per_1k'] for r in group),
            'mean_chunks': statistics.mean(r['returned_chunks'] for r in group),
            'mean_overlap_jaccard': statistics.mean(r['overlap_jaccard'] for r in group)})
    return sorted(summary, key=lambda r: (r['method'], r['lambda'], r['k'], r['budget'], r['before'], r['after']))


def run_split(queries, rankings, vectors, query_vectors, children, parents, tokenizer, split):
    rows = []
    for q in queries:
        original = rankings[q['id']]
        for lam in LAMBDAS:
            ranking = original if lam == 1 else mmr_ranking(original, vectors, query_vectors[q['id']], lam)
            method = 'dense' if lam == 1 else 'mmr'
            for k in DEPTHS:
                for budget in BUDGETS:
                    for before, after in POLICIES:
                        row = evaluate_one(q, ranking, children, parents, tokenizer, k, budget, before, after)
                        rows.append({'split': split, 'query_id': q['id'], 'method': method,
                            'lambda': lam, 'k': k, 'budget': budget, 'before': before, 'after': after, **row})
    return rows, summarize(rows, split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'week8_retrieval/results')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest.get('reference_only'):
        raise ValueError('Frozen Qwen source is incomplete')
    config = manifest['config']
    children, parents, queries, _ = load_data(ROOT / 'week5_hierarchy/kevin_hierarchy')
    dev, test = [[q for q in queries if q['split'] == s] for s in ('dev', 'test')]
    if (len(dev), len(test)) != (20, 80):
        raise ValueError('Frozen split changed')
    tokenizer, checksum = evaluation_tokenizer(config, ROOT / '.cache/models', True)
    if checksum != manifest['measurement_tokenizer_sha256']:
        raise ValueError('Measurement tokenizer changed')
    args.output.mkdir(parents=True, exist_ok=True)
    print('Loading cached Qwen vectors', flush=True)
    vectors, query_vectors = cached_vectors(children, queries, config, manifest)
    # Verify cosine scores reproduce saved Qwen rankings before evaluating MMR.
    records = sorted(children)
    matrix = np.stack([vectors[cid] for cid in records])
    frozen = {s: get_rankings(s, part, children) for s, part in (('dev', dev), ('test', test))}
    for q in queries:
        order = np.argsort(-(matrix @ query_vectors[q['id']]), kind='stable')[:100]
        actual = [records[i] for i in order]
        expected = frozen[q['split']][q['id']]
        if actual != expected:
            raise ValueError(f'Cached vectors do not reproduce frozen ranking for query {q["id"]}')
    dev_rankings = frozen['dev']
    print('Evaluating development split', flush=True)
    _, dev_summary = run_split(dev, dev_rankings, vectors, query_vectors, children, parents, tokenizer, 'dev')
    write_csv(args.output / 'dev_summary.csv', dev_summary)
    # The grid is fixed in code; test results are descriptive, not used to select a winner.
    test_rankings = frozen['test']
    print('Evaluating frozen test split', flush=True)
    test_rows, test_summary = run_split(test, test_rankings, vectors, query_vectors, children, parents, tokenizer, 'test')
    write_csv(args.output / 'test_summary.csv', test_summary)
    write_csv(args.output / 'test_per_query.csv', test_rows)
    baseline = {(r['query_id'], r['method'], r['lambda'], r['k'], r['budget']): r for r in test_rows if r['before'] == r['after'] == 0}
    gains = []
    for row in test_rows:
        if row['before'] == row['after'] == 0:
            continue
        base = baseline[(row['query_id'], row['method'], row['lambda'], row['k'], row['budget'])]
        gains.append({key: row[key] for key in ('query_id', 'method', 'lambda', 'k', 'budget', 'before', 'after')} |
            {'additional_hit': row['hit'] - base['hit'], 'additional_coverage': row['coverage'] - base['coverage'],
             'additional_tokens': row['tokens'] - base['tokens']})
    write_csv(args.output / 'sibling_marginal.csv', gains)
    (args.output / 'manifest.json').write_text(json.dumps({'source_manifest_sha256': digest(SOURCE / 'manifest.json'),
        'measurement_tokenizer_sha256': checksum, 'depths': DEPTHS, 'budgets': BUDGETS,
        'policies': POLICIES, 'mmr_lambdas': LAMBDAS, 'mmr_candidate_pool': 100,
        'mmr_score': 'lambda * query cosine - (1-lambda) * max selected passage cosine',
        'status': 'complete'}, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
