"""Audit saved hierarchy rankings/metrics without rerunning embedding search."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from week5_hierarchy.hierarchy_experiment import ROOT, KS, MODES, load_data
from hierarchy.io import digest, read_jsonl, write_json, write_csv


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    out = args.results
    children, parents, queries, _ = load_data(ROOT / 'week5_hierarchy/kevin_hierarchy')
    queries = {q['id']: q for q in queries if q['split'] == 'test'}
    manifest = json.loads((out / 'manifest.json').read_text())
    for filename, expected in manifest['code_sha256'].items():
        # Historical runs retain their exact execution sources after folder moves.
        snapshot = manifest.get('code_snapshot_paths', {}).get(filename)
        source = out / snapshot if snapshot else ROOT / filename
        assert digest(source) == expected, filename
    predictions = {}
    for file in (out / 'rankings').glob('*.jsonl'):
        method, mode = file.stem.rsplit('_', 1)
        rows = read_jsonl(file)
        assert len(rows) == len(queries)
        assert {r['query_id'] for r in rows} == set(queries)
        lookup = parents if mode == 'parent' else children
        for r in rows:
            ids = r['passage_ids']
            assert len(ids) == len(set(ids)) == len(r['scores']) == 100
            assert set(ids) <= set(lookup)
            assert all(math.isfinite(s) for s in r['scores'])
            assert r['scores'] == sorted(r['scores'], reverse=True)
        predictions[(method, mode)] = {r['query_id']: r['passage_ids'] for r in rows}
    methods = {method for method, mode in predictions}
    assert methods == {'bm25', 'dense', 'hybrid_w0.25', 'hybrid_w0.5', 'hybrid_w0.75'}
    for mode in ('child', 'parent'):
        for qid in queries:
            for weight in (.25, .5, .75):
                scores = {}
                for component, coefficient in (('bm25', weight), ('dense', 1 - weight)):
                    for rank, pid in enumerate(predictions[(component, mode)][qid], 1):
                        scores[pid] = scores.get(pid, 0.) + coefficient / (60 + rank)
                expected = sorted(scores, key=lambda pid: (-scores[pid], pid))[:100]
                assert predictions[(f'hybrid_w{weight}', mode)][qid] == expected
    detail = csv_rows(out / 'per_query.csv')
    lookup = {}
    for row in detail:
        method, mode, qid = row['method'], row['mode'], row['query_id']
        k, budget = int(row['k']), row['token_budget']
        key = (method, mode, qid, k, budget)
        assert key not in lookup
        lookup[key] = row
        seeds = predictions[(method, 'child' if mode == 'child_to_parent' else mode)][qid][:k]
        unit_ids = list(dict.fromkeys(children[c]['parent_id'] for c in seeds)) if mode == 'child_to_parent' else seeds
        included = set(unit_ids) if mode == 'child' else {c for parent in unit_ids for c in parents[parent]['child_ids']}
        gold = set(queries[qid]['gold_child_ids'])
        gold_units = set(queries[qid]['gold_parent_ids']) if mode == 'parent' else gold
        assert float(row['ranked_unit_hit_at_k']) == float(bool(gold_units & set(seeds)))
        assert float(row['gold_child_in_selected_units']) == float(bool(gold & included))
        assert float(row['returned_units']) == len(unit_ids)
        hit, coverage = float(row['gold_child_hit_in_context']), float(row['gold_body_token_coverage'])
        assert 0 <= hit <= coverage <= 1
        assert hit <= float(bool(gold & included))
        if not budget:
            assert hit == coverage == float(bool(gold & included))
            assert float(row['context_truncated']) == 0
        else:
            assert float(row['returned_content_tokens']) <= int(budget)
    assert len(lookup) == len(methods) * 3 * 80 * 3 * 2
    for method in methods:
        for qid in queries:
            for k in KS:
                child = lookup[(method, 'child', qid, k, '')]
                expand = lookup[(method, 'child_to_parent', qid, k, '')]
                assert child['ranked_unit_hit_at_k'] == expand['ranked_unit_hit_at_k']
                assert float(child['gold_child_hit_in_context']) <= float(expand['gold_child_hit_in_context'])
                for mode in MODES:
                    bounded = lookup[(method, mode, qid, k, '2048')]
                    full = lookup[(method, mode, qid, k, '')]
                    assert float(bounded['gold_body_token_coverage']) <= float(full['gold_body_token_coverage'])
    summaries = csv_rows(out / 'summary.csv')
    assert len(summaries) == len(methods) * 3 * 3 * 2
    fields = ['ranked_unit_hit_at_k', 'ranked_unit_recall_at_k', 'gold_child_in_selected_units',
              'gold_child_hit_in_context', 'gold_child_coverage_in_context', 'all_gold_children_in_context',
              'gold_body_token_coverage', 'returned_units', 'unbudgeted_content_tokens',
              'returned_content_tokens', 'context_truncated']
    for row in summaries:
        rows = [lookup[(row['method'], row['mode'], qid, int(row['k']), row['token_budget'])] for qid in queries]
        assert len(rows) == int(row['n_queries']) == 80
        for field in fields:
            assert math.isclose(float(row[field]), statistics.mean(float(r[field]) for r in rows), abs_tol=1e-12)
        assert int(row['gold_context_hit_count']) == sum(float(r['gold_child_hit_in_context']) for r in rows)
        tokens = sorted(float(r['returned_content_tokens']) for r in rows)
        pos = (len(tokens) - 1) * .95
        lo = math.floor(pos)
        p95 = tokens[lo] + (tokens[math.ceil(pos)] - tokens[lo]) * (pos - lo)
        assert math.isclose(float(row['context_tokens_p95']), p95, abs_tol=1e-9)
        assert float(row['context_tokens_min']) == tokens[0]
        assert float(row['context_tokens_max']) == tokens[-1]
    selected = json.loads((out / 'selected.json').read_text())
    sweep = csv_rows(out / 'dev_sweep.csv')
    score = lambda r: tuple(float(r[f]) for f in ('ndcg5', 'mrr5', 'hit5'))
    for method in ('bm25', 'hybrid'):
        best = max((r for r in sweep if r['mode'] == 'joint' and r['method'] == method), key=score)
        assert float(best['k1']) == selected['k1'] and float(best['b']) == selected['b']
        if method == 'hybrid':
            assert float(best['bm25_weight']) == selected['bm25_weight']

    # Paired counts, including budget-induced losses, for meeting discussion.
    comparisons, examples = [], []
    for method in ('bm25', f'hybrid_w{selected["bm25_weight"]}'):
        for budget in ('', '2048'):
            for k in KS:
                for mode in ('parent', 'child_to_parent'):
                    wins, losses = [], []
                    for qid in queries:
                        a = float(lookup[(method, 'child', qid, k, budget)]['gold_child_hit_in_context'])
                        b = float(lookup[(method, mode, qid, k, budget)]['gold_child_hit_in_context'])
                        if b > a:
                            wins.append(qid)
                        elif b < a:
                            losses.append(qid)
                    comparisons.append({'method': method, 'mode_vs_child': mode, 'k': k, 'token_budget': budget,
                                        'wins': len(wins), 'losses': len(losses), 'ties': 80 - len(wins) - len(losses),
                                        'net_hit_difference': (len(wins) - len(losses)) / 80})
        for mode in ('parent', 'child_to_parent'):
            for qid, q in queries.items():
                child = lookup[(method, 'child', qid, 10, '')]
                full = lookup[(method, mode, qid, 10, '')]
                cap = lookup[(method, mode, qid, 10, '2048')]
                if float(full['gold_child_hit_in_context']) > float(child['gold_child_hit_in_context']) or float(full['gold_child_hit_in_context']) > float(cap['gold_child_hit_in_context']):
                    examples.append({'method': method, 'mode': mode, 'query_id': qid, 'question': q['question'],
                                     'gold_child': q['gold_child_ids'][0], 'gold_parent': q['gold_parent_ids'][0],
                                     'child_hit10': child['gold_child_hit_in_context'],
                                     'unbounded_hit10': full['gold_child_hit_in_context'],
                                     'budgeted_hit10': cap['gold_child_hit_in_context'],
                                     'budgeted_coverage10': cap['gold_body_token_coverage'],
                                     'full_context_tokens10': full['returned_content_tokens']})
    write_csv(out / 'paired_comparisons.csv', comparisons)
    if examples:
        write_csv(out / 'examples.csv', examples)
    write_json(out / 'validation.json', {'passed': True, 'rankings': len(predictions),
        'per_query_rows': len(detail), 'aggregate_rows': len(summaries),
        'checks': ['source checksums and exact child spans', 'code hashes', 'query/ID/ranking integrity', 'RRF recomputation',
                   'independent unbounded membership scoring', 'context budget and expansion invariants',
                   'aggregate recomputation', 'development selection'],
        'limit': 'Budgeted coverage uses the handoff evaluator; validated with its hand-calculated tests and invariants, not a second tokenizer implementation.'})
    print(f'Validated {len(predictions)} rankings files, {len(detail)} query rows, {len(summaries)} aggregate rows.')


if __name__ == '__main__':
    main()
