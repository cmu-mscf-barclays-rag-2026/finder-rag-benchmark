"""Audit all six frozen hybrids independently; no producer/helper imports."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/person3_meeting9_constrained_hybrid/v1'
EXPORT = ROOT.parent/'outputs/meeting9_constrained_hybrid_20261109'
C = ROOT/'results/person3_meeting9_constraints/v1'
S = ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
A = ROOT/'results/person3_meeting9_transfer/vidore_finance_v1'
BASES = ('bm25', 'question_dense', 'focused160_mix_dense', 'short_30_50_mix_dense')
GRID = [(f'{label}_bank_rrf_w{int(weight*100):03d}', dense, weight)
        for label, dense in [('question', BASES[1]), ('focusedmix', BASES[2]), ('shortmix', BASES[3])]
        for weight in (.25, .50)]
KS, METRICS = (5, 10, 15, 20, 25), ('hit', 'recall', 'mrr', 'ndcg')
LABELS = {'bm25': 'BM25', 'question_dense': 'Question dense',
          'focused160_mix_dense': 'Focused mix', 'short_30_50_mix_dense': 'Short mix'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv(path):
    return pd.read_csv(path, dtype={'query_id': str}, float_precision='round_trip')


def close(left, right):
    assert np.allclose(left, right, rtol=0, atol=1e-12)


def expand(frame):
    combined = pd.concat([frame, frame.assign(cohort='full309')], ignore_index=True)
    return pd.concat([combined.assign(scope='all'),
                      combined.assign(scope=combined.routed.map({True: 'routed', False: 'fallback'}))], ignore_index=True)


def compare(wanted, filename, keys):
    saved = csv(OUT/filename)
    joined = wanted.merge(saved, on=keys, suffixes=('_expected', '_actual'), validate='one_to_one')
    assert len(joined) == len(wanted) == len(saved), filename
    for column in wanted.columns:
        if column not in keys:
            left, right = joined[column+'_expected'], joined[column+'_actual']
            if pd.api.types.is_numeric_dtype(left):
                close(left, right)
            else:
                assert left.equals(right), (filename, column)
    return len(saved)


def main():
    protected = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'audit.json']
    before = {str(p): sha(p) for p in protected}
    protocol, receipt = read(OUT/'protocol.json'), read(OUT/'freeze_receipt.json')
    assert protocol['candidate_grid'] == [dict(name=n, dense=m, dense_weight=w) for n, m, w in GRID]
    assert protocol['standalone_baselines'] == list(BASES) and protocol['mix_question_weight'] == .5
    assert protocol['k'] == list(KS) and protocol['metrics'] == list(METRICS)
    assert protocol['rrf'] == {'k': 60, 'branch_depth': 100, 'output_depth': 100, 'rank': 'one-based',
        'score': 'w_dense/(60+dense_rank)+(1-w_dense)/(60+bm25_rank)',
        'absent': 0, 'pages': 'union of branch top100 lists', 'ties': 'ascending original corpus index'}
    assert receipt['utc'] == '2026-11-10T01:19:52.933289+00:00'
    assert receipt['all6_fixed_before_outcomes'] is True and receipt['no_fused_outcomes_existed'] is True
    for name in ('protocol', 'selection', 'routes', 'implementation_hashes'):
        assert receipt[name+'_sha256'] == sha(OUT/(name+'.json'))
    checked_hashes = {}
    for manifest in ('source_hashes.json', 'implementation_hashes.json'):
        for name, expected in read(OUT/manifest).items():
            assert sha(ROOT/name) == expected, name
            checked_hashes[name] = expected
    # Verify the inherited audited retrieval implementation and inputs, without
    # recomputing any old dense/BM25 scores or importing its code.
    for manifest in ('source_hashes.json', 'implementation_hashes.json'):
        for name, expected in read(C/manifest).items():
            assert sha(ROOT/name) == expected, name
    previous_audit = read(C/'independent_audit.json')
    assert previous_audit['passed'] is True
    assert previous_audit['original_complete_qrels_IDCG_and_recall_denominators_verified'] is True
    assert previous_audit['full_allowed_page_top100_checks'] == 1236
    for name, expected in read(A/'data/manifest.json')['files'].items():
        assert sha(A/name) == expected, name
    assert (OUT/'routes.json').read_bytes() == (C/'routes.json').read_bytes()
    assert protocol['original_routes_sha256'] == sha(C/'routes.json')
    assert protocol['original_selection_sha256'] == sha(S/'selection.json')
    inputs, saved_ranks = np.load(C/'filtered_rankings.npz'), np.load(OUT/'rankings.npz')
    qids, pids = inputs['query_ids'].tolist(), inputs['passage_ids'].tolist()
    assert len(qids) == len(set(qids)) == 309 and len(pids) == len(set(pids)) == 2942
    assert saved_ranks['query_ids'].tolist() == qids and saved_ranks['passage_ids'].tolist() == pids
    assert set(saved_ranks.files)-{'query_ids', 'passage_ids'} == set(BASES) | {n for n, _, _ in GRID}
    selection = read(OUT/'selection.json')
    assert selection['initial60_ids'] == qids[:60] and selection['remaining249_ids'] == qids[60:]
    assert selection['full309_ids'] == qids and selection['passage_ids'] == pids
    assert qids == [q['query_id'] for q in read(S/'selection.json')['queries']]
    routes = read(OUT/'routes.json')
    assert [r['query_id'] for r in routes] == qids
    pages = [json.loads(line) for line in (A/'data/corpus.jsonl').read_text().splitlines() if line.strip()]
    assert pids == [str(p['corpus_id']) for p in pages]
    sources = np.array([page['doc_id'] for page in pages])
    rels = {qid: {} for qid in qids}
    for row in pd.read_parquet(A/'data/qrels.parquet').itertuples(index=False):
        qid, pid = str(row.query_id), str(row.corpus_id)
        if qid in rels:
            rels[qid][pid] = max(int(row.score), rels[qid].get(pid, 0))
    assert rels == read(S/'qrels.json')

    arrays = {}
    for method in BASES:
        array = inputs[method]
        assert array.shape == (309, 100) and np.array_equal(array, saved_ranks[method])
        assert np.all((array >= 0) & (array < len(pids)))
        assert all(len(set(row)) == 100 for row in array)
        arrays[method] = array
    rank_checks = 0
    for name, dense, weight in GRID:
        independent = []
        for index in range(309):
            scores = np.zeros(len(pids), dtype=np.float64)
            ranks = np.arange(1, 101, dtype=np.float64)
            scores[inputs[dense][index]] += weight/(60+ranks)
            scores[inputs['bm25'][index]] += (1-weight)/(60+ranks)
            union = np.union1d(inputs[dense][index], inputs['bm25'][index])
            ranking = union[np.lexsort((union, -scores[union]))[:100]]
            assert np.array_equal(ranking, saved_ranks[name][index])
            assert len(set(ranking)) == 100 and np.all(scores[ranking] > 0)
            independent.append(ranking)
            rank_checks += 1
        arrays[name] = np.stack(independent)
    allowed_checks = 0
    for method, array in arrays.items():
        for index, route in enumerate(routes):
            assert route['cohort'] == ('initial60' if index < 60 else 'remaining249')
            if route['route_doc_ids']:
                assert set(sources[array[index]]) <= set(route['route_doc_ids'])
            allowed_checks += 1

    def metric(qid, rank, k):
        # Denominators deliberately use the COMPLETE original qrels.
        grades = np.array([rels[qid].get(pids[int(p)], 0) for p in rank[:k]], dtype=float)
        positive = np.flatnonzero(grades > 0)
        ideal = np.array(sorted(rels[qid].values(), reverse=True)[:k], dtype=float)
        idcg = float(np.sum(ideal/np.log2(np.arange(len(ideal))+2)))
        return (float(len(positive) > 0), len(positive)/sum(g > 0 for g in rels[qid].values()),
                1/(int(positive[0])+1) if len(positive) else 0.,
                float(np.sum(grades/np.log2(np.arange(len(grades))+2)))/idcg if idcg else 0.)
    rows = []
    for method, array in arrays.items():
        for i, route in enumerate(routes):
            for k in KS:
                rows.append({'query_id': qids[i], 'cohort': route['cohort'], 'routed': bool(route['route_doc_ids']),
                    'route_reason': route['reason'], 'method': method,
                    'family': 'constrained_standalone' if method in BASES else 'constrained_hybrid',
                    'k': k, **dict(zip(METRICS, metric(qids[i], array[i], k)))})
    expected = pd.DataFrame(rows)
    metric_count = compare(expected, 'per_query_metrics.csv', ['query_id', 'method', 'k'])
    assert metric_count == 15450
    actual = csv(OUT/'per_query_metrics.csv')
    old = csv(C/'per_query_metrics.csv')
    old = old[(old.condition == 'filtered') & old.method.isin(BASES)]
    joined = actual[actual.method.isin(BASES)].merge(old, on=['query_id', 'method', 'k'], suffixes=('', '_old'), validate='one_to_one')
    assert len(joined) == 6180
    for metric_name in METRICS:
        assert np.array_equal(joined[metric_name].to_numpy(), joined[metric_name+'_old'].to_numpy())
    data = expand(expected)
    summary_keys = ['cohort', 'scope', 'method', 'k']
    summary = data.groupby(summary_keys, sort=False).agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in METRICS}).reset_index()
    summary_count = compare(summary, 'summary.csv', summary_keys)
    assert summary_count == 450
    comparison_rows = []
    for name, dense, weight in GRID:
        controls = [('same_mask_bm25', 'bm25'), ('matching_dense', dense)]
        if dense != 'question_dense':
            control = next(n for n, m, w in GRID if m == 'question_dense' and w == weight)
            controls.append(('same_weight_question_hybrid', control))
        for group_keys, group in data.groupby(['cohort', 'scope', 'k'], sort=False):
            hybrid = group[group.method == name].set_index('query_id')
            for relation, control in controls:
                reference = group[group.method == control].set_index('query_id')
                delta = hybrid[list(METRICS)]-reference[list(METRICS)]
                comparison_rows.append(dict(zip(['cohort', 'scope', 'k'], group_keys)) |
                    {'method': name, 'control': control, 'comparison': relation, 'n': len(delta),
                     **{m+'_delta': delta[m].mean() for m in METRICS},
                     'ndcg_wins': int((delta.ndcg > 1e-12).sum()), 'ndcg_losses': int((delta.ndcg < -1e-12).sum()),
                     'ndcg_ties': int((abs(delta.ndcg) <= 1e-12).sum())})
    comparison_count = compare(pd.DataFrame(comparison_rows), 'matched_comparisons.csv',
        ['cohort', 'scope', 'k', 'method', 'control', 'comparison'])
    assert comparison_count == 720

    report = (OUT/'README.md').read_text()
    primary = summary[(summary.cohort == 'remaining249') & (summary.scope == 'all') & (summary.k == 10)].set_index('method')
    for method in BASES:
        row = primary.loc[method]
        assert f'| {LABELS[method]} | {row.ndcg:.4f} | {row.hit:.1%} | {row.recall:.4f} |' in report
    numbers = {}
    for name, dense, weight in GRID:
        row = primary.loc[name]
        question = next(n for n, m, w in GRID if m == 'question_dense' and w == weight)
        bm_delta, dense_delta = row.ndcg-primary.loc['bm25', 'ndcg'], row.ndcg-primary.loc[dense, 'ndcg']
        q_delta = row.ndcg-primary.loc[question, 'ndcg']
        assert f'| {LABELS[dense]} / w={weight:.2f} | {row.ndcg:.4f} | {bm_delta:+.4f} | {dense_delta:+.4f} | {q_delta:+.4f} |' in report
        secondary = summary[(summary.scope == 'all') & (summary.method == name) & (summary.k == 10)].set_index('cohort')
        assert f"| {LABELS[dense]} / w={weight:.2f} | {secondary.loc['initial60', 'ndcg']:.4f} | {secondary.loc['full309', 'ndcg']:.4f} |" in report
        numbers[name] = {'ndcg10': row.ndcg, 'hit10': row.hit, 'recall10': row.recall,
                        'delta_vs_bm25': bm_delta, 'delta_vs_dense': dense_delta, 'delta_vs_same_weight_question': q_delta}
    assert 'All six settings are reported' in report and 'no development selection' in report
    assert 'repeated outcome exposure' in report and 'no new heldout confirmation' in report
    assert 'complete original qrels' in report and '13/249' in report
    assert 'truncated-rank fusion' in report
    # The plot was visually inspected: all six settings, both weights, the BM25
    # reference line, three-decimal annotations, and exploratory caption match.
    figure = {'visually_reviewed': True, 'sha256': sha(OUT/'constrained_hybrid.png'),
              'all_six_bars_weights_values_and_bm25_reference_match': True}
    tree = ast.parse((ROOT/'person3_meeting9_constrained_hybrid/run.py').read_text())
    imports = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports <= {'pathlib', 'datetime', 'argparse', 'hashlib', 'json', 'math', 'os', 'shutil', 'time', 'numpy', 'pandas', 'matplotlib'}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'exec', 'eval', '__import__'} for node in ast.walk(tree))
    resources = read(OUT/'resources.json')
    assert all(resources[key] == 0 for key in ('api_calls', 'new_embeddings', 'new_generations', 'new_models', 'paid_cost_usd'))
    assert {str(p): sha(p) for p in protected} == before
    for path in OUT.iterdir():
        if path.is_file() and path.name != 'audit.json':
            assert sha(path) == sha(EXPORT/path.name), path.name
    audit = {'passed': True, 'utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(Path(__file__)),
        'producer_or_helper_imported_or_executed': False, 'old_retrieval_scores_recomputed': False,
        'api_calls': 0, 'new_embeddings': 0, 'new_pvalues': 0, 'producer_artifacts_unchanged': len(protected),
        'frozen_input_and_implementation_hashes_verified': checked_hashes,
        'previous_independent_constraint_audit_sha256': sha(C/'independent_audit.json'),
        'inherited_retrieval_code_and_inputs_verified': True, 'routes_identical_to_preceding_fixed_routes': True,
        'original_parquet_qrels_verified': True, 'original_full_qrels_IDCG_and_recall_denominators_verified': True,
        'independently_recomputed_rrf_rankings': rank_checks, 'allowed_page_checks_all_methods': allowed_checks,
        'independently_recomputed_metric_rows': metric_count, 'baseline_metric_rows_bit_identical': len(joined),
        'summary_rows_verified': summary_count, 'matched_comparison_rows_verified': comparison_count,
        'all_six_settings_reported_without_development_selection': True,
        'freeze_receipt_internally_consistent': True,
        'freeze_temporal_limit': 'Receipt/content hashes agree; files alone do not independently prove historical execution order.',
        'readme_primary_and_secondary_tables_verified': True, 'figure_review': figure,
        'same_mask_bm25_ndcg10_remaining249': primary.loc['bm25', 'ndcg'], 'remaining249_all_six_results': numbers,
        'interpretation': 'All six fixed hybrids descriptively exceed same-mask BM25 on repeatedly inspected249. Focused mix w0.50 is highest in this fixed grid; no new holdout/significance/generalization or answer-quality claim follows. Stop this round without expanding the grid.'}
    destination = EXPORT/'audit.json'
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    (OUT/'audit.json').write_bytes(destination.read_bytes())
    print(json.dumps({'passed': True, 'rrf_checks': rank_checks, 'metrics': metric_count, 'summaries': summary_count,
        'comparisons': comparison_count, 'baseline_rows_identical': len(joined), 'all_six_results': numbers,
        'audit_json': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
