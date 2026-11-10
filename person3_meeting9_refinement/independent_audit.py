"""Independent, offline audit. Never imports or executes the experiment runner."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/person3_meeting9_refinement/v1'
EXPORT = ROOT.parent / 'outputs/meeting9_refinement_20261109'
DATA = ROOT / 'results/person3_meeting9_transfer/vidore_finance_v1/data'
FULL = ROOT / 'results/person3_meeting9_fullpage/v1'
MERGED = ROOT / 'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
KS = (5, 10, 15, 20, 25)
METRICS = ('hit', 'recall', 'mrr', 'ndcg')
BASE = ('question_dense', 'focused160_mix_dense', 'bm25')
GRID = [(f'{label}_rrf_w{int(weight*100):03d}', method, weight)
        for label, method in [('question', BASE[0]), ('focusedmix', BASE[1])]
        for weight in (.25, .50)]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(a, b):
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def main():
    protected = [p for p in OUT.rglob('*') if p.is_file()]
    before = {str(p): sha(p) for p in protected}
    protocol, choice = read(OUT/'protocol.json'), read(OUT/'development_choice.json')
    receipt = read(OUT/'freeze_receipt.json')
    assert protocol['candidate_grid'] == [dict(name=n, dense=m, dense_weight=w) for n, m, w in GRID]
    assert protocol['rrf']['k'] == 60 and protocol['rrf']['per_branch_route_depth'] == 100
    assert protocol['rrf']['output_depth'] == 100
    assert receipt['protocol_sha256'] == sha(OUT/'protocol.json') == choice['protocol_sha256']
    assert receipt['implementation_hashes_sha256'] == sha(OUT/'implementation_hashes.json')
    assert receipt['no_fused_rankings_or_metrics_existed_at_freeze'] is True
    assert datetime.fromisoformat(receipt['utc']).tzinfo is not None
    assert choice['freeze_receipt_sha256'] == sha(OUT/'freeze_receipt.json')
    assert choice['development_metrics_sha256'] == sha(OUT/'development_per_query_metrics.csv')
    checked_hashes = {}
    for manifest in ['source_hashes.json', 'implementation_hashes.json']:
        for name, expected in read(OUT/manifest).items():
            assert sha(ROOT/name) == expected, name
            checked_hashes[name] = expected
    for name, expected in read(DATA/'manifest.json')['files'].items():
        path = DATA.parent/name
        assert sha(path) == expected, name
    presentation = read(OUT/'presentation_hashes.json')
    assert presentation['code_sha256'] == sha(ROOT/'person3_meeting9_refinement/polish.py')
    assert presentation['evaluation_code_and_raw_metrics_modified'] is False
    for path in OUT.iterdir():
        if path.is_file():
            assert sha(path) == sha(EXPORT/path.name), path.name

    dense, sparse = np.load(FULL/'rankings.npz'), np.load(MERGED/'rankings.npz')
    assert np.array_equal(dense['query_ids'], sparse['query_ids'])
    assert np.array_equal(dense['passage_ids'], sparse['passage_ids'])
    qids, pids = dense['query_ids'].tolist(), dense['passage_ids'].tolist()
    assert len(set(qids)) == len(qids) == 309 and len(set(pids)) == len(pids) == 2942
    queries = pd.read_parquet(DATA/'queries.parquet')
    ids = [str(q) for q in queries[queries.language == 'english'].query_id]
    seed = read(ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/selection.json')['seed']
    assert qids == sorted(ids, key=lambda q: hashlib.sha256((seed+'|'+q).encode()).hexdigest())
    assert [q['query_id'] for q in read(MERGED/'selection.json')['queries']] == qids
    assert read(OUT/'selection.json') == {'development60_ids': qids[:60], 'evaluation249_ids': qids[60:], 'full309_ids': qids}
    assert choice['selector_input_query_ids'] == qids[:60]
    assert choice['selector_query_count'] == 60 and choice['remaining249_used_for_choice'] is False
    rels = {qid: {} for qid in qids}
    original_qrels = pd.read_parquet(DATA/'qrels.parquet')
    for row in original_qrels.itertuples(index=False):
        qid, pid = str(row.query_id), str(row.corpus_id)
        if qid in rels:
            rels[qid][pid] = max(int(row.score), rels[qid].get(pid, 0))
    assert rels == read(MERGED/'qrels.json')
    ranks = {BASE[0]: dense[BASE[0]], BASE[1]: dense[BASE[1]], BASE[2]: sparse[BASE[2]]}
    for method, rank in ranks.items():
        assert rank.shape == (309, 100)
        assert np.all((rank >= 0) & (rank < 2942))
        assert all(len(set(row)) == 100 for row in rank)
        assert choice['selector_input_hashes'][method+'_ranks_only60'] == digest(rank[:60].tolist())
    assert choice['selector_input_hashes']['qrels_only60'] == digest({q: rels[q] for q in qids[:60]})

    def fuse(row_a, row_b, weight):
        scores = np.zeros(len(pids), dtype=np.float64)
        reciprocal = 1 / (60 + np.arange(1, 101, dtype=np.float64))
        scores[row_a] += weight * reciprocal
        scores[row_b] += (1-weight) * reciprocal
        return np.lexsort((np.arange(len(pids)), -scores))[:100]

    def metrics(qid, rank, k):
        grades = np.array([rels[qid].get(pids[int(p)], 0) for p in rank[:k]], dtype=float)
        discounts = 1 / np.log2(np.arange(2, k+2))
        ideal = np.array(sorted(rels[qid].values(), reverse=True)[:k], dtype=float)
        hit_indices = np.flatnonzero(grades > 0)
        idcg = np.dot(ideal, discounts[:len(ideal)])
        return (float(len(hit_indices) > 0), len(hit_indices)/sum(g > 0 for g in rels[qid].values()),
                1/(int(hit_indices[0])+1) if len(hit_indices) else 0., float(np.dot(grades, discounts)/idcg) if idcg else 0.)

    dev = np.load(OUT/'development_rankings.npz')
    evaluated = np.load(OUT/'evaluated_rankings.npz')
    assert dev['query_ids'].tolist() == qids[:60] and dev['passage_ids'].tolist() == pids
    assert evaluated['query_ids'].tolist() == qids and evaluated['passage_ids'].tolist() == pids
    assert set(dev.files)-{'query_ids', 'passage_ids'} == {x[0] for x in GRID}
    recalculated_dev, means = {}, {}
    for name, method, weight in GRID:
        recalc = np.stack([fuse(ranks[method][i], ranks['bm25'][i], weight) for i in range(60)])
        assert np.array_equal(recalc, dev[name])
        recalculated_dev[name] = recalc
        means[name] = float(np.mean([metrics(qids[i], recalc[i], 10)[3] for i in range(60)]))
    maximum = max(means.values())
    eligible = [entry for entry in GRID if maximum-means[entry[0]] <= 1e-12]
    winner = min(eligible, key=lambda x: (x[1] != 'question_dense', x[2]))
    assert choice['candidate'] == dict(name=winner[0], dense=winner[1], dense_weight=winner[2])
    for name in means:
        close(means[name], choice['development_scores_ndcg10'][name])
    # Evaluate only the frozen winner and its predeclared same-weight question comparator.
    permitted = {winner[0], next(n for n, m, w in GRID if m == 'question_dense' and w == winner[2])}
    assert set(evaluated.files)-{'query_ids', 'passage_ids'} == permitted
    fused = {}
    for name, method, weight in GRID:
        if name in permitted:
            later = np.stack([fuse(ranks[method][i], ranks['bm25'][i], weight) for i in range(60, 309)])
            fused[name] = np.concatenate([recalculated_dev[name], later])
            assert np.array_equal(fused[name], evaluated[name])

    rows = []
    for name, rank in {**ranks, **recalculated_dev}.items():
        for i in range(60):
            for k in KS:
                rows.append((qids[i], 'development60', name, k, *metrics(qids[i], rank[i], k)))
    for name, rank in {**ranks, **fused}.items():
        for i in range(60, 309):
            for k in KS:
                rows.append((qids[i], 'remaining249', name, k, *metrics(qids[i], rank[i], k)))
    expected = pd.DataFrame(rows, columns=['query_id', 'cohort', 'method', 'k', *METRICS])
    key = ['query_id', 'cohort', 'method', 'k']
    actual = pd.read_csv(OUT/'per_query_metrics.csv', dtype={'query_id': str})
    assert not actual.duplicated(key).any() and len(actual) == len(expected) == 7080
    paired = expected.merge(actual, on=key, suffixes=('_expected', '_actual'), validate='one_to_one')
    assert len(paired) == len(expected)
    for metric in METRICS:
        close(paired[metric+'_expected'], paired[metric+'_actual'])
    actual_dev = pd.read_csv(OUT/'development_per_query_metrics.csv', dtype={'query_id': str})
    dev_expected = expected[(expected.cohort == 'development60') & expected.method.isin(recalculated_dev)]
    paired_dev = dev_expected.merge(actual_dev, on=key, suffixes=('_expected', '_actual'), validate='one_to_one')
    assert len(paired_dev) == len(dev_expected) == len(actual_dev) == 1200
    for metric in METRICS:
        close(paired_dev[metric+'_expected'], paired_dev[metric+'_actual'])

    methods = set(BASE) | permitted
    pooled = expected[expected.method.isin(methods)].assign(cohort='full309')
    def aggregate(frame, keys):
        return frame.groupby(keys, sort=False).agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in METRICS}).reset_index()
    def verify_summary(wanted, filename, keys):
        saved = pd.read_csv(OUT/filename)
        joined = wanted.merge(saved, on=keys, suffixes=('_expected', '_actual'), validate='one_to_one')
        assert len(joined) == len(saved) == len(wanted)
        for metric in ('n', *METRICS):
            close(joined[metric+'_expected'], joined[metric+'_actual'])
        return len(joined)
    summary = aggregate(pd.concat([expected, pooled]), ['cohort', 'method', 'k'])
    summary_count = verify_summary(summary, 'summary.csv', ['cohort', 'method', 'k'])
    corpus = [json.loads(line) for line in (DATA/'corpus.jsonl').read_text().splitlines() if line.strip()]
    page_sources = {str(row['corpus_id']): row['doc_id'] for row in corpus}
    mappings = []
    for qid in qids:
        sources = sorted({page_sources[pid] for pid, grade in rels[qid].items() if grade > 0})
        mappings.append(dict(query_id=qid, source_group=sources[0] if len(sources) == 1 else 'multiple_sources',
                             relevant_sources='|'.join(sources), source_count=len(sources)))
    mapping = pd.DataFrame(mappings)
    saved_mapping = pd.read_csv(OUT/'query_source_groups.csv', dtype={'query_id': str})
    pd.testing.assert_frame_equal(mapping, saved_mapping)
    counts = mapping.source_count.value_counts().sort_index().to_dict()
    assert counts == {1: 275, 2: 31, 3: 3}
    sourced = expected[expected.method.isin(methods)].merge(mapping, on='query_id', validate='many_to_one')
    sourced = pd.concat([sourced, sourced.assign(cohort='full309')])
    source_summary = aggregate(sourced, ['cohort', 'source_group', 'method', 'k'])
    source_summary_count = verify_summary(source_summary, 'source_descriptives.csv', ['cohort', 'source_group', 'method', 'k'])

    # Static inspection of the frozen runner's imports/calls; do not execute it.
    tree = ast.parse((ROOT/'person3_meeting9_refinement/run.py').read_text())
    imports = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports <= {'pathlib', 'argparse', 'hashlib', 'json', 'math', 'shutil', 'time', 'datetime', 'numpy', 'pandas', 'os', 'matplotlib'}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'eval', 'exec', '__import__', 'compile'} for node in ast.walk(tree))
    resources = read(OUT/'resources.json')
    assert all(resources[k] == 0 for k in ['api_calls', 'new_embeddings', 'new_generations', 'paid_cost_usd'])
    report = (OUT/'README.md').read_text()
    assert 'essentially matches BM25' in report and 'reducing Hit@10 and Recall@10' in report
    assert 'Both cohorts had prior outcome exposure' in report and 'full309 also includes development60' in report
    assert 'selected configuration uses no HyDE' in report and 'HyDE is ineffective' not in report
    assert {str(p): sha(p) for p in protected} == before
    figures = summary[(summary.k == 10) & summary.method.isin([winner[0], 'bm25'])].set_index(['cohort', 'method'])
    numeric = {}
    for cohort in ['development60', 'remaining249', 'full309']:
        selected, baseline = figures.loc[(cohort, winner[0])], figures.loc[(cohort, 'bm25')]
        numeric[cohort] = {'selected_ndcg10': selected.ndcg, 'bm25_ndcg10': baseline.ndcg,
                           **{'delta_'+m+'10': selected[m]-baseline[m] for m in METRICS}}
    audit = dict(passed=True, utc=datetime.now(timezone.utc).isoformat(), audit_script_sha256=sha(Path(__file__)),
                 producer_imported_or_executed=False, audit_api_calls=0, protected_result_files_unchanged=len(protected),
                 frozen_input_and_code_hashes=checked_hashes, original_parquet_qrels_and_manifest_verified=True,
                 freeze_receipt_internally_consistent=True,
                 freeze_temporal_limit='Content hashes and receipt agree; files alone cannot independently prove historical execution order.',
                 four_candidate_development60_recomputed=True, selector_inputs_only60_verified=True,
                 winner=winner[0], development_scores_ndcg10=means, remaining249_fusions=sorted(permitted),
                 unselected_remaining249_fusions_computed_by_audit=0,
                 independently_recomputed_fusion_query_rankings=60*4+249*len(permitted),
                 independently_recomputed_metric_rows=len(expected), summary_rows=summary_count,
                 source_summary_rows=source_summary_count, source_count_distribution=counts,
                 no_api_evidence='Frozen runner static imports/calls and resource record; audit used only local reads and independent NumPy/Pandas arithmetic.',
                 report_assessment='Correctly labels reused benchmark and near-zero 249 difference; no claim that unselected Focused hybrid fails or that HyDE is ineffective.',
                 metrics=numeric,
                 recommendation='Retain BM25 and stop this search. Remaining249 gain is negligible and Hit/Recall decline; full309 includes selected development data. Focused hybrid was not evaluated on249, so this run does not establish HyDE ineffectiveness.')
    destination = EXPORT/'independent_audit.json'
    destination.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'passed': True, 'winner': winner[0], 'metrics': numeric, 'audit_json': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
