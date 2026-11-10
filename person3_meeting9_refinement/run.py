"""Frozen offline rank-fusion refinement. No network/model/API operations."""
from pathlib import Path
import argparse, hashlib, json, math, shutil, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/person3_meeting9_refinement/v1'
EXPORT = ROOT.parent / 'outputs/meeting9_refinement_20261109'
D = ROOT / 'results/person3_meeting9_fullpage/v1'
S = ROOT / 'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
CORPUS = ROOT / 'results/person3_meeting9_transfer/vidore_finance_v1/data/corpus.jsonl'
KS = [5, 10, 15, 20, 25]
GRID = [{'name': f'{label}_rrf_w{int(w*100):03d}', 'dense': dense, 'dense_weight': w}
        for label, dense in [('question', 'question_dense'), ('focusedmix', 'focused160_mix_dense')]
        for w in [.25, .50]]
BASELINES = ['bm25', 'question_dense', 'focused160_mix_dense']

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False)+'\n')
def frozen(p, v):
    if p.exists(): assert read(p) == v, f'Frozen artifact changed: {p}'
    else: dump(p, v)

def inputs():
    dense, sparse = np.load(D/'rankings.npz'), np.load(S/'rankings.npz')
    assert np.array_equal(dense['query_ids'], sparse['query_ids'])
    assert np.array_equal(dense['passage_ids'], sparse['passage_ids'])
    qids, pids = dense['query_ids'].tolist(), dense['passage_ids'].tolist()
    assert len(qids) == len(set(qids)) == 309 and len(pids) == len(set(pids)) == 2942
    assert qids == [q['query_id'] for q in read(S/'selection.json')['queries']]
    ranks = {m: dense[m] for m in BASELINES if m != 'bm25'}
    ranks['bm25'] = sparse['bm25']
    for arr in ranks.values():
        assert arr.shape == (309, 100) and ((arr >= 0) & (arr < 2942)).all()
        assert all(len(set(row)) == 100 for row in arr)
    return qids, pids, ranks, read(S/'qrels.json')

def prepare():
    OUT.mkdir(parents=True, exist_ok=True); EXPORT.mkdir(parents=True, exist_ok=True)
    qids, pids, _, _ = inputs()
    protocol = {
      'status': 'Exploratory refinement of previously inspected ViDoRe Finance English309; not a new holdout or confirmation',
      'frozen_before': 'Any fused rankings or fused outcome metrics',
      'queries': 309, 'corpus_pages': len(pids), 'candidate_grid': GRID,
      'development': 'Original initial60 in retained selection order; already inspected',
      'evaluation': 'Original remaining249 and descriptive full309; both previously inspected',
      'dense': 'Retained full-OCR MiniLM chunk-max ranks: question or original Focused50/50 mix',
      'sparse': 'Retained original-question full-OCR BM25 ranks',
      'rrf': {'k': 60, 'per_branch_route_depth': 100, 'output_depth': 100,
              'score': 'w_dense/(60+1based_dense_rank)+(1-w_dense)/(60+1based_bm25_rank)',
              'absent': 'zero contribution outside retained branch top100',
              'candidate_pages': 'union of branch top100 pages', 'ties': 'ascending original corpus page index'},
      'selection_rule': 'Maximum development60 mean nDCG@10; ties within1e-12 prefer question, then smaller dense weight. Retain BM25 comparator even if selected fusion loses.',
      'evaluation_rule': 'Evaluate selected fusion, same-weight question hybrid, BM25 and both standalone dense baselines on249/309. No other unselected fusion is evaluated there; no reselection or expanded grid.',
      'primary_descriptive_endpoint': 'Mean question-level nDCG@10, linear graded gains',
      'secondary_metrics': {'names': ['Hit', 'Recall', 'MRR', 'nDCG'], 'k': KS},
      'source_descriptives': 'Group275 queries with all positive qrels in one bank by that bank. Keep34 multi-source queries in one separate group; never force assignment. Report descriptively without source-independence inference.',
      'inference': 'No p-values, confidence/confirmation claims, significance search, or new heldout claims; six annual-report sources are shared',
      'rationale_before_fusion': 'Development60 dense/BM25 mean Top10 overlap3.45 question/3.78 Focusedmix; dense adds relevant Top10 pages on20/24 queries and BM25 on25/25.',
      'resources': {'api_calls': 0, 'new_generations': 0, 'new_embeddings': 0, 'new_models': 0, 'paid_cost_usd': 0},
      'limitations': ['Both cohorts were previously inspected; this split does not create a new holdout.',
                      'Truncated top100 route fusion, not full-score fusion.',
                      'Text OCR retrieval only; no image encoder or final-answer reader.',
                      'Focusedmix reuses previously paid generation; question hybrid needs no generation.']}
    frozen(OUT/'protocol.json', protocol)
    frozen(OUT/'selection.json', {'development60_ids': qids[:60], 'evaluation249_ids': qids[60:], 'full309_ids': qids})
    paths = [D/'rankings.npz', D/'protocol.json', D/'per_query_metrics.csv', S/'rankings.npz',
             S/'qrels.json', S/'selection.json', CORPUS]
    frozen(OUT/'source_hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})
    frozen(OUT/'implementation_hashes.json', {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)})
    if not (OUT/'freeze_receipt.json').exists():
        assert not any((OUT/n).exists() for n in ['development_choice.json', 'development_rankings.npz', 'evaluated_rankings.npz', 'per_query_metrics.csv'])
        dump(OUT/'freeze_receipt.json', {'utc': datetime.now(timezone.utc).isoformat(),
          'protocol_sha256': sha(OUT/'protocol.json'), 'implementation_hashes_sha256': sha(OUT/'implementation_hashes.json'),
          'no_fused_rankings_or_metrics_existed_at_freeze': True})
    for name in ['protocol.json', 'selection.json', 'source_hashes.json', 'implementation_hashes.json', 'freeze_receipt.json']:
        shutil.copy2(OUT/name, EXPORT/name)
    print('Frozen4-candidate grid and development60 choice rule; no fused outcomes computed.')

def check_sources():
    for manifest in ['source_hashes.json', 'implementation_hashes.json']:
        for rel, value in read(OUT/manifest).items(): assert sha(ROOT/rel) == value, f'Changed: {rel}'

def rrf(a, b, w):
    scores = {}
    for route, weight in [(a, w), (b, 1-w)]:
        for rank, page in enumerate(route, 1):
            scores[int(page)] = scores.get(int(page), 0.) + weight/(60+rank)
    return np.array(sorted(scores, key=lambda p: (-scores[p], p))[:100], dtype=np.int64)

def score(rank, rels, pids, k):
    grades = [rels.get(pids[int(p)], 0) for p in rank[:k]]
    hits = [i+1 for i, v in enumerate(grades) if v > 0]
    ideal = sum(v/math.log2(i+2) for i, v in enumerate(sorted(rels.values(), reverse=True)[:k]))
    return {'hit': float(bool(hits)), 'recall': len(hits)/sum(v > 0 for v in rels.values()),
            'mrr': 1/hits[0] if hits else 0.,
            'ndcg': sum(v/math.log2(i+2) for i, v in enumerate(grades))/ideal if ideal else 0.}

def metric_rows(qids, pids, ranks, rels, cohort):
    return [{'query_id': q, 'cohort': cohort, 'method': method, 'k': k, **score(rank, rels[q], pids, k)}
            for method, arr in ranks.items() for q, rank in zip(qids, arr) for k in KS]

def choose(frame):
    means = frame[frame.k == 10].groupby('method').ndcg.mean().to_dict()
    best = max(means.values())
    eligible = [c for c in GRID if best-means[c['name']] <= 1e-12]
    return min(eligible, key=lambda c: (c['dense'] != 'question_dense', c['dense_weight'])), means

def develop():
    check_sources(); qids, pids, ranks, rels = inputs()
    # Explicitly restrict every selector input to development60 before scoring.
    qids = qids[:60]; ranks = {m: arr[:60] for m, arr in ranks.items()}; rels = {q: rels[q] for q in qids}
    fused = {c['name']: np.stack([rrf(a, b, c['dense_weight']) for a, b in zip(ranks[c['dense']], ranks['bm25'])]) for c in GRID}
    frame = pd.DataFrame(metric_rows(qids, pids, fused, rels, 'development60'))
    frame.to_csv(OUT/'development_per_query_metrics.csv', index=False)
    chosen, means = choose(frame)
    frozen(OUT/'development_choice.json', {'candidate': chosen, 'development_scores_ndcg10': means,
      'protocol_sha256': sha(OUT/'protocol.json'), 'development_metrics_sha256': sha(OUT/'development_per_query_metrics.csv'),
      'freeze_receipt_sha256': sha(OUT/'freeze_receipt.json'),
      'selector_input_query_ids': qids, 'selector_query_count': len(qids),
      'selector_input_hashes': {'qrels_only60': digest(rels), **{m+'_ranks_only60': digest(a.tolist()) for m, a in ranks.items()}},
      'remaining249_used_for_choice': False})
    np.savez_compressed(OUT/'development_rankings.npz', query_ids=np.array(qids), passage_ids=np.array(pids), **fused)
    print(json.dumps(read(OUT/'development_choice.json'), indent=2))

def evaluate():
    check_sources(); qids, pids, ranks, rels = inputs()
    choice = read(OUT/'development_choice.json'); chosen = choice['candidate']
    assert sha(OUT/'development_per_query_metrics.csv') == choice['development_metrics_sha256']
    assert choice['selector_input_query_ids'] == qids[:60]
    start = time.perf_counter()
    comparator = next(c for c in GRID if c['dense'] == 'question_dense' and c['dense_weight'] == chosen['dense_weight'])
    evaluated_candidates = {c['name']: c for c in [chosen, comparator]}
    dev = np.load(OUT/'development_rankings.npz'); fusion = {}
    for name, c in evaluated_candidates.items():
        later = np.stack([rrf(ranks[c['dense']][i], ranks['bm25'][i], c['dense_weight']) for i in range(60, 309)])
        fusion[name] = np.concatenate([dev[name], later])
    rows = pd.read_csv(OUT/'development_per_query_metrics.csv', dtype={'query_id': str}).to_dict('records')
    rows += metric_rows(qids[:60], pids, {m: a[:60] for m, a in ranks.items()}, {q: rels[q] for q in qids[:60]}, 'development60')
    rows += metric_rows(qids[60:], pids, {m: a[60:] for m, a in {**ranks, **fusion}.items()}, {q: rels[q] for q in qids[60:]}, 'remaining249')
    frame = pd.DataFrame(rows); frame.to_csv(OUT/'per_query_metrics.csv', index=False)
    pooled = frame[frame.method.isin(list(ranks)+list(fusion))].assign(cohort='full309')
    summary = pd.concat([frame, pooled]).groupby(['cohort', 'method', 'k'], sort=False).agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in ['hit', 'recall', 'mrr', 'ndcg']}).reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    np.savez_compressed(OUT/'evaluated_rankings.npz', query_ids=np.array(qids), passage_ids=np.array(pids), **fusion)
    source_groups(qids, rels, frame, list(ranks)+list(fusion))
    dump(OUT/'resources.json', {'local_evaluation_wall_seconds': time.perf_counter()-start, 'api_calls': 0,
       'new_embeddings': 0, 'new_generations': 0, 'paid_cost_usd': 0,
       'timing_scope': 'Cached-rank fusion, metric calculation and artifact writes; no embedding/retrieval timing'})
    validate(); report()

def source_groups(qids, rels, frame, methods):
    pages = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    sources = {str(p['corpus_id']): p['doc_id'] for p in pages}
    groups = []
    for q in qids:
        banks = sorted({sources[p] for p, grade in rels[q].items() if grade > 0})
        groups.append({'query_id': q, 'source_group': banks[0] if len(banks) == 1 else 'multiple_sources', 'relevant_sources': '|'.join(banks), 'source_count': len(banks)})
    mapping = pd.DataFrame(groups)
    assert (mapping.source_count == 1).sum() == 275 and (mapping.source_count > 1).sum() == 34
    assert mapping[mapping.source_count == 1].source_group.nunique() == 6
    mapping.to_csv(OUT/'query_source_groups.csv', index=False)
    data = frame[frame.method.isin(methods)].merge(mapping, on='query_id', validate='many_to_one')
    data = pd.concat([data, data.assign(cohort='full309')])
    result = data.groupby(['cohort', 'source_group', 'method', 'k'], sort=False).agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in ['hit', 'recall', 'mrr', 'ndcg']}).reset_index()
    result.to_csv(OUT/'source_descriptives.csv', index=False)

def validate():
    check_sources(); qids, pids, ranks, rels = inputs()
    choice = read(OUT/'development_choice.json'); chosen = choice['candidate']
    assert choice['protocol_sha256'] == sha(OUT/'protocol.json')
    assert choice['freeze_receipt_sha256'] == sha(OUT/'freeze_receipt.json')
    assert choice['development_metrics_sha256'] == sha(OUT/'development_per_query_metrics.csv')
    frame = pd.read_csv(OUT/'per_query_metrics.csv', dtype={'query_id': str})
    devmetrics = pd.read_csv(OUT/'development_per_query_metrics.csv', dtype={'query_id': str})
    assert choose(devmetrics)[0] == chosen and set(devmetrics.query_id) == set(qids[:60])
    assert choice['selector_query_count'] == 60 and choice['remaining249_used_for_choice'] is False
    assert choice['selector_input_hashes']['qrels_only60'] == digest({q: rels[q] for q in qids[:60]})
    for m, a in ranks.items(): assert choice['selector_input_hashes'][m+'_ranks_only60'] == digest(a[:60].tolist())
    evaluated = np.load(OUT/'evaluated_rankings.npz'); dev = np.load(OUT/'development_rankings.npz')
    names = [n for n in evaluated.files if n not in ['query_ids', 'passage_ids']]
    assert set(frame[frame.cohort == 'remaining249'].method) == set(BASELINES+names)
    assert not frame.duplicated(['query_id', 'method', 'k']).any()
    rank_checks = 0
    for c in GRID:
        limit = 309 if c['name'] in names else 60
        for i in range(limit):
            # Independent array implementation checks dictionary fusion and stable ties.
            scores = np.zeros(2942, dtype=np.float64); w = c['dense_weight']
            scores[ranks[c['dense']][i]] += w/(60+np.arange(1, 101))
            scores[ranks['bm25'][i]] += (1-w)/(60+np.arange(1, 101))
            expected = np.lexsort((np.arange(2942), -scores))[:100]
            got = evaluated[c['name']][i] if c['name'] in names else dev[c['name']][i]
            assert np.array_equal(expected, got) and len(set(got)) == 100
            for k in KS:
                row = frame[(frame.query_id == qids[i]) & (frame.method == c['name']) & (frame.k == k)].iloc[0]
                assert all(abs(row[m]-v) < 1e-12 for m, v in score(got, rels[qids[i]], pids, k).items())
            rank_checks += 1
    old = pd.read_csv(D/'per_query_metrics.csv', dtype={'query_id': str})
    old = old[(old.method.isin(BASELINES)) & old.representation.isin(['fullpage_chunkmax', 'full_ocr_bm25'])]
    check = frame[frame.method.isin(BASELINES)].merge(old, on=['query_id', 'method', 'k'], suffixes=('', '_old'), validate='one_to_one')
    assert len(check) == 309*3*len(KS)
    for m in ['hit', 'recall', 'mrr', 'ndcg']: assert np.allclose(check[m], check[m+'_old'], rtol=0, atol=1e-12)
    summary = pd.read_csv(OUT/'summary.csv')
    for row in summary.to_dict('records'):
        sub = frame[(frame.method == row['method']) & (frame.k == row['k'])]
        if row['cohort'] != 'full309': sub = sub[sub.cohort == row['cohort']]
        assert len(sub) == row['n'] == {'development60': 60, 'remaining249': 249, 'full309': 309}[row['cohort']]
        assert all(abs(sub[m].mean()-row[m]) < 1e-12 for m in ['hit', 'recall', 'mrr', 'ndcg'])
    source = pd.read_csv(OUT/'source_descriptives.csv')
    for row in source.to_dict('records'):
        mapping = pd.read_csv(OUT/'query_source_groups.csv', dtype={'query_id': str})
        ids = set(mapping[mapping.source_group == row['source_group']].query_id)
        sub = frame[(frame.query_id.isin(ids)) & (frame.method == row['method']) & (frame.k == row['k'])]
        if row['cohort'] != 'full309': sub = sub[sub.cohort == row['cohort']]
        assert len(sub) == row['n']
        assert all(abs(sub[m].mean()-row[m]) < 1e-12 for m in ['hit', 'recall', 'mrr', 'ndcg'])
    dump(OUT/'validation.json', {'passed': True, 'api_calls': 0, 'new_pvalues': 0,
      'independent_array_rrf_rank_checks': rank_checks, 'unchanged_baseline_metric_rows': len(check),
      'source_and_implementation_hashes_verified': True, 'selector_inputs_exactly60_verified': True,
      'remaining249_fusions': names, 'source_counts': {'single_source': 275, 'multiple_sources': 34},
      'metric_rows': len(frame), 'summary_rows': len(summary), 'source_summary_rows': len(source),
      'status': 'Exploratory previously inspected benchmark; no new heldout confirmation'})

def report():
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    choice = read(OUT/'development_choice.json'); chosen = choice['candidate']; name = chosen['name']
    summary = pd.read_csv(OUT/'summary.csv'); sources = pd.read_csv(OUT/'source_descriptives.csv')
    names = ['bm25', 'question_dense', 'focused160_mix_dense'] + [n for n in summary[summary.cohort == 'remaining249'].method.unique() if n not in BASELINES]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, cohort in zip(axes, ['remaining249', 'full309']):
        data = summary[(summary.cohort == cohort) & (summary.k == 10)].set_index('method')
        values = data.loc[names, 'ndcg'].to_numpy()
        bars = ax.bar(np.arange(len(names)), values, color=['#c58b44' if n == 'bm25' else '#347da5' if n == name else '#9aa6b1' for n in names])
        for bar, val in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, val+.006, f'{val:.3f}', ha='center', fontsize=8)
        ax.set_xticks(np.arange(len(names)), [n.replace('_dense', '').replace('focused160_mix', 'Focused mix').replace('question', 'Question').replace('_rrf_w', ' RRF .') for n in names], rotation=25, ha='right', fontsize=8)
        ax.set_title(cohort); ax.set_ylim(0, max(values)*1.17); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel('nDCG@10'); fig.suptitle('Offline rank fusion: previously inspected ViDoRe Finance')
    fig.tight_layout(rect=[0,.05,1,.94]); fig.text(.5,.012,'Exploratory descriptive results; no new holdout or p-values. Selected on initial60.',ha='center',fontsize=9)
    fig.savefig(OUT/'rrf_refinement.png', dpi=180); plt.close(fig)
    lines = ['# Offline RRF refinement', '', 'Exploratory results on the previously inspected ViDoRe Finance English309 benchmark: 2,942 pages from six annual reports. This is not a new heldout confirmation. API calls, new generations/embeddings, new models, and new p-values: **0**.', '',
      'Four candidates and development60 nDCG@10 selection were frozen before fused outcomes. RRF uses k=60, one-based ranks, each branch’s top100 route, zero for absent pages, and stable corpus-index ties. Ties prefer question, then smaller dense weight.', '',
      f'Selected: **{name}**. Remaining249 evaluates only the selected fusion and its same-weight question-hybrid comparator, plus unchanged BM25 and dense baselines. No reselection or expanded grid.', '',
      '| Cohort | Method | nDCG@10 | Hit@10 | Recall@10 |', '|---|---|---:|---:|---:|']
    for cohort in ['development60', 'remaining249', 'full309']:
        for row in summary[(summary.cohort == cohort) & (summary.k == 10)].to_dict('records'):
            lines.append(f"| {cohort} | {row['method']} | {row['ndcg']:.4f} | {row['hit']:.1%} | {row['recall']:.4f} |")
    for cohort in ['remaining249', 'full309']:
        data = summary[(summary.cohort == cohort) & (summary.k == 10)].set_index('method')
        comparator = next(n for n in names if n.startswith('question_rrf'))
        lines += ['', f"{cohort}, selected minus BM25: **{data.loc[name,'ndcg']-data.loc['bm25','ndcg']:+.4f}**; minus same-weight question hybrid: **{data.loc[name,'ndcg']-data.loc[comparator,'ndcg']:+.4f}**; minus matching dense standalone: **{data.loc[name,'ndcg']-data.loc[chosen['dense'],'ndcg']:+.4f}**."]
    lines += ['', '![Descriptive fusion results](rrf_refinement.png)', '', '| Relevant-source group (full309) | n | Selected nDCG@10 | BM25 nDCG@10 | Difference |', '|---|---:|---:|---:|---:|']
    for bank, rows in sources[(sources.cohort == 'full309') & (sources.k == 10)].groupby('source_group', sort=False):
        data = rows.set_index('method'); a, b = data.loc[name], data.loc['bm25']
        lines.append(f"| {bank} | {int(a['n'])} | {a.ndcg:.4f} | {b.ndcg:.4f} | {a.ndcg-b.ndcg:+.4f} |")
    lines += ['', 'The275 single-source queries are grouped by bank;34 multi-source queries stay in a separate row. Shared sources are not treated as independent evidence. Both cohorts had prior outcome exposure; full309 also includes development60. This measures OCR retrieval, not visual understanding or final-answer quality.', '',
      'Reproduce: `.venv/bin/python person3_meeting9_refinement/run.py prepare`, then `develop`, then `evaluate`. `validate` independently checks fusion with an array implementation, exact baseline reproduction, selector hashes restricted to60, source groups, summary denominators, and unchanged input/code hashes. Retained ranks, per-query metrics, protocol, choice, hashes, and resources accompany this report.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    for path in OUT.iterdir():
        if path.is_file(): shutil.copy2(path, EXPORT/path.name)
    print((OUT/'README.md').read_text())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare','develop','evaluate','validate','report'])
    {'prepare': prepare, 'develop': develop, 'evaluate': evaluate, 'validate': validate, 'report': report}[parser.parse_args().stage]()
