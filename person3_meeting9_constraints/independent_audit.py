"""Independent offline audit: no producer/helper import, embedding, API or tuning."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/person3_meeting9_constraints/v1'
EXPORT = ROOT.parent/'outputs/meeting9_constraints_20261109'
D = ROOT/'results/person3_meeting9_fullpage/v1'
S = ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
A = ROOT/'results/person3_meeting9_transfer/vidore_finance_v1'
B = ROOT/'results/person3_meeting9_transfer/vidore_finance_remaining249_v1'
METHODS = ('bm25', 'question_dense', 'focused160_mix_dense', 'short_30_50_mix_dense')
METRICS = ('hit', 'recall', 'mrr', 'ndcg')
KS = (5, 10, 15, 20, 25)
ALIASES = {'bank_of_america_2024': ['Bank of America'], 'citigroup_2024': ['Citigroup', 'Citi'],
           'goldman_sachs_2024': ['Goldman Sachs'],
           'jpmorgan_chase_2024': ['JPMorgan', 'JPMorgan Chase', 'JPMorganChase', 'J.P. Morgan'],
           'morgan_stanley_2024': ['Morgan Stanley'], 'wells_fargo_2024': ['Wells Fargo']}
NEGATION = r'\b(?:not|no|except|excluding|exclude|without|other\s+than|rather\s+than|instead\s+of)\b'
OPEN = r'\b(?:(?:other|another)\s+banks?|peers?|competitors?|industry)\b'
QUOTED = r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|(?<!\w)\x27[^\x27\n]*\x27(?!\w)'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right, tolerance=1e-12):
    assert np.allclose(left, right, rtol=0, atol=tolerance)


def independent_route(text):
    # Independently implement the frozen literal rule; never load producer/router.
    text = unicodedata.normalize('NFKC', text)
    spans = set()
    for company, names in ALIASES.items():
        for name in names:
            letters = ''.join(c for c in name if c not in ' .-\t\n\r')
            pattern = r'(?<!\w)' + r'[\s.\-]*'.join(map(re.escape, letters)) + r'(?!\w)'
            spans.update((company, match.start(), match.end()) for match in re.finditer(pattern, text, re.I))
    companies = {company for company, _, _ in spans}
    reason = 'explicit_alias' if companies else 'no_explicit_alias'
    if companies and re.search(NEGATION, text, re.I):
        reason = 'abstain_negation_or_exclusion_marker'
    if any(quote.start() < end and start < quote.end()
           for quote in re.finditer(QUOTED, text) for _, start, end in spans):
        reason = 'abstain_quoted_company_mention'
    if len(companies) == 1 and re.search(OPEN, text, re.I):
        reason = 'abstain_open_comparison'
    return (sorted(companies) if reason == 'explicit_alias' else []), reason


def expand(frame):
    combined = pd.concat([frame, frame.assign(cohort='full309')], ignore_index=True)
    return pd.concat([combined.assign(scope='all'),
                      combined.assign(scope=combined.routed.map({True: 'routed', False: 'fallback'}))], ignore_index=True)


def main():
    # Ignore the new audit files themselves when checking producer artifacts.
    protected = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'independent_audit.json']
    before = {str(p): sha(p) for p in protected}
    protocol, receipt = read(OUT/'protocol.json'), read(OUT/'freeze_receipt.json')
    assert protocol['aliases'] == ALIASES and protocol['methods'] == list(METHODS)
    assert protocol['negation_or_exclusion_regex'] == NEGATION
    assert protocol['open_comparison_regex'] == OPEN and protocol['quoted_mention_regex'] == QUOTED
    assert protocol['same_mask_for_all_methods'] is True and protocol['year_filter'] is False
    assert protocol['primary_descriptive_cohort'] == 'All remaining249, including routed and fallback queries'
    assert receipt['utc'] == '2026-11-10T01:09:31.220783+00:00'
    assert receipt['protocol_sha256'] == sha(OUT/'protocol.json')
    assert receipt['routes_sha256'] == sha(OUT/'routes.json')
    assert receipt['implementation_hashes_sha256'] == sha(OUT/'implementation_hashes.json')
    assert receipt['no_filtered_or_replay_results_existed'] is True and receipt['qrels_loaded_by_prepare'] is False
    checked_hashes = {}
    for filename in ['source_hashes.json', 'implementation_hashes.json', 'diagnosis_input_code_hashes.json']:
        for name, expected in read(OUT/filename).items():
            assert sha(ROOT/name) == expected, name
            checked_hashes[name] = expected
    for name, expected in read(A/'data/manifest.json')['files'].items():
        assert sha(A/name) == expected, name

    qs = read(S/'selection.json')['queries']
    pages = [json.loads(line) for line in (A/'data/corpus.jsonl').read_text().splitlines() if line.strip()]
    qids, pids = [q['query_id'] for q in qs], [str(page['corpus_id']) for page in pages]
    assert len(qids) == len(set(qids)) == 309 and len(pids) == len(set(pids)) == 2942
    original_queries = pd.read_parquet(A/'data/queries.parquet')
    english = {str(row.query_id): row.query for row in original_queries.itertuples() if row.language == 'english'}
    seed = read(A/'selection.json')['seed']
    assert qids == sorted(english, key=lambda q: hashlib.sha256((seed+'|'+q).encode()).hexdigest())
    assert all(q['question'] == english[q['query_id']] for q in qs)
    routes = read(OUT/'routes.json')
    assert len(routes) == 309 and [r['query_id'] for r in routes] == qids
    sources = [page['doc_id'] for page in pages]
    masks, routing = [], []
    for index, (q, saved) in enumerate(zip(qs, routes)):
        docs, reason = independent_route(q['question'])
        allowed = np.array([i for i, bank in enumerate(sources) if not docs or bank in docs], dtype=np.int64)
        cohort = 'initial60' if index < 60 else 'remaining249'
        assert saved['question'] == q['question'] and saved['cohort'] == cohort
        assert saved['route_doc_ids'] == docs and saved['reason'] == reason
        assert saved['allowed_pages'] == len(allowed) and len(allowed) >= 100
        for mention in saved['literal_matches']:
            normalized = unicodedata.normalize('NFKC', q['question'])
            assert normalized[mention['start']:mention['end']] == mention['literal']
            assert mention['alias'] in ALIASES[mention['doc_id']]
        masks.append(allowed)
        routing.append(dict(query_id=q['query_id'], cohort=cohort, routed=bool(docs), route_reason=reason))
    meta = pd.DataFrame(routing)
    original_rels = {qid: {} for qid in qids}
    for row in pd.read_parquet(A/'data/qrels.parquet').itertuples(index=False):
        qid, pid = str(row.query_id), str(row.corpus_id)
        if qid in original_rels:
            original_rels[qid][pid] = max(int(row.score), original_rels[qid].get(pid, 0))
    assert original_rels == read(S/'qrels.json')
    rels = original_rels  # Complete original qrels throughout, irrespective of masks.

    # Dense score reconstruction uses retained vectors, not an encoder/model.
    chunkmeta = read(D/'chunks.json')
    chunks = np.concatenate([np.load(p) for p in sorted((D/'chunk_vectors').glob('*.npy'))])
    questions = np.concatenate([np.load(part/'vectors/questions.npy') for part in (A, B)])
    assert chunks.shape == (14017, 384) and questions.shape == (309, 384)
    assert chunks.dtype == questions.dtype == np.float32
    starts = chunkmeta['page_chunk_starts']
    assert len(starts) == 2943 and starts[-1] == len(chunks)
    assert [page['page_id'] for page in chunkmeta['pages']] == pids
    vectors = {'question_dense': questions}
    for style in ('focused160', 'short_30_50'):
        hyde = np.concatenate([np.load(part/'vectors'/f'{style}.npy') for part in (A, B)])
        mixed = .5*questions + .5*hyde
        vectors[style+'_mix_dense'] = mixed / np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-12)
    full_scores = {}
    maximum_score_error = {}
    gate = read(OUT/'replay_validation.json')
    assert gate['routes_sha256'] == sha(OUT/'routes.json')
    for method, queries in vectors.items():
        scores = []
        for query in queries:
            chunk_scores = chunks @ query
            scores.append([chunk_scores[left:right].max() for left, right in zip(starts[:-1], starts[1:])])
        full_scores[method] = np.asarray(scores)

    # Independent BM25 calculation; no import of the helper or any stored index.
    token_pattern = re.compile(r'[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*')
    vectorizer = CountVectorizer(lowercase=True, tokenizer=lambda text: token_pattern.findall(text.lower()),
                                 token_pattern=None, dtype=np.float32)
    counts = vectorizer.fit_transform([page['markdown'] for page in pages]).tocsr().astype(np.float32)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    average_length = max(float(lengths.mean()), 1e-12)
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((len(pages)-document_frequency+.5)/(document_frequency+.5)).astype(np.float32)
    row_number = np.repeat(np.arange(len(pages)), np.diff(counts.indptr))
    frequencies = counts.data.copy()
    counts.data = (frequencies*2.2/(frequencies+1.2*(.25+.75*lengths[row_number]/average_length)))*idf[counts.indices]
    sparse_scores = []
    for q in qs:
        terms = vectorizer.transform([q['question']]).tocsr()
        terms.data[:] = 1
        sparse_scores.append((terms @ counts.T).toarray().ravel())
    full_scores['bm25'] = np.stack(sparse_scores)
    unfiltered, filtered = np.load(OUT/'unfiltered_rankings.npz'), np.load(OUT/'filtered_rankings.npz')
    retained_dense, retained_sparse = np.load(D/'rankings.npz'), np.load(S/'rankings.npz')
    for archive in (unfiltered, filtered, retained_dense, retained_sparse):
        assert archive['query_ids'].tolist() == qids and archive['passage_ids'].tolist() == pids
    for archive in (unfiltered, filtered):
        assert set(archive.files)-{'query_ids', 'passage_ids'} == set(METHODS)

    def metric(qid, ranking, k):
        grades = np.array([rels[qid].get(pids[int(i)], 0) for i in ranking[:k]], dtype=float)
        positive_ranks = np.flatnonzero(grades > 0)
        ideal = np.array(sorted(rels[qid].values(), reverse=True)[:k], dtype=float)
        idcg = float(np.sum(ideal/np.log2(np.arange(len(ideal))+2)))
        return (float(len(positive_ranks) > 0), len(positive_ranks)/sum(g > 0 for g in rels[qid].values()),
                1/(int(positive_ranks[0])+1) if len(positive_ranks) else 0.,
                float(np.sum(grades/np.log2(np.arange(len(grades))+2)))/idcg if idcg else 0.)

    rows, rank_checks, fallback_checks, backfilled = [], 0, 0, {}
    corpus_indices = np.arange(len(pages))
    for method in METHODS:
        scores = full_scores[method]
        assert scores.shape == (309, 2942) and np.isfinite(scores).all()
        score_file = OUT/f'full_scores_{method}.npy'
        assert sha(score_file) == gate['full_score_hashes'][method]
        saved_scores = np.load(score_file)
        maximum_score_error[method] = float(np.max(np.abs(scores-saved_scores)))
        close(scores, saved_scores, 1e-6)
        original = retained_sparse[method] if method == 'bm25' else retained_dense[method]
        backfilled[method] = 0
        for i, qid in enumerate(qids):
            complete_rank = np.lexsort((corpus_indices, -scores[i]))[:100]
            allowed = masks[i]
            allowed_rank = allowed[np.lexsort((allowed, -scores[i, allowed]))[:100]]
            assert np.array_equal(complete_rank, unfiltered[method][i])
            assert np.array_equal(complete_rank, original[i])
            assert np.array_equal(allowed_rank, filtered[method][i])
            assert len(set(allowed_rank)) == 100 and set(allowed_rank) <= set(allowed)
            backfilled[method] += len(set(allowed_rank)-set(complete_rank))
            if not routing[i]['routed']:
                assert np.array_equal(complete_rank, allowed_rank)
                fallback_checks += 1
            for condition, ranking in [('unfiltered', complete_rank), ('filtered', allowed_rank)]:
                for k in KS:
                    rows.append({**routing[i], 'method': method, 'condition': condition, 'k': k,
                                 **dict(zip(METRICS, metric(qid, ranking, k)))})
            rank_checks += 1
    expected = pd.DataFrame(rows)
    def compare_frame(wanted, saved_name, keys, exact=()):
        saved = pd.read_csv(OUT/saved_name, dtype={'query_id': str})
        joined = wanted.merge(saved, on=keys, suffixes=('_expected', '_actual'), validate='one_to_one')
        assert len(joined) == len(wanted) == len(saved), saved_name
        for column in wanted.columns:
            if column not in keys:
                left, right = joined[column+'_expected'], joined[column+'_actual']
                if column in exact or not pd.api.types.is_numeric_dtype(left):
                    assert left.equals(right), (saved_name, column)
                else:
                    close(left, right)
        return len(saved)
    metric_count = compare_frame(expected, 'per_query_metrics.csv', ['query_id', 'method', 'condition', 'k'], ['cohort', 'routed', 'route_reason'])
    assert metric_count == 12360
    expanded = expand(expected)
    keys = ['cohort', 'scope', 'method', 'condition', 'k']
    summary = expanded.groupby(keys, sort=False).agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in METRICS}).reset_index()
    summary_count = compare_frame(summary, 'summary.csv', keys)
    difference_rows = []
    for group_keys, group in expanded.groupby(['cohort', 'scope', 'method', 'k'], sort=False):
        before_group = group[group.condition == 'unfiltered'].set_index('query_id')
        after_group = group[group.condition == 'filtered'].set_index('query_id')
        difference = after_group[list(METRICS)]-before_group[list(METRICS)]
        difference_rows.append(dict(zip(['cohort', 'scope', 'method', 'k'], group_keys)) |
            {'n': len(difference), **{m+'_delta': difference[m].mean() for m in METRICS},
             'ndcg_wins': int((difference.ndcg > 1e-12).sum()), 'ndcg_losses': int((difference.ndcg < -1e-12).sum()),
             'ndcg_ties': int((abs(difference.ndcg) <= 1e-12).sum())})
    delta_count = compare_frame(pd.DataFrame(difference_rows), 'matched_deltas.csv', ['cohort', 'scope', 'method', 'k'])
    hyde_rows = []
    for group_keys, group in expanded[expanded.k == 10].groupby(['cohort', 'scope', 'condition'], sort=False):
        question = group[group.method == 'question_dense'].set_index('query_id')
        for method in ('focused160_mix_dense', 'short_30_50_mix_dense'):
            hyde = group[group.method == method].set_index('query_id')
            difference = hyde[list(METRICS)]-question[list(METRICS)]
            hyde_rows.append(dict(zip(['cohort', 'scope', 'condition'], group_keys)) |
                {'method': method, 'control': 'question_dense', 'k': 10, 'n': len(difference),
                 **{m+'_delta': difference[m].mean() for m in METRICS}})
    hyde_count = compare_frame(pd.DataFrame(hyde_rows), 'hyde_contrasts.csv', ['cohort', 'scope', 'condition', 'method', 'control', 'k'])

    retention_rows = []
    pid_index = {pid: i for i, pid in enumerate(pids)}
    for i, qid in enumerate(qids):
        positives = {pid: grade for pid, grade in rels[qid].items() if grade > 0}
        allowed = set(masks[i])
        retained = {pid: grade for pid, grade in positives.items() if pid_index[pid] in allowed}
        excluded = {pid: grade for pid, grade in positives.items() if pid not in retained}
        retention_rows.append({**routing[i], 'positive_pages': len(positives), 'retained_positive_pages': len(retained),
            'positive_page_retention': len(retained)/len(positives), 'positive_grade_mass': sum(positives.values()),
            'retained_positive_grade_mass': sum(retained.values()), 'grade_mass_retention': sum(retained.values())/sum(positives.values()),
            'any_gold_excluded': bool(excluded), 'excluded_positive_pages_json': json.dumps(excluded, sort_keys=True)})
    retention = pd.DataFrame(retention_rows)
    retention_count = compare_frame(retention, 'qrel_retention.csv', ['query_id'])
    ret_summary = expand(retention).groupby(['cohort', 'scope'], sort=False).agg(n=('query_id', 'size'),
        queries_with_gold_excluded=('any_gold_excluded', 'sum'), mean_positive_page_retention=('positive_page_retention', 'mean'),
        mean_grade_mass_retention=('grade_mass_retention', 'mean'), positive_pages=('positive_pages', 'sum'),
        retained_positive_pages=('retained_positive_pages', 'sum'), positive_grade_mass=('positive_grade_mass', 'sum'),
        retained_positive_grade_mass=('retained_positive_grade_mass', 'sum')).reset_index()
    ret_summary_count = compare_frame(ret_summary, 'qrel_retention_summary.csv', ['cohort', 'scope'])
    route_rows = []
    for cohort in ('initial60', 'remaining249', 'full309'):
        subset = meta if cohort == 'full309' else meta[meta.cohort == cohort]
        indices = [i for i, r in enumerate(routing) if cohort == 'full309' or r['cohort'] == cohort]
        route_rows.append({'cohort': cohort, 'n': len(subset), 'routed': int(subset.routed.sum()),
            'multiple_banks': sum(len(routes[i]['route_doc_ids']) > 1 for i in indices), 'fallback': int((~subset.routed).sum()),
            **{reason: int((subset.route_reason == reason).sum()) for reason in sorted(set(meta.route_reason))}})
    compare_frame(pd.DataFrame(route_rows), 'routing_summary.csv', ['cohort'])
    assert fallback_checks == 54*4

    report = (OUT/'README.md').read_text()
    labels = {'bm25': 'BM25', 'question_dense': 'Question dense', 'focused160_mix_dense': 'Focused mix', 'short_30_50_mix_dense': 'Short mix'}
    primary = summary[(summary.cohort == 'remaining249') & (summary.scope == 'all') & (summary.k == 10)].set_index(['method', 'condition'])
    numbers = {}
    for method in METHODS:
        old, new = primary.loc[(method, 'unfiltered')], primary.loc[(method, 'filtered')]
        line = f'| {labels[method]} | {old.ndcg:.4f} | {new.ndcg:.4f} | {new.ndcg-old.ndcg:+.4f} | {old.hit:.1%}→{new.hit:.1%} | {old.recall:.4f}→{new.recall:.4f} |'
        assert line in report
        numbers[method] = {'original_ndcg10': old.ndcg, 'filtered_ndcg10': new.ndcg, 'ndcg10_delta': new.ndcg-old.ndcg,
                           'hit10_delta': new.hit-old.hit, 'recall10_delta': new.recall-old.recall}
    assert 'previously inspected' in report and 'no generalization or final-answer accuracy claim' in report
    assert 'gold retention' in report and 'complete original qrels' in report
    # Chart was visually checked externally: method labels, all249 scope, eight
    # three-decimal annotations and bars agree with this independently verified table.
    figure_review = {'visually_reviewed': True, 'path': str(OUT/'company_constraint.png'),
                     'sha256': sha(OUT/'company_constraint.png'), 'labels_scope_and_rounded_values_match': True}
    for filename in ['run.py', 'router.py']:
        tree = ast.parse((ROOT/'person3_meeting9_constraints'/filename).read_text())
        if filename == 'router.py':
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'route')
            assert [a.arg for a in function.args.args] == ['question']
            imports = {alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names}
            assert imports == {'re', 'unicodedata'}
    resources = read(OUT/'resources.json')
    assert all(resources[k] == 0 for k in ('api_calls', 'new_embeddings', 'new_generations', 'new_models', 'paid_cost_usd'))
    assert {str(p): sha(p) for p in protected} == before
    for path in OUT.iterdir():
        if path.is_file() and path.name != 'independent_audit.json':
            assert sha(path) == sha(EXPORT/path.name), path.name
    audit = {'passed': True, 'utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(Path(__file__)),
        'producer_or_helper_imported_or_executed': False, 'audit_api_calls': 0, 'new_embeddings': 0, 'new_pvalues': 0,
        'producer_artifacts_unchanged': len(protected), 'input_code_and_diagnosis_hashes_verified': checked_hashes,
        'original_parquet_query_text_and_qrels_verified': True, 'question_only_routes_recomputed': 309,
        'full_scores_recomputed_from_original_OCR_or_cached_vectors': {m: list(full_scores[m].shape) for m in METHODS},
        'max_absolute_score_error': maximum_score_error, 'unfiltered_top100_exact_replay_checks': rank_checks,
        'same_question_metadata_masks_all_methods': True, 'full_allowed_page_top100_checks': rank_checks,
        'pages_backfilled_beyond_original_top100': backfilled, 'fallback_identity_checks': fallback_checks,
        'original_complete_qrels_IDCG_and_recall_denominators_verified': True,
        'metric_rows_verified': metric_count, 'summary_rows_verified': summary_count, 'matched_delta_rows_verified': delta_count,
        'hyde_contrast_rows_verified': hyde_count, 'qrel_retention_rows_verified': retention_count,
        'qrel_retention_summary_rows_verified': ret_summary_count, 'routing_counts': route_rows,
        'freeze_receipt_internally_consistent': True,
        'freeze_temporal_limit': 'Receipt/content hashes agree; files alone do not independently prove historical execution order.',
        'readme_primary_table_verified': True, 'figure_review': figure_review, 'remaining249_metrics': numbers,
        'remaining249_queries_losing_any_positive_label': int(ret_summary[(ret_summary.cohort == 'remaining249') & (ret_summary.scope == 'all')].queries_with_gold_excluded.iloc[0]),
        'interpretation': 'All249 descriptive improvements are reproducible under one fixed question-only constraint. Preserve full qrels and disclose excluded positives. Reused cohorts/shared reports do not establish new-source generalization, statistical significance, or final-answer accuracy. No rule expansion or reselection.'}
    destination = EXPORT/'independent_audit.json'
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    # Retain the identical audit alongside producer results as requested.
    (OUT/'independent_audit.json').write_bytes(destination.read_bytes())
    print(json.dumps({'passed': True, 'metric_rows': metric_count, 'rank_checks': rank_checks,
                      'fallback_checks': fallback_checks, 'remaining249_metrics': numbers,
                      'audit_json': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
