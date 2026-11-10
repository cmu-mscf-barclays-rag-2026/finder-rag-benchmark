"""Cost table and fixed single-thread warm CPU profiling. No API or downloads."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/person3_meeting9_followup/costs_v1'
EXPORT = ROOT.parent/'outputs/meeting9_followup_20261109/costs'
OUT.mkdir(parents=True, exist_ok=True)
EXPORT.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false',
                  HF_HOME=str(OUT/'hf_runtime'), TORCH_HOME=str(OUT/'torch_runtime'),
                  MPLCONFIGDIR=str(OUT/'matplotlib_cache'))

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from threadpoolctl import threadpool_limits, threadpool_info

A = ROOT/'results/person3_meeting9_transfer/vidore_finance_v1'
B = ROOT/'results/person3_meeting9_transfer/vidore_finance_remaining249_v1'
D = ROOT/'results/person3_meeting9_fullpage/v1'
C = ROOT/'results/person3_meeting9_constraints/v1'
H = ROOT/'results/person3_meeting9_constrained_hybrid/v1'
S = ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
LEDGER = ROOT/'results/person3_meeting9_workflow/v1/cost_summary.json'
SNAPSHOT = ROOT.parent/'model_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
METHODS = ('bm25', 'question_bank_rrf_w050', 'focusedmix_bank_rrf_w050')
LABELS = dict(zip(METHODS, ('Constrained BM25', 'Question hybrid .50', 'Focused hybrid .50')))
REPEATS, WARMUP, SEED = 3, 20, 20261109


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def csv(path):
    return pd.read_csv(path, dtype={'query_id': str}, float_precision='round_trip')


def quantiles(values):
    return {'p50': float(np.median(values)), 'p95': float(np.quantile(values, .95))}


def metadata_state(directory):
    return {str(path.relative_to(directory)): [path.lstat().st_size, path.lstat().st_mtime_ns]
            for path in directory.rglob('*')}


def main():
    start_all = time.perf_counter()
    source_paths = [Path(__file__), LEDGER, A/'successful_api_outputs.json', B/'successful_api_outputs.json',
        ROOT/'results/person3_meeting9_ablation/factorial20_v1/successful_api_outputs.json',
        A/'data/corpus.jsonl', S/'selection.json', H/'rankings.npz', H/'per_query_metrics.csv',
        C/'routes.json', C/'filtered_rankings.npz', ROOT/'person3_meeting9_constraints/router.py',
        D/'chunks.json', D/'resources.json', D/'execution_backend.json', D/'index_validation.json']
    source_paths += sorted((D/'chunk_vectors').glob('*.npy'))
    source_paths += [part/'vectors'/name for part in (A, B) for name in ('questions.npy', 'focused160.npy')]
    hashes_before = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    model_root = SNAPSHOT.parents[1]
    cache_before = metadata_state(model_root)
    snapshot_hashes = {path.name: sha(path) for path in SNAPSHOT.iterdir() if path.is_file()}
    assert 'model.safetensors' in snapshot_hashes
    ledger = read(LEDGER)
    protocol = {'scope': 'Fixed cost/performance follow-up, no quality tuning or API calls',
        'methods': list(METHODS), 'primary_cohort': 'remaining249 including fallback', 'secondary_cohort': 'full309',
        'warmup_queries_per_method': WARMUP, 'repeats': REPEATS, 'all_queries': 309,
        'query_order': 'Seeded permutation per repeat; method order rotated per query', 'seed': SEED,
        'cpu_threads': 1, 'encoder_batch_size': 1, 'device': 'CPU float32',
        'cache_state': 'Model/index resident, warmed OS caches; cached query vectors for measured retrieval',
        'retrieval': 'Question-only router; global BM25 statistics/full chunk scoring; allowed-page top100; fixed RRF k60/.50',
        'encoder': 'Pinned local MiniLM snapshot only; fresh question/retained Focused-text profiling; no old vector writes',
        'encoder_warmup': '20 question + 20 retained Focused-text single-text calls',
        'modeled_path': 'Per-query median local retrieval + per-query median measured encoding + historical Focused API duration where needed',
        'not_e2e': 'Stage-sum estimate, not measured end-to-end; excludes queue, scheduling, reader, process/index/model startup',
        'rates_per_million': ledger['prices_per_million'], 'rates_scope': 'Retained experiment rates, not a current pricing quote; cached discount ignored',
        'historical_generation': 'Observed request duration; retain explicit concurrency and label missing legacy fields unknown',
        'document_index_build': 'Existing 8-thread chunk encoding timing retained; new 1-thread BM25 build separately measured',
        'no_api_calls': True, 'no_new_document_embeddings': True, 'no_retrieval_cache_modifications': True}
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json') == protocol
    else:
        dump(OUT/'protocol.json', protocol)
    dump(OUT/'source_hashes.json', hashes_before)

    calls = []
    focused = {}
    for source in source_paths[2:5]:
        cohort = 'initial60' if source.parent == A else 'remaining249' if source.parent == B else 'ablation40'
        for row in read(source):
            usage = row['usage']
            cost = (usage['input_tokens']*ledger['prices_per_million']['input'] + usage['output_tokens']*ledger['prices_per_million']['output'])/1e6
            calls.append({'key': row['key'], 'dataset': row['dataset'], 'style': row['style'], 'query_id': str(row['query_id']),
                'cohort': cohort, 'model': row['model_returned'], 'input_tokens': usage['input_tokens'],
                'output_tokens': usage['output_tokens'], 'total_tokens': usage['total_tokens'],
                'cached_input_tokens': usage.get('input_tokens_details', {}).get('cached_tokens', 0),
                'estimated_usd': cost, 'historical_request_seconds': row['latency_seconds'],
                'request_concurrency': row.get('request_concurrency'),
                'concurrency_provenance': 'explicit_per_call' if 'request_concurrency' in row else 'legacy_field_missing',
                'source': str(source.relative_to(ROOT))})
            if source.parent in (A, B) and row['style'] == 'focused160':
                focused[str(row['query_id'])] = row
    per_call = pd.DataFrame(calls)
    assert len(per_call) == per_call.key.nunique() == 778
    assert len(focused) == 309 and np.isfinite(per_call.historical_request_seconds).all()
    per_call.to_csv(OUT/'per_call.csv', index=False)
    call_summary = per_call.groupby(['dataset', 'style', 'cohort'], sort=False).agg(n=('key', 'size'),
        mean_input_tokens=('input_tokens', 'mean'), mean_output_tokens=('output_tokens', 'mean'),
        estimated_total_usd=('estimated_usd', 'sum'), estimated_mean_usd=('estimated_usd', 'mean'),
        request_p50_seconds=('historical_request_seconds', 'median'),
        request_p95_seconds=('historical_request_seconds', lambda v: float(np.quantile(v, .95))),
        explicit_concurrency_count=('request_concurrency', 'count')).reset_index()
    call_summary.to_csv(OUT/'historical_generation_summary.csv', index=False)
    assert per_call[per_call.cohort == 'remaining249'].request_concurrency.eq(2).all()
    assert abs(per_call.estimated_usd.sum()+.018135-ledger['new_cost_and_reserves_usd']) < 1e-9

    qs = read(S/'selection.json')['queries']
    qids = [q['query_id'] for q in qs]
    pages = [json.loads(line) for line in (A/'data/corpus.jsonl').read_text().splitlines() if line.strip()]
    sources = np.array([page['doc_id'] for page in pages])
    chunkmeta = read(D/'chunks.json')
    load_start = time.perf_counter()
    chunk_vectors = np.concatenate([np.load(path) for path in sorted((D/'chunk_vectors').glob('*.npy'))])
    question_vectors = np.concatenate([np.load(part/'vectors/questions.npy') for part in (A, B)])
    focused_vectors = np.concatenate([np.load(part/'vectors/focused160.npy') for part in (A, B)])
    vector_load_seconds = time.perf_counter()-load_start
    starts = np.array(chunkmeta['page_chunk_starts'])
    expected_ranks = np.load(H/'rankings.npz')
    assert qids == expected_ranks['query_ids'].tolist()
    assert expected_ranks['passage_ids'].tolist() == [str(page['corpus_id']) for page in pages]
    spec = importlib.util.spec_from_file_location('costs_fixed_router', ROOT/'person3_meeting9_constraints/router.py')
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    saved_routes = read(C/'routes.json')
    token_re = re.compile(r'[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*')
    def tokenize(text):
        return token_re.findall(text.lower())
    with threadpool_limits(limits=1):
        build_start = time.perf_counter()
        vectorizer = CountVectorizer(lowercase=True, tokenizer=tokenize, token_pattern=None, dtype=np.float32)
        bm = vectorizer.fit_transform([page['markdown'] for page in pages]).tocsr().astype(np.float32)
        lengths = np.asarray(bm.sum(axis=1)).ravel()
        average = max(float(lengths.mean()), 1e-12)
        frequency = np.asarray((bm > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(pages)-frequency+.5)/(frequency+.5)).astype(np.float32)
        rows = np.repeat(np.arange(len(pages)), np.diff(bm.indptr))
        tf = bm.data
        bm.data = tf*2.2/(tf+1.2*(.25+.75*lengths[rows]/average))*idf[bm.indices]
        bm25_build_seconds = time.perf_counter()-build_start
        pools = threadpool_info()
        def pipeline(index, method):
            whole_start = time.perf_counter()
            tick = time.perf_counter()
            route = router.route(qs[index]['question'])
            allowed = np.flatnonzero(np.isin(sources, route['route_doc_ids'])) if route['route_doc_ids'] else np.arange(len(pages))
            route_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            query_terms = vectorizer.transform([qs[index]['question']]).tocsr()
            query_terms.data[:] = 1
            bm_scores = (query_terms @ bm.T).toarray().ravel()
            sparse_rank = allowed[np.argsort(-bm_scores[allowed], kind='stable')[:100]]
            sparse_seconds = time.perf_counter()-tick
            dense_seconds = mix_seconds = rrf_seconds = 0.
            result = sparse_rank
            if method != 'bm25':
                tick = time.perf_counter()
                vector = question_vectors[index]
                if method == 'focusedmix_bank_rrf_w050':
                    vector = .5*vector+.5*focused_vectors[index]
                    vector = vector/np.maximum(np.linalg.norm(vector), 1e-12)
                mix_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                chunk_scores = chunk_vectors @ vector
                page_scores = np.maximum.reduceat(chunk_scores, starts[:-1])
                dense_rank = allowed[np.argsort(-page_scores[allowed], kind='stable')[:100]]
                dense_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                scores = {}
                for branch in (dense_rank, sparse_rank):
                    for rank, page in enumerate(branch, 1):
                        scores[int(page)] = scores.get(int(page), 0.)+.5/(60+rank)
                result = np.array(sorted(scores, key=lambda p: (-scores[p], p))[:100])
                rrf_seconds = time.perf_counter()-tick
            total = time.perf_counter()-whole_start
            return result, {'routing_filter_seconds': route_seconds, 'bm25_seconds': sparse_seconds,
                'mix_seconds': mix_seconds, 'dense_seconds': dense_seconds, 'rrf_seconds': rrf_seconds,
                'retrieval_total_seconds': total}, route

        for index in range(WARMUP):
            for method in METHODS:
                pipeline(index, method)
        timings, rank_checks = [], 0
        rng = np.random.default_rng(SEED)
        for trial in range(REPEATS):
            for position, index in enumerate(rng.permutation(309)):
                for offset in range(3):
                    method = METHODS[(position+trial+offset) % 3]
                    rank, measured, route = pipeline(int(index), method)
                    assert route['route_doc_ids'] == saved_routes[index]['route_doc_ids']
                    assert np.array_equal(rank, expected_ranks[method][index]), (method, qids[index])
                    rank_checks += 1
                    timings.append({'trial': trial, 'query_id': qids[index], 'cohort': 'initial60' if index < 60 else 'remaining249',
                                    'method': method, **measured})
            print(f'Retrieval trial{trial+1}/{REPEATS}: exact ranks verified', flush=True)
    retrieval_raw = pd.DataFrame(timings)
    retrieval_raw.to_csv(OUT/'retrieval_raw.csv', index=False)

    # Direct snapshot path avoids all hub metadata lookup or cache mutation.
    print('Loading pinned local encoder for profiling only', flush=True)
    load_start = time.perf_counter()
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(SNAPSHOT), local_files_only=True, trust_remote_code=False).to(device='cpu', dtype=torch.float32).eval()
    encoder_import_and_load_seconds = time.perf_counter()-load_start
    def encode(text):
        tokens = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.inference_mode():
            output = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).to(dtype=torch.float32)
            vector = ((output*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1e-12)).numpy()[0]
        return vector/np.maximum(np.linalg.norm(vector), 1e-12)
    for index in range(WARMUP):
        encode(qs[index]['question'])
        encode(focused[qids[index]]['text'])
    encoder_rows, max_vector_error = [], {'question': 0., 'focused_text': 0.}
    for trial in range(REPEATS):
        for index in np.random.default_rng(SEED+trial).permutation(309):
            pairs = [('question', qs[index]['question'], question_vectors[index]),
                     ('focused_text', focused[qids[index]]['text'], focused_vectors[index])]
            if (int(index)+trial) % 2:
                pairs.reverse()
            for kind, text, prior in pairs:
                tick = time.perf_counter()
                vector = encode(text)
                seconds = time.perf_counter()-tick
                difference = float(np.max(np.abs(vector-prior)))
                max_vector_error[kind] = max(max_vector_error[kind], difference)
                assert float(vector @ prior) > .99999
                encoder_rows.append({'trial': trial, 'query_id': qids[index], 'cohort': 'initial60' if index < 60 else 'remaining249',
                                     'kind': kind, 'seconds': seconds, 'max_abs_error_vs_prior': difference})
        print(f'Encoder trial{trial+1}/{REPEATS}: local only; original cache intact', flush=True)
    encoding_raw = pd.DataFrame(encoder_rows)
    encoding_raw.to_csv(OUT/'encoding_raw.csv', index=False)
    assert 'onnxruntime' not in sys.modules
    assert metadata_state(model_root) == cache_before
    assert {path.name: sha(path) for path in SNAPSHOT.iterdir() if path.is_file()} == snapshot_hashes

    # Reduce repeated measurements per query before cohort quantiles and modeled sums.
    stage_columns = [c for c in retrieval_raw if c.endswith('_seconds')]
    retrieval = retrieval_raw.groupby(['query_id', 'cohort', 'method'], sort=False)[stage_columns].median().reset_index()
    encoding = encoding_raw.groupby(['query_id', 'kind'], sort=False).seconds.median().unstack()
    quality = csv(H/'per_query_metrics.csv')
    quality = quality[(quality.k == 10) & quality.method.isin(METHODS)]
    profile = quality.merge(retrieval, on=['query_id', 'cohort', 'method'], validate='one_to_one')
    local_encode, historical_seconds, api_cost, api_input, api_output = [], [], [], [], []
    for row in profile.itertuples():
        is_hyde = row.method == 'focusedmix_bank_rrf_w050'
        qencode = encoding.loc[row.query_id, 'question'] if row.method != 'bm25' else 0.
        hencode = encoding.loc[row.query_id, 'focused_text'] if is_hyde else 0.
        original = focused[row.query_id]
        usage = original['usage']
        local_encode.append(qencode+hencode)
        historical_seconds.append(original['latency_seconds'] if is_hyde else 0.)
        api_input.append(usage['input_tokens'] if is_hyde else 0)
        api_output.append(usage['output_tokens'] if is_hyde else 0)
        api_cost.append((usage['input_tokens']*2.5+usage['output_tokens']*15)/1e6 if is_hyde else 0.)
    profile['encoding_stage_sum_seconds'] = local_encode
    profile['historical_generation_seconds'] = historical_seconds
    profile['estimated_generation_usd_uncached_query'] = api_cost
    profile['api_input_tokens'] = api_input
    profile['api_output_tokens'] = api_output
    profile['incremental_generation_usd_cached_query'] = 0.
    profile['modeled_local_online_seconds'] = profile.retrieval_total_seconds+profile.encoding_stage_sum_seconds
    profile['modeled_uncached_query_seconds'] = profile.modeled_local_online_seconds+profile.historical_generation_seconds
    profile.to_csv(OUT/'per_query_quality_cost_latency.csv', index=False)
    method_rows = []
    stages = []
    for cohort in ('remaining249', 'full309'):
        sub = profile if cohort == 'full309' else profile[profile.cohort == cohort]
        for method, group in sub.groupby('method', sort=False):
            record = {'cohort': cohort, 'method': method, 'n': len(group), 'ndcg10': group.ndcg.mean(),
                'hit10': group.hit.mean(), 'recall10': group.recall.mean(), 'mean_api_input_tokens': group.api_input_tokens.mean(),
                'mean_api_output_tokens': group.api_output_tokens.mean(),
                'mean_estimated_generation_usd_uncached': group.estimated_generation_usd_uncached_query.mean(),
                'incremental_generation_usd_cached': 0.}
            for column in ('retrieval_total_seconds', 'encoding_stage_sum_seconds', 'historical_generation_seconds',
                           'modeled_local_online_seconds', 'modeled_uncached_query_seconds'):
                record.update({column+'_'+key: value for key, value in quantiles(group[column]).items()})
            method_rows.append(record)
            for column in stage_columns:
                stages.append({'cohort': cohort, 'method': method, 'stage': column, **quantiles(group[column])})
    table = pd.DataFrame(method_rows)
    table.to_csv(OUT/'quality_cost_table.csv', index=False)
    pd.DataFrame(stages).to_csv(OUT/'retrieval_stage_summary.csv', index=False)
    encoding_raw.groupby('kind').seconds.agg(n='size', p50='median', p95=lambda v: float(np.quantile(v, .95))).reset_index().to_csv(OUT/'encoding_summary.csv', index=False)

    vector_files = sorted((D/'chunk_vectors').glob('*.npy'))
    storage = {'bm25_build_seconds_this_single_thread_run': bm25_build_seconds,
        'bm25_resident_csr_bytes': int(bm.data.nbytes+bm.indices.nbytes+bm.indptr.nbytes),
        'bm25_vocabulary_utf8_json_bytes': len(json.dumps({term: int(index) for term, index in vectorizer.vocabulary_.items()}, sort_keys=True).encode()),
        'bm25_storage_note': 'CSR component bytes + vocabulary JSON estimate; no persisted new sparse index',
        'dense_vector_resident_bytes': int(chunk_vectors.nbytes), 'dense_vector_npy_files_bytes': sum(p.stat().st_size for p in vector_files),
        'dense_chunk_manifest_bytes': (D/'chunks.json').stat().st_size, 'corpus_jsonl_bytes': (A/'data/corpus.jsonl').stat().st_size,
        'pinned_model_safetensors_bytes': (SNAPSHOT/'model.safetensors').stat().st_size,
        'vector_file_load_seconds_this_run': vector_load_seconds, 'encoder_import_and_load_seconds_this_run': encoder_import_and_load_seconds,
        'historical_document_chunk_encoding_seconds': read(D/'resources.json')['summed_batch_encode_seconds'],
        'historical_document_encoding_conditions': read(D/'execution_backend.json'),
        'index_scope': 'BM25 baseline uses one sparse index; both hybrids share the same sparse+14017x384 dense index. No extra document index for HyDE.',
        'excluded_build_work': 'Historical dense timing excludes model load, chunk preparation and disk writes; do not sum as controlled full index-build timing.'}
    dump(OUT/'index_build_storage.json', storage)
    def sysctl(name):
        result = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    environment = {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
        'machine': platform.machine(), 'cpu_brand': sysctl('machdep.cpu.brand_string'), 'memory_bytes': sysctl('hw.memsize'),
        'logical_cpu_count': os.cpu_count(), 'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
        'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'threadpoolctl_during_retrieval': pools, 'thread_env': {name: os.environ[name] for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        'timing_limit': 'One shared machine/process, warm resident model/index and OS caches; no resource isolation or production SLA.'}
    dump(OUT/'environment.json', environment)
    assert hashes_before == {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    dump(OUT/'validation.json', {'passed': True, 'api_calls': 0, 'new_document_embeddings': 0, 'retrieval_vectors_written': 0,
        'query_encoder_profile_calls': len(encoding_raw)+WARMUP*2, 'query_encoder_timed_calls': len(encoding_raw),
        'timed_retrieval_calls': len(retrieval_raw), 'exact_retained_rank_replay_checks': rank_checks,
        'max_query_profile_vector_errors': max_vector_error, 'legacy_missing_concurrency_calls': int(per_call.request_concurrency.isna().sum()),
        'remaining249_concurrency': 2, 'old_source_hashes_unchanged': True, 'model_cache_metadata_unchanged': True,
        'model_snapshot_hashes': snapshot_hashes, 'old_successful_call_cost_usd': float(per_call.estimated_usd.sum()),
        'old_timeout_reserve_usd': .018135, 'old_ledger_snapshot_cumulative_usd': ledger['cumulative_estimate_usd'],
        'reader_calls_and_cost_excluded_from_this_component': True, 'elapsed_seconds_including_setup': time.perf_counter()-start_all,
        'independent_audit_required': 'Another team member must audit this cost component; this file records producer checks only.'})
    report(table, call_summary, storage)
    for path in OUT.iterdir():
        if path.is_file():
            shutil.copy2(path, EXPORT/path.name)
    print(table.to_string(index=False), flush=True)


def report(table, generation, storage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = table[table.cohort == 'remaining249'].set_index('method').loc[list(METHODS)]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for method, color in zip(METHODS, ('#bc873c', '#458cb1', '#245a81')):
        row = data.loc[method]
        axes[0].scatter(row.mean_estimated_generation_usd_uncached, row.ndcg10, color=color, s=65)
        axes[0].annotate(LABELS[method], (row.mean_estimated_generation_usd_uncached, row.ndcg10),
                         xytext=(7, -12 if method == 'bm25' else 7), textcoords='offset points', fontsize=8)
    axes[0].set(xlabel='Estimated generation USD / uncached query', ylabel='nDCG@10', title='Quality and generation cost')
    axes[0].set_xlim(-.00015, float(data.mean_estimated_generation_usd_uncached.max())*1.8)
    x = np.arange(3)
    axes[1].bar(x-.18, data.retrieval_total_seconds_p50*1000, .36, label='Measured cached-vector retrieval')
    axes[1].bar(x+.18, data.modeled_uncached_query_seconds_p50*1000, .36, label='Modeled uncached query')
    axes[1].set_yscale('log')
    axes[1].set_xticks(x, ['BM25', 'Question hybrid', 'Focused hybrid'])
    axes[1].set(ylabel='Milliseconds, log scale', title='Warm service; generation included only in model')
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('ViDoRe Finance: all remaining249, fixed methods')
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.text(.5, .012, 'One CPU thread; warm model/index. Modeled sums include historical concurrency2 API duration; not measured E2E. Reader excluded.', ha='center', fontsize=8)
    fig.savefig(OUT/'quality_cost_latency.png', dpi=180)
    plt.close(fig)
    lines = ['# Table 4: quality, generation cost and CPU stages', '',
        'All remaining249, including fallback; quality comes from the fixed, previously inspected benchmark. No new API calls. The local profile uses one CPU thread, batch1,20-query warmup,309 queries and three rotated/interleaved repeats. Each query’s repeated times are reduced to a median before cohort p50/p95. Model/index and OS caches are warm.', '',
        '| Method | nDCG@10 | Hit@10 | Recall@10 | Fresh-query generation USD | Cached-vector retrieval p50 / p95 ms | Modeled uncached query p50 / p95 s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        row = data.loc[method]
        lines.append(f'| {LABELS[method]} | {row.ndcg10:.4f} | {row.hit10:.1%} | {row.recall10:.4f} | {row.mean_estimated_generation_usd_uncached:.6f} | {row.retrieval_total_seconds_p50*1000:.3f} / {row.retrieval_total_seconds_p95*1000:.3f} | {row.modeled_uncached_query_seconds_p50:.4f} / {row.modeled_uncached_query_seconds_p95:.4f} |')
    lines += ['', '![Quality, generation cost and timing](quality_cost_latency.png)', '',
        '**Measured:** cached-vector retrieval includes literal routing/filter creation, BM25 query scoring/allowed-page sorting, full chunk-max dense scoring where needed, and fixed RRF. It recomputes retrieval rather than serving saved scores/ranks. Encoder tokenization/forward/pooling are measured separately from a pinned local snapshot, with no network or old-cache writes. Per-stage raw times and all309 secondary aggregates accompany this table.', '',
        '**Modeled:** per-query median retrieval + per-query median question encoding (+ retained Focused-text encoding for Focused), plus that query’s previously observed Focused generation duration. Quantiles are computed after these per-query sums; p50/p95 values are not added. This is an uncached-query estimate for a warm resident service, not measured E2E or a process cold start. It excludes external queueing/scheduling, reader time, model/index load and disk I/O. Historical remaining249 generation ran at recorded concurrency2; initial60 concurrency is mixed and96 legacy call records omit the field. Missing fields stay unknown. No generation is repeated.', '',
        'Generation USD uses the retained experiment rates USD2.50/M input and USD15/M output, with cached discounts ignored; these are estimates, not billing or current-price claims. BM25/question hybrid need no generation. Focused’s saved passage has a historical creation cost; incremental API cost is zero when that passage/vector is cached. `per_call.csv` retains all778 prior successful calls, including unused Short and ablation calls, without charging them to the three deployed paths. Reader calls/costs are excluded for root integration.', '',
        f"One-off/index context: current single-thread BM25 build **{storage['bm25_build_seconds_this_single_thread_run']:.4f}s**; resident CSR **{storage['bm25_resident_csr_bytes']:,}bytes**, plus vocabulary JSON estimate **{storage['bm25_vocabulary_utf8_json_bytes']:,}bytes**. Shared dense vectors **{storage['dense_vector_resident_bytes']:,}bytes** (both hybrids use the same document index). Prior document chunk encoding **{storage['historical_document_chunk_encoding_seconds']:.3f}s** used8 threads and excludes load/chunk preparation/disk writes; it is separately sourced, not a controlled build comparison. Current vector-file load and encoder import/load times are in `index_build_storage.json`; model weights are common to both hybrid encoders.", '',
        'These are shared-machine stage measurements, not a production SLA or end-to-end RAG benchmark. All old input/model-cache hashes and metadata remain unchanged. Another agent must independently audit this component.', '',
        'Reproduce offline from repository root: `.venv/bin/python person3_meeting9_followup/costs/run.py`. Only this component’s new directories are written.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
