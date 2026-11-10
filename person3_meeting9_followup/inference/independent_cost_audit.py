"""Independent arithmetic/source audit of retained costs; never profiles or calls models."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'results/person3_meeting9_followup/costs_v1'
OUT=ROOT/'results/person3_meeting9_followup/cost_audit_v1'
EXPORT=ROOT.parent/'outputs/meeting9_followup_20261109/cost_audit'
A=ROOT/'results/person3_meeting9_transfer/vidore_finance_v1'
B=ROOT/'results/person3_meeting9_transfer/vidore_finance_remaining249_v1'
H=ROOT/'results/person3_meeting9_constrained_hybrid/v1'
D=ROOT/'results/person3_meeting9_fullpage/v1'
S=ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
METHODS=['bm25','question_bank_rrf_w050','focusedmix_bank_rrf_w050']
STAGES=['routing_filter_seconds','bm25_seconds','mix_seconds','dense_seconds','rrf_seconds','retrieval_total_seconds']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(p):return pd.read_csv(p,dtype={'query_id':str})
def near(a,b,tol=1e-12):assert np.allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def quantiles(a):return float(np.median(a)),float(np.quantile(a,.95))

OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
producer_before={p.name:sha(p) for p in C.iterdir() if p.is_file()}
source_hashes=read(C/'source_hashes.json')
for rel,h in source_hashes.items():assert sha(ROOT/rel)==h,rel
ledger=read(ROOT/'results/person3_meeting9_workflow/v1/cost_summary.json')
protocol=read(C/'protocol.json');validation=read(C/'validation.json');rates=ledger['prices_per_million']
assert rates==protocol['rates_per_million']=={'input':2.5,'output':15}
assert protocol['cpu_threads']==1 and protocol['repeats']==3 and protocol['warmup_queries_per_method']==20
assert protocol['no_api_calls'] and protocol['no_new_document_embeddings']

original_calls=[];focused={}
for source,cohort in [(A/'successful_api_outputs.json','initial60'),(B/'successful_api_outputs.json','remaining249'),(ROOT/'results/person3_meeting9_ablation/factorial20_v1/successful_api_outputs.json','ablation40')]:
    for row in read(source):
        u=row['usage'];assert u['total_tokens']==u['input_tokens']+u['output_tokens']
        usd=(u['input_tokens']*rates['input']+u['output_tokens']*rates['output'])/1e6
        original_calls.append({'key':row['key'],'dataset':row['dataset'],'style':row['style'],'query_id':str(row['query_id']),'cohort':cohort,
          'model':row['model_returned'],'input_tokens':u['input_tokens'],'output_tokens':u['output_tokens'],'total_tokens':u['total_tokens'],
          'cached_input_tokens':u.get('input_tokens_details',{}).get('cached_tokens',0),'estimated_usd':usd,
          'historical_request_seconds':row['latency_seconds'],'request_concurrency':row.get('request_concurrency'),
          'concurrency_provenance':'explicit_per_call' if 'request_concurrency' in row else 'legacy_field_missing','source':str(source.relative_to(ROOT))})
        if cohort!='ablation40' and row['style']=='focused160':focused[str(row['query_id'])]=row
calls=pd.DataFrame(original_calls);saved_calls=csv(C/'per_call.csv');assert len(calls)==len(saved_calls)==778 and calls.key.nunique()==778 and len(focused)==309
joined=calls.merge(saved_calls,on='key',suffixes=('','_saved'),validate='one_to_one')
for col in ['dataset','style','query_id','cohort','model','concurrency_provenance','source']:assert (joined[col]==joined[col+'_saved']).all()
for col in ['input_tokens','output_tokens','total_tokens','cached_input_tokens','estimated_usd','historical_request_seconds','request_concurrency']:near(joined[col],joined[col+'_saved'])
assert calls[calls.cohort=='remaining249'].request_concurrency.eq(2).all()
assert calls.request_concurrency.isna().sum()==256 and calls[calls.cohort=='initial60'].request_concurrency.isna().sum()==96
near(calls.estimated_usd.sum(),1.218915);near(calls.estimated_usd.sum()+.018135,ledger['new_cost_and_reserves_usd'])
near(ledger['prior_cumulative_estimate_usd']+ledger['new_cost_and_reserves_usd'],ledger['cumulative_estimate_usd'])
generation_summary=csv(C/'historical_generation_summary.csv')
for r in generation_summary.to_dict('records'):
    g=calls[(calls.dataset==r['dataset'])&(calls['style']==r['style'])&(calls.cohort==r['cohort'])];assert len(g)==r['n']
    expected={'mean_input_tokens':g.input_tokens.mean(),'mean_output_tokens':g.output_tokens.mean(),'estimated_total_usd':g.estimated_usd.sum(),
      'estimated_mean_usd':g.estimated_usd.mean(),'request_p50_seconds':np.median(g.historical_request_seconds),'request_p95_seconds':np.quantile(g.historical_request_seconds,.95),'explicit_concurrency_count':g.request_concurrency.count()}
    for col,v in expected.items():near(r[col],v)

qids=[q['query_id'] for q in read(S/'selection.json')['queries']]
retrieval=csv(C/'retrieval_raw.csv');encoding=csv(C/'encoding_raw.csv')
assert len(retrieval)==2781 and len(encoding)==1854
assert not retrieval.duplicated(['trial','query_id','method']).any() and not encoding.duplicated(['trial','query_id','kind']).any()
assert set(retrieval.trial)==set(encoding.trial)=={0,1,2}
for trial in range(3):
    for method in METHODS:assert set(retrieval[(retrieval.trial==trial)&(retrieval.method==method)].query_id)==set(qids)
    for kind in ['question','focused_text']:assert set(encoding[(encoding.trial==trial)&(encoding.kind==kind)].query_id)==set(qids)
assert np.isfinite(retrieval[STAGES].to_numpy()).all() and (retrieval[STAGES]>=0).all().all()
assert np.isfinite(encoding.seconds).all() and (encoding.seconds>0).all()
assert (retrieval.retrieval_total_seconds+1e-12>=retrieval[STAGES[:-1]].sum(axis=1)).all()
# Independently verify the retained interleaving schedule without running any pipeline.
rng=np.random.default_rng(20261109);expected_order=[]
for trial in range(3):
    for position,index in enumerate(rng.permutation(309)):
        for offset in range(3):expected_order.append((trial,qids[int(index)],METHODS[(position+trial+offset)%3]))
assert list(retrieval[['trial','query_id','method']].itertuples(index=False,name=None))==expected_order
expected_encoding=[]
for trial in range(3):
    for index in np.random.default_rng(20261109+trial).permutation(309):
        kinds=['question','focused_text']
        if (int(index)+trial)%2:kinds.reverse()
        expected_encoding += [(trial,qids[int(index)],kind) for kind in kinds]
assert list(encoding[['trial','query_id','kind']].itertuples(index=False,name=None))==expected_encoding
for kind in ['question','focused_text']:
    near(encoding[encoding.kind==kind].max_abs_error_vs_prior.max(),validation['max_query_profile_vector_errors'][kind])
    assert encoding[encoding.kind==kind].max_abs_error_vs_prior.max()<1e-5
retrieval_medians=retrieval.groupby(['query_id','cohort','method'])[STAGES].median()
encoding_medians=encoding.groupby(['query_id','kind']).seconds.median().unstack()
quality=csv(H/'per_query_metrics.csv');quality=quality[(quality.k==10)&quality.method.isin(METHODS)]
profile=csv(C/'per_query_quality_cost_latency.csv');assert len(profile)==len(quality)==927
quality_join=profile.merge(quality,on=['query_id','method','k'],suffixes=('','_source'),validate='one_to_one')
for col in ['hit','recall','mrr','ndcg']:near(quality_join[col],quality_join[col+'_source'])
assert (quality_join.cohort==quality_join.cohort_source).all()
reconstructed=[]
for r in quality.to_dict('records'):
    key=(r['query_id'],r['cohort'],r['method']);stage=retrieval_medians.loc[key];is_hyde=r['method']=='focusedmix_bank_rrf_w050'
    enc=(encoding_medians.loc[r['query_id'],'question'] if r['method']!='bm25' else 0.)+(encoding_medians.loc[r['query_id'],'focused_text'] if is_hyde else 0.)
    original=focused[r['query_id']];u=original['usage'];api_seconds=original['latency_seconds'] if is_hyde else 0.
    usd=(u['input_tokens']*rates['input']+u['output_tokens']*rates['output'])/1e6 if is_hyde else 0.
    reconstructed.append({**{col:r[col] for col in ['query_id','cohort','method','hit','recall','mrr','ndcg']},**stage.to_dict(),
      'encoding_stage_sum_seconds':enc,'historical_generation_seconds':api_seconds,'estimated_generation_usd_uncached_query':usd,
      'api_input_tokens':u['input_tokens'] if is_hyde else 0,'api_output_tokens':u['output_tokens'] if is_hyde else 0,'incremental_generation_usd_cached_query':0.,
      'modeled_local_online_seconds':stage.retrieval_total_seconds+enc,'modeled_uncached_query_seconds':stage.retrieval_total_seconds+enc+api_seconds})
reconstructed=pd.DataFrame(reconstructed);comparison=reconstructed.merge(profile,on=['query_id','cohort','method'],suffixes=('','_saved'),validate='one_to_one')
for col in reconstructed.columns:
    if col not in ['query_id','cohort','method']:near(comparison[col],comparison[col+'_saved'])
reconstructed.to_csv(OUT/'reconstructed_per_query.csv',index=False)

table=[]
for cohort in ['remaining249','full309']:
    sub=reconstructed if cohort=='full309' else reconstructed[reconstructed.cohort==cohort]
    for method,g in sub.groupby('method',sort=False):
        row={'cohort':cohort,'method':method,'n':len(g),'ndcg10':g.ndcg.mean(),'hit10':g.hit.mean(),'recall10':g.recall.mean(),
          'mean_api_input_tokens':g.api_input_tokens.mean(),'mean_api_output_tokens':g.api_output_tokens.mean(),
          'mean_estimated_generation_usd_uncached':g.estimated_generation_usd_uncached_query.mean(),'incremental_generation_usd_cached':0.}
        for col in ['retrieval_total_seconds','encoding_stage_sum_seconds','historical_generation_seconds','modeled_local_online_seconds','modeled_uncached_query_seconds']:
            p50,p95=quantiles(g[col]);row[col+'_p50']=p50;row[col+'_p95']=p95
        table.append(row)
rebuilt=pd.DataFrame(table);saved=csv(C/'quality_cost_table.csv');joined=rebuilt.merge(saved,on=['cohort','method'],suffixes=('','_saved'),validate='one_to_one');assert len(joined)==6
for col in rebuilt.columns:
    if col not in ['cohort','method']:near(joined[col],joined[col+'_saved'])
rebuilt.to_csv(OUT/'reconstructed_table.csv',index=False)
stage_summary=csv(C/'retrieval_stage_summary.csv')
for r in stage_summary.to_dict('records'):
    g=reconstructed[reconstructed.method==r['method']]
    if r['cohort']!='full309':g=g[g.cohort==r['cohort']]
    p50,p95=quantiles(g[r['stage']]);near(r['p50'],p50);near(r['p95'],p95)
encoding_summary=csv(C/'encoding_summary.csv')
for r in encoding_summary.to_dict('records'):
    g=encoding[encoding.kind==r['kind']];assert len(g)==r['n']==927
    p50,p95=quantiles(g.seconds);near(r['p50'],p50);near(r['p95'],p95)

storage=read(C/'index_build_storage.json');vector_files=sorted((D/'chunk_vectors').glob('*.npy'))
assert storage['dense_vector_resident_bytes']==sum(np.load(p,mmap_mode='r').nbytes for p in vector_files)
assert storage['dense_vector_npy_files_bytes']==sum(p.stat().st_size for p in vector_files)
near(storage['historical_document_chunk_encoding_seconds'],read(D/'resources.json')['summed_batch_encode_seconds'])
snapshot=ROOT.parent/'model_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
for name,h in validation['model_snapshot_hashes'].items():assert sha(snapshot/name)==h
assert storage['pinned_model_safetensors_bytes']==(snapshot/'model.safetensors').stat().st_size
env=read(C/'environment.json');assert env['torch_threads']==env['torch_interop_threads']==1
assert all(pool['num_threads']==1 for pool in env['threadpoolctl_during_retrieval'])
readme=(C/'README.md').read_text();assert '**Measured:**' in readme and '**Modeled:**' in readme and 'not measured E2E' in readme and 'Reader calls/costs are excluded' in readme
assert producer_before=={p.name:sha(p) for p in C.iterdir() if p.is_file()}
notes=['encoding_summary.csv intentionally summarizes raw repeats (927 observations per kind), not the309 query-median distribution; use the main table/per-query profile for cohort latency.',
       'There are96 missing-concurrency initial60 records and256 across all778 successful records including160 ablation calls; missing remains unknown.',
       'Modeled uncached durations are per-query stage sums plus historical API duration, not measured E2E; quantiles are taken after summing.',
       'Current model-content hashes match the producer record. Historical before/after metadata equality is supported by producer assertions; the old metadata snapshot was not retained for independent reconstruction.',
       'No rerun was performed: clock validity, per-trial rank assertions and fresh encoder outputs are inspected via retained raw data/code, not independently reprofiled.',
       'Sparse CSR/vocabulary byte totals remain producer measurements; dense file sizes and retained historical encoding duration were independently checked.']
summary={'passed':True,'audit_scope':'Independent source/usage/quality/arithmetic/reduction audit; no timing/model rerun','api_calls':0,'encoder_calls':0,
  'successful_call_records_reconciled':778,'quality_rows_reconciled':927,'retrieval_raw_rows':2781,'encoding_raw_rows':1854,
  'rebuilt_table_rows':6,'stage_summary_rows_verified':len(stage_summary),'generation_summary_rows_verified':len(generation_summary),
  'per_query_median_then_addition_then_quantiles_verified':True,'original_cost_usage_and_concurrency_verified':True,
  'quality_exact_source_verified':True,'modeled_vs_measured_labels_verified':True,'producer_files_unchanged':True,
  'successful_call_cost_usd':float(calls.estimated_usd.sum()),'timeout_reserve_usd':.018135,'historical_cumulative_estimate_usd':ledger['cumulative_estimate_usd'],
  'reader_costs_excluded_from_component':True,'notes':notes,'audit_code_sha256':sha(__file__)}
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'source_hashes.json').write_text(json.dumps({str(C/p):h for p,h in producer_before.items()},indent=2)+'\n')
(OUT/'README.md').write_text('# Independent cost audit\n\nPassed:778 successful API usage records,927 quality/profile rows,2781 retrieval timings,1854 encoder timings,6 method/cohort table rows and all summaries reconcile independently. No API/model/profiling calls were made; producer files remain unchanged.\n\n'+'\n\n'.join(notes)+'\n')
for p in OUT.iterdir():
    if p.is_file():(EXPORT/p.name).write_bytes(p.read_bytes())
print(json.dumps(summary,indent=2))
