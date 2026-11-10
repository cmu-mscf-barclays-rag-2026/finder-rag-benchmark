"""Independent saved-result audit; no network, model, embeddings or paid API.
Run from any cwd. Missing cohort results remain explicitly pending.
"""
from pathlib import Path
import json,hashlib,math,ast
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];BASE=R/'results/person3_meeting9_transfer/vidore_finance_v1';D=BASE/'data';OTHER=BASE.with_name('vidore_finance_remaining249_v1');REV='7f432c176d82e27546501ad8064a713ac3071809';SEED='meeting9-vidore-finance-transfer-v1-20261109'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((D/'manifest.json').read_text());assert manifest['revision']==REV
for rel,h in manifest['files'].items():assert sha(BASE/rel)==h
assert json.loads((D/'repository_metadata.json').read_text())['sha']==REV
corpus=[json.loads(x) for x in (D/'corpus.jsonl').read_text().splitlines()];assert [x['corpus_id'] for x in corpus]==list(range(2942))
blocks=[json.loads(p.read_text()) for p in sorted((D/'ocr_blocks').glob('*.json'))];assert len(blocks)==30 and all(b['upstream_revision']==REV for b in blocks)
assert [r for b in blocks for r in b['rows']]==corpus
assert all(x['image_available'] and x['image_width']>0 and x['image_height']>0 for x in corpus)
meta=pd.read_parquet(D/'documents_metadata.parquet');assert len(meta)==6
for r in meta.to_dict('records'):
 pages=[x['page_number_in_doc'] for x in corpus if x['doc_id']==r['doc_id']];assert sorted(pages)==list(range(r['page_number']))
assert len({(x['doc_id'],x['page_number_in_doc']) for x in corpus})==2942
queries=pd.read_parquet(D/'queries.parquet');qr=pd.read_parquet(D/'qrels.parquet');english=queries[queries.language.eq('english')];assert len(english)==309 and english.query_id.nunique()==309
assert not qr.duplicated(['query_id','corpus_id']).any() and set(qr.corpus_id)<=set(range(2942)) and set(qr.score)=={1,2}
ordered=sorted(english.to_dict('records'),key=lambda q:hashlib.sha256((SEED+'|'+str(q['query_id'])).encode()).hexdigest())
s60=json.loads((BASE/'selection.json').read_text());s249=json.loads((OTHER/'selection.json').read_text());assert s60['seed']==s249['seed']==SEED
assert [q['query_id'] for q in s60['queries']]==[str(q['query_id']) for q in ordered[:60]]
assert [q['query_id'] for q in s249['queries']]==[str(q['query_id']) for q in ordered[60:]]
ids60={q['query_id'] for q in s60['queries']};ids249={q['query_id'] for q in s249['queries']};assert not ids60&ids249 and len(ids60|ids249)==309
qlookup={str(q['query_id']):q for q in english.to_dict('records')}
for q in s60['queries']+s249['queries']:assert q['question']==qlookup[q['query_id']]['query']
blank={x['corpus_id'] for x in corpus if not x['markdown'].strip()};blank_rels=qr[qr.query_id.isin(english.query_id)&qr.corpus_id.isin(blank)];assert len(blank_rels)==0
prot=json.loads((BASE/'protocol.json').read_text());old=json.loads((R/'results/person3_cross_dataset/gpt54_focused160_v1/protocol.json').read_text());assert prot['prompt_templates']['focused160']==old['profiles']['focused160']['prompts']['fiqa']
module=ast.parse((R/'person3_hyde_pilot/run.py').read_text());short=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SHORT' for t in n.targets));assert prot['prompt_templates']['short_30_50']==short
sourcehash=json.loads((BASE/'source_hashes.json').read_text());assert sourcehash['data_manifest_sha256']==sha(D/'manifest.json');assert sourcehash['prompt_source_protocol_sha256']==sha(R/'results/person3_cross_dataset/gpt54_focused160_v1/protocol.json');assert sourcehash['prompt_source_code_sha256']==sha(R/'person3_hyde_pilot/run.py')
def audit_cohort(path,n):
 required=['rankings.npz','per_query_metrics.csv','primary_tests.csv','summary.csv','protocol.json','selection.json','successful_api_outputs.json','qrels.json']
 if not all((path/x).exists() for x in required):return {'status':'pending','missing':[x for x in required if not (path/x).exists()]}
 hashes={x:sha(path/x) for x in required};p=json.loads((path/'protocol.json').read_text());assert p['dataset_revision']==REV and p['prompt_templates']==prot['prompt_templates']
 sel=json.loads((path/'selection.json').read_text())['queries'];z=np.load(path/'rankings.npz');ids=z['passage_ids'].tolist();qid=z['query_ids'].tolist();assert len(sel)==n and qid==[q['query_id'] for q in sel];assert ids==[str(i) for i in range(2942)]
 official={x:{} for x in qid}
 for r in qr.to_dict('records'):
  q=str(r['query_id'])
  if q in official:official[q][str(r['corpus_id'])]=int(r['score'])
 assert official==json.loads((path/'qrels.json').read_text())
 f=pd.read_csv(path/'per_query_metrics.csv',dtype={'query_id':str});assert len(f)==n*6*5 and not f.duplicated(['query_id','method','k']).any();lookup=f.set_index(['query_id','method','k']);calc={};methods=[m for m in z.files if m not in ['query_ids','passage_ids']];assert len(methods)==6
 for m in methods:
  assert z[m].shape==(n,100)
  for q,ranks in zip(qid,z[m]):
   assert len(set(ranks))==100 and all(0<=i<2942 for i in ranks)
   rel=official[q];ideal=sorted(rel.values(),reverse=True)
   for k in p['top_k']:
    gains=[rel.get(ids[int(i)],0) for i in ranks[:k]];where=[j+1 for j,g in enumerate(gains) if g>0];dcg=sum(g/math.log2(j+2) for j,g in enumerate(gains));idcg=sum(g/math.log2(j+2) for j,g in enumerate(ideal[:k]));values={'hit':int(bool(where)),'recall':len(where)/len(rel),'precision':len(where)/k,'mrr':1/min(where) if where else 0.,'ndcg':dcg/idcg}
    row=lookup.loc[(q,m,k)]
    for metric,v in values.items():assert math.isclose(v,row[metric],abs_tol=1e-13),(q,m,k,metric,v,row[metric])
    if k==10:calc[q,m]=values['ndcg']
 summary=pd.read_csv(path/'summary.csv')
 for r in summary.to_dict('records'):
  g=f[f.method.eq(r['method'])&f.k.eq(r['k'])];assert len(g)==r['n']==n
  for col in ['hit','recall','precision','mrr','ndcg']:assert math.isclose(g[col].mean(),r[col],abs_tol=1e-13)
 tests=pd.read_csv(path/'primary_tests.csv');assert len(tests)==4;raw=[];details=[];sortedqid=sorted(qid)
 for (a,b),r in zip(p['primary_comparisons'],tests.to_dict('records')):
  assert (r['candidate'],r['control'])==(a,b)
  d=np.array([calc[q,a]-calc[q,b] for q in sortedqid]);obs=abs(d.sum());rng=np.random.default_rng(20261109);extreme=0
  for st in range(0,100000,5000):
   signs=rng.integers(0,2,size=(5000,n),dtype=np.int8)*2-1;totals=np.sum(signs*d[None,:],axis=1);extreme+=int(np.count_nonzero(abs(totals)>=obs-1e-12))
  pv=(extreme+1)/100001;assert math.isclose(pv,r['p_raw'],abs_tol=1e-14),(a,b,pv,r['p_raw']);assert math.isclose(d.mean(),r['mean_delta'],abs_tol=1e-13)
  wins=int(np.count_nonzero(d>1e-12));loss=int(np.count_nonzero(d<-1e-12));ties=n-wins-loss;assert (wins,loss,ties)==(r['wins'],r['losses'],r['ties'])
  raw.append(pv);details.append({'candidate':a,'control':b,'n':n,'mean_delta_recomputed':float(d.mean()),'wins':wins,'losses':loss,'ties':ties,'extreme_draws':extreme,'p_raw_recomputed':pv})
 order=np.argsort(raw,kind='stable');adjusted=np.empty(4);running=0.
 for rank,i in enumerate(order):running=max(running,(4-rank)*raw[i]);adjusted[i]=min(1.,running)
 assert np.allclose(adjusted,tests.p_holm,atol=1e-14)
 for d,a in zip(details,adjusted):d['p_holm_recomputed']=float(a)
 outs=json.loads((path/'successful_api_outputs.json').read_text());assert len(outs)==2*n and len({(x['query_id'],x['style']) for x in outs})==2*n and len({x['response_id'] for x in outs})==2*n
 qs={x['query_id']:x for x in sel}
 for o in outs:
  assert o['prompt']==p['prompt_templates'][o['style']].format(domain='finance',q=qs[o['query_id']]['question']);assert o['model_returned']==p['model'] and o['status']=='completed' and o['text'].strip()
 markdown_checked=False
 if (path/'findings.md').exists():
  md=(path/'findings.md').read_text();labels={'question_dense':'Question dense','bm25':'BM25','focused160_pure_dense':'Focused: pure HyDE','focused160_mix_dense':'Focused: 50/50 mix','short_30_50_pure_dense':'Short: pure HyDE','short_30_50_mix_dense':'Short: 50/50 mix'}
  for method,label in labels.items():
   g=summary[summary.method.eq(method)].set_index('k')
   expected=f'| {label} | {g.loc[10,"ndcg"]:.4f} | {g.loc[5,"hit"]:.1%} | {g.loc[10,"hit"]:.1%} | {g.loc[10,"recall"]:.4f} | {g.loc[10,"mrr"]:.4f} |';assert expected in md
  for row in tests.to_dict('records'):
   expected=f'| {labels[row["candidate"]]} vs {labels[row["control"]]} | {row["mean_delta"]:+.4f} | {row["p_holm"]:.5f} |';assert expected in md
  markdown_checked=True
 assert all(sha(path/x)==h for x,h in hashes.items())
 return {'status':'passed','README_tables_match_CSV_when_present':markdown_checked,'queries':n,'corpus_pages':len(ids),'methods':len(methods),'metric_rows_independently_recomputed':len(f),'generation_outputs':len(outs),'raw_p_tests':4,'holm_family_size':4,'significant_holm':int(sum(adjusted<.05)),'input_sha256':hashes,'query_order_for_MC':'lexicographic string query_id order, matching saved per-query pivot','tests':details,'all_summary_means_match':True,'generations_match_frozen_prompts_and_snapshot':True}
def audit_full309():
 path=BASE.with_name('vidore_finance_full309_summary_v1')
 required=['selection.json','qrels.json','per_query_metrics.csv','summary.csv','separate_primary_families.csv','rankings.npz','generation_audit.csv','validation.json','completion_status.json']
 if not all((path/x).exists() for x in required):return {'status':'pending','expected_queries':309,'missing':[x for x in required if not (path/x).exists()],'inference':'descriptive only; do not create an extra significance family'}
 hashes={x:sha(path/x) for x in required}
 mergedsel=json.loads((path/'selection.json').read_text());assert mergedsel['n']==309 and mergedsel['queries']==s60['queries']+s249['queries']
 a=pd.read_csv(BASE/'per_query_metrics.csv',dtype={'query_id':str});b=pd.read_csv(OTHER/'per_query_metrics.csv',dtype={'query_id':str});f=pd.read_csv(path/'per_query_metrics.csv',dtype={'query_id':str});combined=pd.concat([a,b],ignore_index=True)
 assert len(f)==len(combined)==309*6*5 and not f.duplicated(['query_id','method','k']).any()
 pd.testing.assert_frame_equal(f,combined,check_exact=False,atol=1e-13,rtol=1e-13)
 summary=pd.read_csv(path/'summary.csv')
 for row in summary.to_dict('records'):
  g=f[f.method.eq(row['method'])&f.k.eq(row['k'])];assert len(g)==row['n']==309
  for metric in ['hit','recall','precision','mrr','ndcg']:assert math.isclose(g[metric].mean(),row[metric],abs_tol=1e-13)
 z=np.load(path/'rankings.npz');za=np.load(BASE/'rankings.npz');zb=np.load(OTHER/'rankings.npz')
 assert z.files==za.files==zb.files and np.array_equal(z['passage_ids'],za['passage_ids']) and np.array_equal(z['passage_ids'],zb['passage_ids'])
 for method in [x for x in z.files if x!='passage_ids']:assert np.array_equal(z[method],np.concatenate([za[method],zb[method]]))
 assert set(z['query_ids'].tolist())=={str(x) for x in english.query_id} and len(set(z['query_ids']))==309
 official={str(q):{} for q in english.query_id}
 for row in qr.to_dict('records'):
  if str(row['query_id']) in official:official[str(row['query_id'])][str(row['corpus_id'])]=int(row['score'])
 assert json.loads((path/'qrels.json').read_text())==official
 families=pd.read_csv(path/'separate_primary_families.csv');assert len(families)==8 and set(families.stage)=={'initial60','remaining249'}
 for stage,source in [('initial60',BASE),('remaining249',OTHER)]:
  g=families[families.stage.eq(stage)].drop(columns=['stage']).reset_index(drop=True);original=pd.read_csv(source/'primary_tests.csv');pd.testing.assert_frame_equal(g,original,check_exact=False,atol=1e-13,rtol=1e-13)
 audit=pd.read_csv(path/'generation_audit.csv',dtype={'query_id':str});assert len(audit)==618 and not audit.duplicated(['query_id','style']).any()
 outputs=json.loads((BASE/'successful_api_outputs.json').read_text())+json.loads((OTHER/'successful_api_outputs.json').read_text());assert len({o['response_id'] for o in outputs})==618
 validation=json.loads((path/'validation.json').read_text());assert validation['passed'] and validation['queries']==309 and validation['full309_significance_tests']==0
 for relative,digest in validation['source_sha256'].items():assert sha(R/relative)==digest
 assert json.loads((path/'completion_status.json').read_text())['complete']
 assert all(sha(path/x)==digest for x,digest in hashes.items())
 return {'status':'passed','queries':309,'corpus_pages':2942,'metric_rows':len(f),'six_methods_and_five_depths':True,'selection_exactly_concatenates_disjoint_frozen_stages':True,'all_official_English_query_ids_covered':True,'qrels_equal_official_English_subset':True,'rankings_exactly_concatenate_two_stages':True,'metrics_exactly_concatenate_two_stages':True,'all_summary_means_independently_recomputed':True,'unique_successful_generations':618,'separate_primary_tests':8,'families_retained':['initial60 Holm4','remaining249 Holm4'],'new_full309_significance_tests':0,'source_hashes_verified':True,'input_sha256':hashes,'inference':'Descriptive full-split aggregate only.'}

cohorts={'pilot60':audit_cohort(BASE,60),'remaining249':audit_cohort(OTHER,249)}
result={'data_integrity_passed':True,'dataset':manifest['dataset'],'revision':REV,'pinned_file_sha256_verified':manifest['files'],'corpus_pages':2942,'source_documents':6,'metadata_page_counts_and_contiguous_indices_verified':True,'source_page_order':'Corpus IDs follow upstream lexical page-number order, not numeric page order; explicit corpus_id/doc_id/page_number mapping verified.','OCR_rows_equal_saved_revision_blocks':True,'saved_repository_metadata_revision_matches':True,'page_image_metadata':{'available_flags_and_positive_dimensions':True,'images_downloaded_or_visually_audited':False,'limit':'Original image URLs/raw HTTP bodies not retained in OCR blocks; pinned image-URL validation is recorded by download code and provenance, not independently replayed remotely in this offline audit.'},'english_queries':309,'pilot60_and_remaining249_disjoint':True,'union_covers_all_309_english_queries':True,'selection_matches_independent_SHA256_order':True,'question_text_matches_official_queries':True,'blank_OCR_pages':len(blank),'english_qrels_on_blank_OCR_pages':len(blank_rels),'focused_and_short_prompts_match_previous_frozen_sources':True,'cohorts':cohorts,'full309_aggregate':audit_full309(),'statistic':'linear graded nDCG uses score/log2(rank+1), not exponential gain; all saved ranking metrics recomputed independently','MC_signflip':'two-sided absolute paired sum, 100000 pseudorandom +/- draws, seed20261109 reset per contrast, int8 RNG, 1e-12 comparison tolerance, plus-one p=(extreme+1)/100001; all query pairs retained, including zero differences','limitations':['Independent local audit of saved data and results; no network, embeddings, reader or paid API executed.','Page-level labels are not fact-level evidence-sufficiency annotations.','Question-level tests do not cluster-adjust the six shared annual-report sources.','Text-only OCR transfer, dense first256tokens versus BM25 fullOCR; not a visual model or official MTEB leaderboard reproduction.']}
out=R.parent/'outputs/meeting9_20261109/independent_transfer_audit.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
print('cohorts',[(k,v['status']) for k,v in cohorts.items()]);print('full309',result['full309_aggregate']['status']);print(out)
