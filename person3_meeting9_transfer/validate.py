"""Independent saved-output checks, full rank replay, and graded-metric reconstruction."""
from pathlib import Path
import importlib.util,json,math,hashlib,argparse
import numpy as np
import pandas as pd
s=importlib.util.spec_from_file_location('transfer_run',Path(__file__).with_name('run.py'));r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
a=argparse.ArgumentParser();a.add_argument('--cohort',choices=['pilot60','remaining249'],default='pilot60');args=a.parse_args();r.configure(args.cohort)
manifest=r.read(r.DATA/'manifest.json')
for path,digest in manifest['files'].items():assert r.sha((r.BASE_OUT/path).read_bytes())==digest
selection=r.read(r.OUT/'selection.json');allq=pd.read_parquet(r.DATA/'queries.parquet');q=allq[allq.language=='english'].to_dict('records');ordered=sorted(q,key=lambda x:hashlib.sha256((r.SEED+'|'+str(x['query_id'])).encode()).hexdigest());expected=ordered[:60] if r.N==60 else ordered[60:];assert [x['query_id'] for x in selection['queries']]==[str(x['query_id']) for x in expected]
corpus,_,_=r.sources();assert len(corpus)==2942 and len({x['corpus_id'] for x in corpus})==2942
rel=r.read(r.OUT/'qrels.json');qr=pd.read_parquet(r.DATA/'qrels.parquet')
for qid,gold in rel.items():
 selected=qr[qr.query_id==int(qid)];ind={str(pid):int(g.score.max()) for pid,g in selected.groupby('corpus_id')};assert gold==ind
selected_qrels=qr[qr.query_id.isin([int(x) for x in rel])]
assert set(selected_qrels.corpus_id).issubset({c['corpus_id'] for c in corpus})
assert len({(c['doc_id'],c['page_number_in_doc']) for c in corpus})==2942
r.dump(r.OUT/'data_validation.json',{'passed':True,'selected_queries':r.N,'selected_qrels_rows':len(selected_qrels),'qrels_all_page_ids_exist':True,'grade_distribution':{str(k):int(v) for k,v in selected_qrels.score.value_counts().items()},'source_document_page_pairs_unique':True,'all_page_images_available':all(c['image_available'] for c in corpus),'image_ocr_alignment':'Official image and markdown fields in same corpus row; pinned revision and nontruncation checked at download; all corpus IDs contiguous0..2941. No OCR semantic correctness or visualquality audit claimed.'})
z=np.load(r.OUT/'rankings.npz');assert len(z['query_ids'])==r.N and len(z['passage_ids'])==2942
scores=pd.read_csv(r.OUT/'per_query_metrics.csv',dtype={'query_id':str});checked=0
for a in scores.to_dict('records'):
 i=z['query_ids'].tolist().index(a['query_id']);ranking=[str(x) for x in z['passage_ids'][z[a['method']][i,:a['k']]]];gold=rel[a['query_id']];hits=[j+1 for j,pid in enumerate(ranking) if gold.get(pid,0)>0]
 dcg=sum(gold.get(pid,0)/math.log2(j+2) for j,pid in enumerate(ranking));idcg=sum(v/math.log2(j+2) for j,v in enumerate(sorted(gold.values(),reverse=True)[:a['k']]))
 expected={'hit':int(bool(hits)),'recall':len(hits)/sum(v>0 for v in gold.values()),'precision':len(hits)/a['k'],'mrr':1/min(hits) if hits else 0,'ndcg':dcg/idcg}
 for key,value in expected.items():assert abs(a[key]-value)<1e-12
 checked+=1
for _,g in scores.groupby(['query_id','method']):
 g=g.sort_values('k');assert (g.hit.diff().dropna()>=0).all() and (g.recall.diff().dropna()>=-1e-12).all()
ce=np.concatenate([np.load(r.CORPUS_VECTORS/f'corpus_{i:04d}.npy') for i in range(0,2942,128)]);qe=np.load(r.OUT/'vectors/questions.npy');replayed=0;near_ties=[]
for name in ['question_dense','focused160_pure_dense','focused160_mix_dense','short_30_50_pure_dense','short_30_50_mix_dense']:
 if name=='question_dense':vectors=qe
 else:
  style=name.replace('_pure_dense','').replace('_mix_dense','');he=np.load(r.OUT/'vectors'/(style+'.npy'));vectors=he if name.endswith('_pure_dense') else r.h.normalize(.5*qe+.5*he)
 for i,v in enumerate(vectors):
  similarity=(ce.astype(np.float64)*v.astype(np.float64)).sum(axis=1);new=np.argsort(-similarity,kind='stable')[:100];old=z[name][i]
  if not np.array_equal(new,old):
   assert all(abs(similarity[a]-similarity[b])<=1e-6 for a,b in zip(new,old));near_ties.append({'query_id':str(z['query_ids'][i]),'method':name,'max_score_difference':max(float(abs(similarity[a]-similarity[b])) for a,b in zip(new,old))})
  replayed+=1
bm=r.h.BM25([x['markdown'] for x in corpus]);qs=selection['queries']
for i,q in enumerate(qs):assert np.array_equal(bm.rank(q['question']),z['bm25'][i])
tests=pd.read_csv(r.OUT/'primary_tests.csv')
wide=scores[scores.k==10].pivot(index='query_id',columns='method',values='ndcg')
for test in tests.to_dict('records'):
 delta=(wide[test['candidate']]-wide[test['control']]).to_numpy();rng=np.random.default_rng(20261109);extreme=0;observed=abs(delta.sum())
 for _ in range(100):
  signs=rng.integers(0,2,size=(1000,len(delta)),dtype=np.int8)*2-1
  extreme+=int((np.abs(np.sum(signs*delta[None,:],axis=1))>=observed-1e-12).sum())
 assert abs(test['p_raw']-(extreme+1)/100001)<1e-12
order=np.argsort(tests.p_raw.to_numpy(),kind='stable');adj=np.minimum(1,np.maximum.accumulate((len(tests)-np.arange(len(tests)))*tests.p_raw.to_numpy()[order]));assert np.max(np.abs(tests.p_holm.to_numpy()[order]-adj))<1e-12
r.dump(r.OUT/'validation.json',{'passed':True,'source_hashes':True,'selection_recomputed':True,'qrels_recomputed':True,'independent_metric_rows':checked,'float64_dense_ranking_replays':replayed,'float32_near_ties':near_ties,'bm25_ranking_replays':r.N,'permutation_tests_recomputed':4,'holm_corrections':4,'api_calls':0,'visual_evaluation':False})
print('Validated',checked,'metricrows;',replayed,f'denserankings;{r.N}BM25;4Holm')
