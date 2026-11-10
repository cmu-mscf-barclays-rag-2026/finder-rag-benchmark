"""Frozen text-only HyDE transfer on ViDoRe V3 Finance EN. No paid calls here."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE_OUT=ROOT/'results/person3_meeting9_transfer/vidore_finance_v1';OUT=BASE_OUT;DATA=BASE_OUT/'data';EXPORT=ROOT.parent/'outputs/meeting9_20261109/transfer';CORPUS_VECTORS=BASE_OUT/'vectors';N=60;COHORT='pilot60'
s=importlib.util.spec_from_file_location('transfer_helpers',ROOT/'person3_cross_dataset/run.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h);h.OUT=OUT
s=importlib.util.spec_from_file_location('pilot_templates',ROOT/'person3_hyde_pilot/run.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
MODEL='gpt-5.4-2026-03-05';REV='7f432c176d82e27546501ad8064a713ac3071809';SEED='meeting9-vidore-finance-transfer-v1-20261109';KS=[5,10,15,20,25]
STYLES={'focused160':json.loads((p.CROSS/'protocol.json').read_text())['profiles']['focused160']['prompts']['fiqa'],'short_30_50':p.SHORT}
PAIRS=[('short_30_50_mix_dense','question_dense'),('focused160_mix_dense','question_dense'),('short_30_50_mix_dense','focused160_mix_dense'),('short_30_50_pure_dense','focused160_pure_dense')]
def read(path):return json.loads(path.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(path)
def frozen(path,x):
 x=json.loads(json.dumps(x)) # Compare the persisted JSON representation (tuples become arrays).
 if path.exists():assert read(path)==x,'Frozen artifact mismatch: '+str(path)
 else:dump(path,x)
def jsonl(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def sources():
 corpus=jsonl(DATA/'corpus.jsonl');queries=pd.read_parquet(DATA/'queries.parquet').to_dict('records');qrels=pd.read_parquet(DATA/'qrels.parquet')
 return corpus,queries,qrels

def configure(cohort):
 global OUT,EXPORT,N,COHORT
 COHORT=cohort
 if cohort=='remaining249':
  OUT=ROOT/'results/person3_meeting9_transfer/vidore_finance_remaining249_v1';EXPORT=ROOT.parent/'outputs/meeting9_20261109/transfer/remaining249';N=249
 h.OUT=OUT

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);corpus,allq,qr=sources();qs=[q for q in allq if q['language']=='english'];assert len(qs)==309
 ordered_all=sorted(qs,key=lambda q:sha((SEED+'|'+str(q['query_id'])).encode()))
 ordered=ordered_all[:60] if COHORT=='pilot60' else ordered_all[60:]
 assert len(ordered)==N
 if COHORT!='pilot60':assert not ({str(q['query_id']) for q in ordered}&{q['query_id'] for q in read(BASE_OUT/'selection.json')['queries']})
 selection=[{'query_id':str(q['query_id']),'question':q['query'],'language':q['language'],'query_types':list(q['query_types']),'content_type':list(q['content_type'])} for q in ordered]
 frozen(OUT/'selection.json',{'seed':SEED,'n':N,'english_queries_available':309,'selection':('first60 ascending SHA256(seed|query_id), no outcome-based selection' if COHORT=='pilot60' else 'remaining249 after frozen first60 ascending SHA256(seed|query_id), no outcome-based selection'),'queries':selection})
 jobs=[{'dataset':'vidore_finance_en','query_id':q['query_id'],'style':style,'prompt':template.format(domain='finance',q=q['question']),'model':MODEL,'max_output_tokens':160,'temperature':0,'reasoning':'none','store':False} for q in selection for style,template in STYLES.items()]
 pending=OUT/'pending_calls.jsonl';text=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs)
 if pending.exists():assert pending.read_text()==text
 else:pending.write_text(text)
 frozen(OUT/'protocol.json',{'dataset':'vidore/vidore_v3_finance_en','dataset_revision':REV,'queries':N,'available_english_queries':309,'full_corpus_pages':2942,'document_representation':'provided OCR markdown, entirepage ranking unit; dense first256 MiniLM tokens; BM25 fullOCR','model':MODEL,'encoder':h.EMBED,'encoder_revision':h.REV,'encoder_max_length':256,'encoder_device':'offline CPU float32','pooling':'attentionmask mean + L2','mix_question_weight':.5,'generation_cap':160,'temperature':0,'reasoning':'none','store':False,'tools':[],'api_retries':0,'prompt_templates':STYLES,'prompt_origin':'Exact previous FiQA focused160 and Short templates, frozen without adapting to outcomes','retrieval_scope':'all2942pages, never only positivepages','top_k':KS,'primary_metric':'nDCG@10, linear graded gains score/log2(rank+1)','primary_comparisons':PAIRS,'primary_test':'two-sided paired signflip,100000draws,seed20261109; Holm across4','secondary':'other methods/cutoffs/subgroups descriptive, no significance cherry-picking','bm25':{'k1':1.2,'b':.75,'query':'originalquestion','binary_query_counts':True},'tie_break':'stablecorpusindex','new_calls':2*N,'scope':f'new dataset transfer of previously selected prompts; custom{N}-query text-only subset, not official MTEB leaderboard reproduction; no visualmodel comparison or finalanswer evaluation','data_leakage_control':'generator sees onlyquestion, never answers/sourcepages/qrels; promptparameters never tuned on theseoutcomes'})
 frozen(OUT/'source_hashes.json',{'data_manifest_sha256':sha((DATA/'manifest.json').read_bytes()),'prompt_source_protocol_sha256':sha((p.CROSS/'protocol.json').read_bytes()),'prompt_source_code_sha256':sha((ROOT/'person3_hyde_pilot/run.py').read_bytes())})
 rel={q['query_id']:{} for q in selection}
 for row in qr.to_dict('records'):
  qid=str(row['query_id']);pid=str(row['corpus_id']);grade=int(row['score'])
  if qid in rel:rel[qid][pid]=max(grade,rel[qid].get(pid,0))
 assert all(any(v>0 for v in r.values()) for r in rel.values())
 frozen(OUT/'qrels.json',rel)
 print(f'Frozen{N}englishqueries,2942pages,{2*N}generationjobs',flush=True)

def baseline():
 qs=read(OUT/'selection.json')['queries'];corpus,_,_=sources();texts=[c['markdown'] for c in corpus]
 vecdir=OUT/'vectors';vecdir.mkdir(exist_ok=True)
 e=h.encoder()
 if not (OUT/'page_token_audit.json').exists():
  lengths=e.tokenizer(texts,truncation=False,add_special_tokens=True,return_length=True,verbose=False)['length']
  dump(OUT/'page_token_audit.json',{'pages':len(lengths),'encoder_max_length':256,'pages_over_256':int(sum(x>256 for x in lengths)),'fraction_over_256':float(np.mean(np.array(lengths)>256)),'mean_tokens':float(np.mean(lengths)),'median_tokens':float(np.median(lengths)),'max_tokens':int(max(lengths)),'note':'Full supplied page OCR lengths including special tokens; over256 content is truncated for dense encoding, not BM25'})
 timings=[]
 for i in range(0,len(texts),128):
  path=CORPUS_VECTORS/f'corpus_{i:04d}.npy'
  if not path.exists():
   start=time.perf_counter();v=e.encode(texts[i:i+128]);np.save(path,v);timings.append({'kind':'corpus','start':i,'n':len(v),'seconds':time.perf_counter()-start});print('Encoded',i+len(v),'/2942',flush=True)
 if not (vecdir/'questions.npy').exists():
  start=time.perf_counter();np.save(vecdir/'questions.npy',e.encode([q['question'] for q in qs]));timings.append({'kind':'queries','n':len(qs),'seconds':time.perf_counter()-start})
 ce=np.concatenate([np.load(CORPUS_VECTORS/f'corpus_{i:04d}.npy') for i in range(0,len(texts),128)]);qe=np.load(vecdir/'questions.npy');assert ce.shape==(2942,384) and qe.shape==(N,384)
 e.encode([qs[0]['question']]) # warmup excluded from perquery encoding benchmark
 encode_times=[]
 for q,v in zip(qs,qe):
  start=time.perf_counter();fresh=e.encode([q['question']])[0];elapsed=time.perf_counter()-start
  assert np.max(np.abs(fresh-v))<1e-5
  encode_times.append({'query_id':q['query_id'],'seconds':elapsed})
 start=time.perf_counter();bm=h.BM25(texts);build=time.perf_counter()-start
 qrank=[];brank=[];querytime=[]
 for q,v in zip(qs,qe):
  start=time.perf_counter();qrank.append(h.top(v@ce.T));dense=time.perf_counter()-start
  start=time.perf_counter();brank.append(bm.rank(q['question']));sparse=time.perf_counter()-start
  querytime.append({'query_id':q['query_id'],'dense_search_seconds':dense,'bm25_search_seconds':sparse})
 np.savez_compressed(OUT/'baseline_rankings.npz',query_ids=np.array([q['query_id'] for q in qs]),passage_ids=np.array([str(c['corpus_id']) for c in corpus]),question_dense=np.stack(qrank),bm25=np.stack(brank))
 if timings:dump(OUT/'encoding_timings.json',timings)
 dump(OUT/'retrieval_timings.json',{'bm25_build_seconds':build,'single_query_encode':encode_times,'single_query_encode_p50_seconds':float(np.median([x['seconds'] for x in encode_times])),'single_query_encode_p95_seconds':float(np.quantile([x['seconds'] for x in encode_times],.95)),'dense_search_p50_seconds':float(np.median([x['dense_search_seconds'] for x in querytime])),'dense_search_p95_seconds':float(np.quantile([x['dense_search_seconds'] for x in querytime],.95)),'bm25_search_p50_seconds':float(np.median([x['bm25_search_seconds'] for x in querytime])),'bm25_search_p95_seconds':float(np.quantile([x['bm25_search_seconds'] for x in querytime],.95)),'queries':querytime,'note':'Local singleprocess walltimes, not network end-to-end production latency; concurrentjobs/hardware load can affect values'})
 print('Baseline retrieval complete',flush=True)

def metric(rank,rels,k):
 g=np.array([rels.get(str(x),0) for x in rank[:k]],dtype=float);positive=g>0;ideal=sorted(rels.values(),reverse=True)[:k];idcg=sum(v/np.log2(i+2) for i,v in enumerate(ideal));where=np.flatnonzero(positive)
 return {'hit':float(bool(len(where))),'recall':float(positive.sum()/sum(x>0 for x in rels.values())),'precision':float(positive.sum()/k),'mrr':1/(where[0]+1) if len(where) else 0.,'ndcg':float(sum(v/np.log2(i+2) for i,v in enumerate(g))/idcg) if idcg else 0.}

def evaluate():
 corpus,_,_=sources();qs=read(OUT/'selection.json')['queries'];rels=read(OUT/'qrels.json');z=np.load(OUT/'baseline_rankings.npz');ranks={k:z[k] for k in ['question_dense','bm25']};ids=z['passage_ids'].tolist();qid=[q['query_id'] for q in qs];assert qid==z['query_ids'].tolist()
 outputs=read(OUT/'successful_api_outputs.json');lookup={(o['query_id'],o['style']):o for o in outputs};assert len(lookup)==len(outputs)==2*N
 e=h.encoder();ce=np.concatenate([np.load(CORPUS_VECTORS/f'corpus_{i:04d}.npy') for i in range(0,2942,128)]);qe=np.load(OUT/'vectors/questions.npy');audit=[]
 for style in STYLES:
  oo=[lookup[q['query_id'],style] for q in qs]
  for q,o in zip(qs,oo):
   assert o['dataset']=='vidore_finance_en' and o['model_returned']==MODEL and o['prompt']==STYLES[style].format(domain='finance',q=q['question']) and o['text']
   audit.append({'query_id':q['query_id'],'style':style,'word_count':len(o['text'].split()),'status':o['status'],'input_tokens':o['usage']['input_tokens'],'output_tokens':o['usage']['output_tokens'],'latency_seconds':o.get('latency_seconds')})
  path=OUT/'vectors'/(style+'.npy')
  if not path.exists():np.save(path,e.encode([o['text'] for o in oo]))
  he=np.load(path);assert he.shape==(N,384)
  for route,vec in [('pure_dense',he),('mix_dense',h.normalize(.5*qe+.5*he))]:ranks[style+'_'+route]=np.stack([h.top(v@ce.T) for v in vec])
 np.savez_compressed(OUT/'rankings.npz',query_ids=np.array(qid),passage_ids=np.array(ids),**ranks)
 rows=[]
 for method,rr in ranks.items():
  for q,rank in zip(qs,rr):
   assert len(set(rank))==100 and min(rank)>=0 and max(rank)<2942
   for k in KS:rows.append({'query_id':q['query_id'],'method':method,'k':k,**metric([ids[i] for i in rank],rels[q['query_id']],k)})
 f=pd.DataFrame(rows);f.to_csv(OUT/'per_query_metrics.csv',index=False);summary=f.groupby(['method','k']).agg(n=('query_id','size'),**{m:(m,'mean') for m in ['hit','recall','precision','mrr','ndcg']}).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 wide=f[f.k==10].pivot(index='query_id',columns='method',values='ndcg');tests=[]
 for a,b in PAIRS:
  d=(wide[a]-wide[b]).to_numpy();rng=np.random.default_rng(20261109);observed=abs(d.sum());count=0
  for _ in range(100):count+=int((np.abs((rng.integers(0,2,size=(1000,len(d)),dtype=np.int8)*2-1)@d)>=observed-1e-12).sum())
  tests.append({'candidate':a,'control':b,'n':N,'mean_delta':float(d.mean()),'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'p_raw':(count+1)/100001})
 t=pd.DataFrame(tests);order=np.argsort(t.p_raw.to_numpy(),kind='stable');corrected=np.minimum(1,np.maximum.accumulate((len(t)-np.arange(len(t)))*t.p_raw.to_numpy()[order]));t['p_holm']=0.;t.loc[order,'p_holm']=corrected;t['significant_holm']=t.p_holm<.05;t.to_csv(OUT/'primary_tests.csv',index=False)
 pd.DataFrame(audit).to_csv(OUT/'generation_audit.csv',index=False)
 groups=[]
 for typ in ['Text','Table','Chart']:
  selected={q['query_id'] for q in qs if typ in q['content_type']}
  for m,g in f[(f.k==10)&f.query_id.isin(selected)].groupby('method'):groups.append({'content_type':typ,'method':m,'n':len(g),'ndcg10':g.ndcg.mean(),'hit10':g.hit.mean()})
 pd.DataFrame(groups).to_csv(OUT/'content_type_descriptive.csv',index=False)
 dump(OUT/'completion_status.json',{'complete':True,'queries':N,'corpus_pages':2942,'generation_outputs':2*N,'metric_rows':len(f),'methods':len(ranks),'visual_model_evaluated':False,'answer_reader_evaluated':False})
 report()
 print(f'Complete{N}queries/{2*N}outputs/6methods',flush=True)

def report():
 EXPORT.mkdir(parents=True,exist_ok=True);cohort_flag=' --cohort remaining249' if COHORT=='remaining249' else '';summary=pd.read_csv(OUT/'summary.csv');t=pd.read_csv(OUT/'primary_tests.csv')
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 labels={'question_dense':'Question dense','bm25':'BM25','focused160_pure_dense':'Focused: pure HyDE','focused160_mix_dense':'Focused: 50/50 mix','short_30_50_pure_dense':'Short: pure HyDE','short_30_50_mix_dense':'Short: 50/50 mix'}
 order=list(labels);vals=summary[summary.k==10].set_index('method').loc[order];fig,axes=plt.subplots(1,2,figsize=(11,4.8));colors=['#8b929a','#b4bbc3','#a6c8de','#5395c4','#94d2b3','#27956c']
 for ax,col,title in [(axes[0],'ndcg','nDCG@10'),(axes[1],'hit','Hit@10')]:
  ax.barh([labels[m] for m in order],vals[col],color=colors);ax.invert_yaxis();ax.set_xlim(0,max(.05,vals[col].max()*1.23));ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  for i,v in enumerate(vals[col]):ax.text(v+.003,i,f'{v:.3f}',va='center',fontsize=9)
 fig.suptitle(f'ViDoRe V3 Finance EN: frozen prompt transfer ({N} queries / 2,942 pages)');fig.tight_layout(rect=[0,.06,1,.94]);fig.text(.5,.018,'Text-only OCR retrieval, pinned MiniLM; custom query subset. No visual-model or final-answer evaluation.',ha='center',fontsize=9);fig.savefig(EXPORT/'transfer_results.png',dpi=180);fig.savefig(EXPORT/'transfer_results.svg');plt.close(fig)
 rows=['# Frozen HyDE transfer: ViDoRe V3 Finance EN','',f'{N} English queries selected by fixed hash; all 2,942 financial-report pages searched. Exact previous FiQA Focused and Short prompts were frozen before generation.','','| Method | nDCG@10 | Hit@5 | Hit@10 | Recall@10 | MRR@10 |','|---|---:|---:|---:|---:|---:|']
 for m in order:
  g=summary[summary.method==m].set_index('k');rows.append(f'| {labels[m]} | {g.loc[10,"ndcg"]:.4f} | {g.loc[5,"hit"]:.1%} | {g.loc[10,"hit"]:.1%} | {g.loc[10,"recall"]:.4f} | {g.loc[10,"mrr"]:.4f} |')
 rows+=['','| Method | Hit@5 | Hit@10 | Hit@15 | Hit@20 | Hit@25 |','|---|---:|---:|---:|---:|---:|']
 for m in order:
  g=summary[summary.method==m].set_index('k');rows.append('| '+labels[m]+' | '+' | '.join(f'{g.loc[k,"hit"]:.1%}' for k in KS)+' |')
 rows+=['','![Transfer results](transfer_results.png)','','| Planned comparison | Delta nDCG@10 | Holm p (4 tests) |','|---|---:|---:|']
 for r in t.to_dict('records'):rows.append(f'| {labels[r["candidate"]]} vs {labels[r["control"]]} | {r["mean_delta"]:+.4f} | {r["p_holm"]:.5f} |')
 rows+=['','Two-sided paired sign-flip tests, 100,000 draws, seed 20261109; Holm over the four predeclared comparisons. Other cutoffs and nonexclusive query content-type groups are descriptive.','','The corpus includes page images, OCR markdown, graded relevance judgments, bounding boxes and reference answers from six banks’ annual reports. This run uses supplied OCR only. Dense representations keep the first 256 MiniLM tokens of each whole page; BM25 uses full OCR. Tables and long pages can therefore expose text extraction and truncation limitations. These are custom subset results, not an official MTEB leaderboard reproduction.','','## Reproduce','','From repository root, using saved generations:','','```bash',f'.venv/bin/python person3_meeting9_transfer/run.py baseline{cohort_flag}',f'.venv/bin/python person3_meeting9_transfer/run.py evaluate{cohort_flag}',f'.venv/bin/python person3_meeting9_transfer/validate.py{cohort_flag}','```','','The runner makes no API calls. Do not delete successful_api_outputs.json. Downloading and preparing data uses download.py then run.py prepare; pinned source manifests and OCR are retained locally.','','Sources: [official dataset](https://huggingface.co/datasets/vidore/vidore_v3_finance_en), [MTEB benchmarks](https://docs.mteb.org/overview/available_benchmarks/).']
 if COHORT=='remaining249':rows+=['','This remaining 249-query stage was frozen before the 60-query HyDE retrieval results were read. Its four primary comparisons form a separate Holm family. Combined 309-query results are descriptive only.']
 rows+=['','Question-level paired tests do not adjust for shared annual-report sources. Page-level relevance and reference answers do not establish fact-level evidence sufficiency.','','Graded nDCG uses raw qrels grades, matching [MTEB retrieval evaluation](https://github.com/embeddings-benchmark/mteb/blob/61ad84cc5463c5b7ac047e5a6cb66b56d5eb0735/mteb/_evaluators/retrieval_metrics.py) and [NIST ndcg_cut gains](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c). Exact source snapshots and hashes are retained in data/metric_sources/.','','Dense page truncation audit: 2,764/2,942 pages (93.95%) exceed 256 encoder tokens; median full page length is 822 tokens. HyDE comparisons within the dense route share this representation. BM25 has full OCR access, so cross-route differences also reflect input coverage.']
 text='\n'.join(rows)+'\n';(EXPORT/'README.md').write_text(text);(OUT/'findings.md').write_text(text)
 import shutil
 for name in ['summary.csv','primary_tests.csv','content_type_descriptive.csv','generation_audit.csv','protocol.json','selection.json','completion_status.json','page_token_audit.json','retrieval_timings.json']:
  shutil.copy2(OUT/name,EXPORT/name)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','baseline','evaluate','report']);a.add_argument('--cohort',choices=['pilot60','remaining249'],default='pilot60');args=a.parse_args();configure(args.cohort);globals()[args.stage]()
