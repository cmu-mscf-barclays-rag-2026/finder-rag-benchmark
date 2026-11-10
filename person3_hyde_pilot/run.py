"""Isolated, resumable 20 x 3 dense-only pilot. Never invokes full-run stages."""
from pathlib import Path
import argparse, collections, csv, hashlib, importlib.util, json, os, sqlite3, sys, time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
LEGAL=ROOT/'results/person3_openai_hyde'
CROSS=ROOT/'results/person3_cross_dataset/gpt54_focused160_v1'
OUT=ROOT/'results/person3_hyde_pilot/focused_weight_v1'
CACHE=ROOT/'data/external_benchmarks/pilot_cache_v1'
SEED='hyde-prompt-weight-pilot-v1-20261030'
MODEL='gpt-5.4-2026-03-05'
METHODS=['question_dense','current_mix50','current_mix25','short_mix50']
QUOTAS={'Single-Document QA':7,'Multi-Document QA':5,'Long In-context Learning':3,'Code Repository Understanding':2,'Long-dialogue History Understanding':2,'Long Structured Data Understanding':1}
SHORT="Write one hypothetical evidence paragraph of 30 to 50 words to help retrieve passages for the question below about {domain}. Preserve the question's named entities, quantities, jurisdiction, constraints, and any comparison between alternatives. Do not add entities, conditions, assumptions, citations, or conclusions not supported by the question. Stay on the specific information needed; avoid generic background or broad advice. Do not answer with an invented factual result. No headings.\nQuestion: {q}\nHypothetical evidence paragraph:"
DOMAINS={'legal':'Victorian criminal law','fiqa':'finance','longbench2':'the given task'}
spec=importlib.util.spec_from_file_location('historical_helpers',ROOT/'person3_cross_dataset/run.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
# Redirect every mutable helper path before invoking any helper.
h.OUT=OUT;h.CACHE=CACHE

def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def jsonl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def frozen(p,obj):
 if p.exists():assert read(p)==obj,f'Frozen file changed: {p}'
 else:dump(p,obj)
def select_hash(q):return sha((SEED+'|'+q['query_id']).encode())
def sources():
 legal=pd.read_csv(LEGAL/'evaluation_queries.csv',dtype=str,keep_default_na=False).rename(columns={'id':'query_id'}).to_dict('records')
 fiqa=jsonl(CROSS/'fiqa_queries.jsonl')
 lb=h.load_lb()
 lb=[q|{'query_id':str(q['_id'])} for q in lb]
 return {'legal':legal,'fiqa':fiqa,'longbench2':lb}
def selected():return read(OUT/'selection.json')['datasets']
def historical():
 db=sqlite3.connect(f'file:{CROSS}/api_checkpoint.sqlite?mode=ro',uri=True)
 out={k:json.loads(p) for k,p in db.execute("SELECT key,payload FROM calls WHERE status='saved'")};db.close();return out

def verify_sources():
 for item in read(CROSS/'protocol.json')['dataset_sources']:
  assert h.sha(h.DATA/item['name'])==item['sha256'],item['name']
 for name,digest in read(LEGAL/'protocol.json')['data_hashes'].items():
  assert h.sha(ROOT/'data/legal_rag_bench'/name)==digest,name
 index=pd.read_csv(LEGAL/'passage_index.csv',dtype=str)
 raw=jsonl(ROOT/'data/legal_rag_bench/corpus.jsonl')
 assert [str(x['id']) for x in raw]==index.passage_id.tolist()
 assert len(raw)==4876

def prepare():
 verify_sources()
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 allq=sources();ds={}
 for name,rows in allq.items():
  assert len({q['query_id'] for q in rows})==len(rows)
  qs=sorted(rows,key=select_hash)[:20] if name!='longbench2' else [q for domain,n in sorted(QUOTAS.items()) for q in sorted([q for q in rows if q['domain']==domain],key=select_hash)[:n]]
  ds[name]=[q|{'selection_hash':select_hash(q),'source_index':rows.index(q)} for q in qs];assert len(qs)==20
 frozen(OUT/'selection.json',{'seed':SEED,'datasets':ds,'source_revisions':{'legal':read(LEGAL/'protocol.json')['data_revision'],'cross':read(CROSS/'protocol.json')['dataset_sources']}})
 protocol={'model':MODEL,'api':{'reasoning':{'effort':'none'},'temperature':0,'store':False,'max_retries':0,'tools':[]},'conditions':{'question_dense':{'w':1},'current_mix50':{'w':.5,'generation':'focused160'},'current_mix25':{'w':.75,'generation':'focused160'},'short_mix50':{'w':.5,'generation':'short'}},'short_prompt':SHORT,'domains':DOMAINS,'generation_cap':160,'reader_cap':32,'reader_prompt':h.READER,'embedding_model':h.EMBED,'embedding_revision':h.REV,'embedding_max_length':256,'chunk_tokens':224,'overlap':32,'topk_reader':5,'depth':100,'stable_ties':'corpus index','budget_guard_usd':2.,'lost_attempt_allowance_usd':.076705,'prices':{'input':2.5,'output':15},'pricing_checked':'2026-11-02','pricing_source':'https://developers.openai.com/api/docs/models/gpt-5.4','metrics':{'legal':'Hit@5','fiqa':'nDCG@10','longbench2':'MC accuracy'},'scope':'exploratory subset of previously evaluated benchmarks; not held out','source_commit':'a857f9abe669b770e45dcbdaddd2c758a18dd66a'}
 frozen(OUT/'protocol.json',protocol)
 outputs=historical();lg={o['query_id']:o for o in jsonl(LEGAL/'generated_passages.jsonl') if o['profile']=='focused160'}
 prompts=[]
 for name,qs in ds.items():
  for q in qs:
   old=lg[q['query_id']] if name=='legal' else outputs[f"hyde::{name}::{q['query_id']}"]
   template=read(LEGAL/'protocol.json')['profiles']['focused160']['prompt'] if name=='legal' else read(CROSS/'protocol.json')['profiles']['focused160']['prompts'][name]
   assert old['prompt']==template.format(q=q['question']) and old['model_returned']==MODEL and old['text']
   prompts.append({'dataset':name,'query_id':q['query_id'],'current':old,'short_prompt':SHORT.format(domain=DOMAINS[name],q=q['question'])})
 frozen(OUT/'prompts.json',prompts)
 import importlib.metadata,platform
 dump(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','torch','transformers','openai','scipy','scikit-learn','pyarrow','ijson','httpx','matplotlib','huggingface-hub']}})
 dump(OUT/'code_hashes.json',{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [Path(__file__),ROOT/'person3_cross_dataset/run.py']})
 print('Selection frozen:',{k:len(v) for k,v in ds.items()},'LongBench lengths:',dict(collections.Counter(q['length'] for q in ds['longbench2'])),flush=True)
 baseline_report()

def baseline_report():
 ds=selected();rows=[]
 for name,qs in ds.items():
  if name=='legal':f=pd.read_csv(LEGAL/'per_query_metrics.csv',dtype={'query_id':str});metric='hit5'
  elif name=='fiqa':f=pd.read_csv(CROSS/'fiqa_per_query_metrics.csv',dtype={'query_id':str});f=f[f.k==10];metric='ndcg'
  else:f=pd.read_csv(CROSS/'longbench2_predictions.csv',dtype={'query_id':str});metric='correct'
  f=f[f.query_id.isin([q['query_id'] for q in qs])]
  for method,old in [('question_dense','question_dense'),('current_mix50','focused160_mix_dense')]:
   for r in f[f.method==old].to_dict('records'):rows.append({'dataset':name,'query_id':r['query_id'],'method':method,'value':float(r[metric]),'source':'historical_subset; dense replay pending'})
 pd.DataFrame(rows).to_csv(OUT/'historical_subset_per_query.csv',index=False)
 pd.DataFrame(rows).groupby(['dataset','method']).agg(value=('value','mean'),n=('value','size')).to_csv(OUT/'historical_subset_summary.csv')

def load_vectors(ds,e):
 qs=selected()[ds];idx=[q['source_index'] for q in qs]
 if ds=='legal':
  z=np.load(LEGAL/'embeddings.npz');qe=z['questions'][idx]
  p=CACHE/'legal_current_hyde.npy'
  if p.exists():he=np.load(p)
  else:
   texts={x['query_id']:x['current']['text'] for x in read(OUT/'prompts.json') if x['dataset']==ds}
   he=e.encode([texts[q['query_id']] for q in qs]);np.save(p,he)
 else:
  z=np.load(CROSS/f'{ds}_query_embeddings.npz');qe=z['questions'][idx];he=z['hyde'][idx]
 return qe,he

def compatibility(e):
 rows=[]
 for ds,qs in selected().items():
  qe,he=load_vectors(ds,e);check=e.encode([q['question'] for q in qs]);delta=float(np.max(np.abs(check-qe)))
  assert delta<=1e-5,(ds,'question coordinate mismatch',delta)
  r={'dataset':ds,'question_max_abs_diff':delta,'n':20}
  if ds!='legal':
   texts={x['query_id']:x['current']['text'] for x in read(OUT/'prompts.json') if x['dataset']==ds}
   delta=float(np.max(np.abs(e.encode([texts[q['query_id']] for q in qs])-he)))
   assert delta<=1e-5,(ds,'hyde mismatch',delta);r['hyde_max_abs_diff']=delta
  rows.append(r)
 dump(OUT/'encoder_compatibility.json',rows);print('Encoder compatibility passed',rows,flush=True)

def resumable_vectors(e,texts,folder):
 folder.mkdir(parents=True,exist_ok=True);manifest={'count':len(texts),'text_sha256':sha(json.dumps(texts,ensure_ascii=False).encode()),'embedding_revision':h.REV}
 frozen(folder/'manifest.json',manifest)
 blocks=[]
 for start in range(0,len(texts),512):
  p=folder/f'{start:07d}.npy';batch=texts[start:start+512]
  if not p.exists():
   v=e.encode(batch);tmp=p.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(p)
   print('Encoded',folder.name,min(start+512,len(texts)),'/',len(texts),flush=True)
  v=np.load(p);assert v.shape==(len(batch),384) and v.dtype==np.float32 and np.isfinite(v).all();blocks.append(v)
 return np.concatenate(blocks)

def corpus(ds,q,e):
 if ds=='legal':return np.load(LEGAL/'embeddings.npz')['corpus'],None
 if ds=='fiqa':
  f=pd.read_parquet(h.DATA/'fiqa_corpus.parquet').to_dict('records');assert len(f)==57638
  texts=[' '.join([str(x.get('title','')),str(x['text'])]).strip() for x in f]
  return resumable_vectors(e,texts,CACHE/'fiqa_corpus'),[str(x['_id']) for x in f]
 text=h.context_for(q);assert sha(text.encode())==q['context_sha256']
 folder=CACHE/'selected_contexts'/q['context_sha256'];folder.mkdir(parents=True,exist_ok=True)
 if (folder/'chunks.json').exists():ch=read(folder/'chunks.json')
 else:
  chunks,spans=h.chunks_for(e,text);ch={'chunks':chunks,'spans':spans};dump(folder/'chunks.json',ch)
 for s,(a,b) in zip(ch['chunks'],ch['spans']):assert text[a:b]==s
 old=read(CROSS/'longbench2_retrieval'/f"{q['query_id']}.json")
 assert len(ch['chunks'])==old['n_chunks']
 for method in ['question_dense','focused160_mix_dense']:
  for c in old['methods'][method]['top5']:
   assert ch['chunks'][c['chunk_id']]==c['text'] and ch['spans'][c['chunk_id']]==c['span']
 return resumable_vectors(e,ch['chunks'],folder),ch

def validate_rank(new,old,scores):
 if np.array_equal(new,old):return {'exact':True}
 # Same float32 score groups; no arbitrary rank tolerance.
 assert len(new)==len(old) and len(set(old))==len(old)
 assert np.max(np.abs(scores[new]-scores[old]))<=1e-6,'Rank difference exceeds score tolerance'
 assert np.all(scores[old][:-1]>=scores[old][1:]-1e-6)
 return {'exact':False,'max_score_difference':float(np.max(np.abs(scores[new]-scores[old]))),'new':new.tolist(),'old':old.tolist()}

def retrieve(include_short=False):
 e=h.encoder();compatibility(e);ds=selected();audit=[]
 api=Store() if include_short else None
 for name,qs in ds.items():
  qe,he=load_vectors(name,e);se=None
  if include_short:
   ps={p['query_id']:p for p in read(OUT/'prompts.json') if p['dataset']==name}
   texts=[api.get('short',name,q['query_id'],ps[q['query_id']]['short_prompt'],160)['text'] for q in qs]
   se=e.encode(texts)
  shared=corpus(name,qs[0],e) if name!='longbench2' else None
  if name=='legal':oldz=np.load(LEGAL/'saved_rankings.npz')
  elif name=='fiqa':oldz=np.load(CROSS/'fiqa_rankings.npz')
  for i,q in enumerate(qs):
   ce,meta=shared if shared else corpus(name,q,e)
   vectors=[qe[i],h.normalize(.5*qe[i]+.5*he[i]),h.normalize(.75*qe[i]+.25*he[i])]
   if include_short:vectors.append(h.normalize(.5*qe[i]+.5*se[i]))
   records={}
   for method,v in zip(METHODS,vectors):
    scores=v@ce.T;r=h.top(scores)
    if method in METHODS[:2]:
     oldmethod='question_dense' if method==METHODS[0] else 'focused160_mix_dense'
     old=np.array(read(CROSS/'longbench2_retrieval'/f"{q['query_id']}.json")['methods'][oldmethod]['top100']) if name=='longbench2' else oldz[oldmethod][q['source_index']]
     audit.append({'dataset':name,'query_id':q['query_id'],'method':method,**validate_rank(r,old,scores)})
     # Preserve original rank ordering for validated <=1e-6 ties to reuse exact reader prompts.
     r=old
    rec={'top100':r.tolist()}
    if name=='longbench2':rec['top5']=[{'chunk_id':int(j),'span':meta['spans'][j],'text':meta['chunks'][j]} for j in r[:5]]
    elif name=='fiqa':rec['passage_ids']=[meta[j] for j in r]
    records[method]=rec
   dump(OUT/'rankings'/name/f"{q['query_id']}.json",{'query_id':q['query_id'],'methods':records,'corpus_size':len(ce),'context_sha256':q.get('context_sha256')})
   print('Ranked',name,i+1,'/20',flush=True)
 dump(OUT/'ranking_replay.json',{'checks':audit,'passed':True,'near_ties':sum(not a['exact'] for a in audit)})

class Store:
 def __init__(self):
  OUT.mkdir(parents=True,exist_ok=True)
  self.db=sqlite3.connect(OUT/'api_checkpoint.sqlite');self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, kind TEXT, status TEXT, reserve REAL, cost REAL, payload TEXT)');self.db.commit();self.client=None
 def key(self,kind,ds,qid,prompt,cap):
  # Reader dedup uses complete prompt identity across conditions/queries.
  obj={'kind':kind,'dataset':ds,'query_id':qid if kind=='short' else None,'prompt_sha256':sha(prompt.encode()),'model':MODEL,'cap':cap,'reasoning':'none','temperature':0,'store':False,'tools':[]}
  return kind+'::'+sha(json.dumps(obj,sort_keys=True).encode())
 def get(self,kind,ds,qid,prompt,cap):
  r=self.db.execute('SELECT status,payload FROM calls WHERE key=?',(self.key(kind,ds,qid,prompt,cap),)).fetchone()
  assert r and r[0]=='saved',f'Missing or unresolved {kind} {ds} {qid}'
  return json.loads(r[1])
 def cost(self):return .076705+self.db.execute('SELECT COALESCE(SUM(COALESCE(cost,reserve)),0) FROM calls').fetchone()[0]
 def connect(self):
  from openai import OpenAI
  p=ROOT/'.env.local'
  if not p.exists():raise RuntimeError('Configure Group004 key locally with configure_key.py; never send it in chat')
  assert not p.is_symlink() and p.stat().st_mode&0o077==0,'Secret file must have permissions 600'
  values=dict(line.split('=',1) for line in p.read_text().splitlines() if '=' in line)
  self.client=OpenAI(api_key=values['OPENAI_API_KEY'].strip().strip('\"\''),organization=h.ORG,max_retries=0,timeout=120)
  self.client.models.retrieve(MODEL) # read-only, before any paid calls
  dump(OUT/'model_access.json',{'model':MODEL,'read_only_access':'passed','checked_at_unix':time.time()})
 def call(self,kind,ds,qid,prompt,cap):
  key=self.key(kind,ds,qid,prompt,cap)
  old=self.db.execute('SELECT status,payload FROM calls WHERE key=?',(key,)).fetchone()
  if old:
   if old[0]=='saved':return json.loads(old[1])
   raise RuntimeError('Unresolved attempt; automatic retry prohibited: '+key)
  if self.db.execute("SELECT COUNT(*) FROM calls WHERE status!='saved'").fetchone()[0]:raise RuntimeError('Unresolved attempt blocks further paid calls')
  if not self.client:self.connect()
  reserve=((len(prompt.encode())+256)*2.5+cap*15)/1e6
  self.db.execute('BEGIN IMMEDIATE')
  if self.cost()+reserve>2.:self.db.rollback();raise RuntimeError('USD 2 guard reached')
  self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?)',(key,kind,'pending',reserve,None,None));self.db.commit()
  try:
   r=self.client.responses.create(model=MODEL,input=prompt,max_output_tokens=cap,reasoning={'effort':'none'},temperature=0,store=False)
   u=r.usage.model_dump() if r.usage else {};text=r.output_text.strip()
   payload={'key':key,'dataset':ds,'query_id':qid,'prompt':prompt,'model_returned':r.model,'response_id':r.id,'status':r.status,'text':text,'usage':u,'word_count':len(text.split()),'incomplete_details':r.incomplete_details.model_dump() if r.incomplete_details else None,'cap':cap}
   cost=(u.get('input_tokens',0)*2.5+u.get('output_tokens',0)*15)/1e6 if u else None
   self.db.execute('UPDATE calls SET status=?,cost=?,payload=? WHERE key=?',('response_received',cost,json.dumps(payload),key));self.db.commit()
   assert text and u and r.model==MODEL and r.status in ['completed','incomplete'],'Invalid output retained; no retry'
   self.db.execute('UPDATE calls SET status=? WHERE key=?',('saved',key));self.db.commit();self.export()
   print('Saved',kind,ds,qid,'estimated recovery USD',round(self.cost(),6),flush=True)
   return payload
  except Exception as exc:
   self.db.execute("UPDATE calls SET status='failed_or_unknown' WHERE key=? AND status='pending'",(key,));self.db.commit();self.export()
   raise RuntimeError('API halted; checkpoint retained; '+type(exc).__name__) from None
 def export(self):
  rows=[json.loads(p) for p, in self.db.execute("SELECT payload FROM calls WHERE status='saved'")]
  dump(OUT/'successful_api_outputs.json',rows)
  dump(OUT/'cost_summary.json',{'guard_usd':2.,'lost_attempt_allowance_usd':.076705,'new_plus_reserved_usd':self.cost()-.076705,'recovery_total_usd':self.cost(),'successful_calls':len(rows),'states':dict(self.db.execute('SELECT status,count(*) FROM calls GROUP BY status')),'not_billing_statement':True})

def generate():
 api=Store()
 for p in read(OUT/'prompts.json'):api.call('short',p['dataset'],p['query_id'],p['short_prompt'],160)
 api.export();assert len([x for x in read(OUT/'successful_api_outputs.json') if x['cap']==160])==60

def answers(paid=False):
 api=Store();old=historical();rows=[];missing=[]
 for q in selected()['longbench2']:
  f=OUT/'rankings/longbench2'/f"{q['query_id']}.json"
  if not f.exists():continue
  ranks=read(f)['methods']
  for method,r in ranks.items():
   context='\n\n'.join(f'Retrieved chunk {i+1}: {c["text"]}' for i,c in enumerate(r['top5']))
   prompt=h.READER.format(context=context,**{k:q[k] for k in ['question','choice_A','choice_B','choice_C','choice_D']})
   key=h.answer_key(prompt);cached=old.get(key)
   if cached:
    assert cached['prompt']==prompt and cached['model_returned']==MODEL and cached['status']=='completed'
    o=cached;origin='historical_exact_prompt'
   else:
    assert method not in METHODS[:2],'A/B reader cache mismatch; halt before expanding scope'
    try:o=api.get('reader','longbench2',q['query_id'],prompt,32);origin='pilot_cache'
    except AssertionError:
     missing.append(key)
     if not paid:continue
     o=api.call('reader','longbench2',q['query_id'],prompt,32);origin='new'
   rows.append({'query_id':q['query_id'],'method':method,'pred':h.parse_answer(o['text']),'gold':q['answer'],'correct':int(h.parse_answer(o['text'])==q['answer']),'call_key':key,'prompt_sha256':sha(prompt.encode()),'source':origin})
   dump(OUT/'reader_predictions.json',rows)
 dump(OUT/'reader_cache_audit.json',{'logical_saved':len(rows),'missing_unique_prompts':len(set(missing)),'historical_hits':sum(r['source']=='historical_exact_prompt' for r in rows)})
 api.export()

def report():
 ds=selected();rows=[];lg=np.load(LEGAL/'saved_rankings.npz')['gold'];reader={(x['query_id'],x['method']):x for x in read(OUT/'reader_predictions.json')} if (OUT/'reader_predictions.json').exists() else {}
 for name,qs in ds.items():
  for q in qs:
   path=OUT/'rankings'/name/f"{q['query_id']}.json"
   if not path.exists():continue
   for method,r in read(path)['methods'].items():
    if name=='legal':v=float(int(lg[q['source_index']]) in r['top100'][:5])
    elif name=='fiqa':v=h.metrics(r['passage_ids'],q['qrels'],10)['ndcg']
    else:
     if (q['query_id'],method) not in reader:continue
     v=float(reader[q['query_id'],method]['correct'])
    rows.append({'dataset':name,'query_id':q['query_id'],'method':method,'value':v})
 f=pd.DataFrame(rows,columns=['dataset','query_id','method','value']);f.to_csv(OUT/'per_query_metrics.csv',index=False)
 summary=[];pairs=[]
 for name in ds:
  sub=f[f.dataset==name] if len(f) else pd.DataFrame()
  for method in METHODS:
   values=sub[sub.method==method].value if len(sub) else []
   summary.append({'dataset':name,'method':method,'metric':read(OUT/'protocol.json')['metrics'][name],'n':len(values),'expected_n':20,'value':float(np.mean(values)) if len(values)==20 else None})
  if len(sub):
   wide=sub.pivot(index='query_id',columns='method',values='value')
   for a,b in [('current_mix25','current_mix50'),('short_mix50','current_mix50')]+[(m,'question_dense') for m in METHODS[1:]]:
    if a in wide and b in wide:
     delta=(wide[a]-wide[b]).dropna();pairs.append({'dataset':name,'candidate':a,'control':b,'n':len(delta),'mean_delta':float(delta.mean()),'wins':int((delta>0).sum()),'losses':int((delta<0).sum()),'ties':int((delta==0).sum())})
 pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_changes.csv',index=False)
 short=Store().db.execute("SELECT count(*) FROM calls WHERE kind='short' AND status='saved'").fetchone()[0]
 complete=all(r['n']==20 for r in summary) and short==60
 dump(OUT/'completion_status.json',{'complete':complete,'short_outputs':short,'expected_short_outputs':60,'evaluated_pairs':len(rows),'expected_pairs':240,'longbench_predictions':len(reader),'expected_longbench_predictions':80})
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,name in zip(axes,ds):
  vals=[r for r in summary if r['dataset']==name];scale=1 if name=='fiqa' else 100;y=[r['value']*scale if r['value'] is not None else 0 for r in vals]
  ax.bar(['A','B','C','D'],y,color=['#718096','#3182ce','#38a169','#dd6b20']);ax.set_ylim(0,scale);ax.set_title(name);ax.set_ylabel(read(OUT/'protocol.json')['metrics'][name]+(' (%)' if scale==100 else ''))
  for i,r in enumerate(vals):ax.text(i,y[i]+.025*scale,f"{r['value']*scale:.3f}" if r['value'] is not None else f"pending\n({r['n']}/20)",ha='center',fontsize=8)
 fig.suptitle('Exploratory pilot: 20 fixed questions per dataset'+(' — INCOMPLETE' if not complete else ''));fig.text(.5,.015,'A: question only    B: current 50/50    C: current 75/25    D: short 50/50 (question/HyDE)',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(OUT/'pilot.png',dpi=180);fig.savefig(OUT/'pilot.svg');plt.close(fig)
 lines=['# Prompt/weight pilot','',f'Complete: {complete}. Short outputs: {short}/60. Scored condition-query pairs: {len(rows)}/240.','', '| Dataset | Metric | A | B | C | D |','|---|---|---:|---:|---:|---:|']
 for name in ds:
  vals=[r for r in summary if r['dataset']==name];lines.append('| '+name+' | '+vals[0]['metric']+' | '+' | '.join(f"{r['value']:.6f} (20/20)" if r['value'] is not None else f"pending ({r['n']}/20)" for r in vals)+' |')
 lines+=['','A: question only; B: focused160 50/50; C: focused160 75/25; D: short 50/50.','Exploratory reused subset, not a held-out test. One Legal/LongBench question is 5 percentage points. Missing values are not zero.','Full benchmark outputs are preserved. Historical LongBench dense replay coverage remains 219/1509; this pilot only rebuilds selected contexts.']
 lines+=['','## Descriptive failure examples']
 for name in ds:
  sub=f[f.dataset==name]
  if not len(sub):continue
  wide=sub.pivot(index='query_id',columns='method',values='value')
  for candidate in ['current_mix25','short_mix50']:
   if candidate not in wide or 'current_mix50' not in wide:continue
   delta=(wide[candidate]-wide['current_mix50']).dropna()
   for qid,v in delta[delta<0].sort_values().head(2).items():
    q=next(q for q in ds[name] if q['query_id']==qid)
    lines.append(f'- {name}, {qid}, {candidate} vs B: {v:+.4f}. Question: '+q['question'][:220].replace('\n',' ')+' (Observed loss; cause not established.)')
 if (OUT/'successful_api_outputs.json').exists():
  lengths=[{'dataset':o['dataset'],'query_id':o['query_id'],'word_count':o['word_count'],'outside_30_50':not 30<=o['word_count']<=50,'status':o['status'],'cap':o['cap']} for o in read(OUT/'successful_api_outputs.json') if o['cap']==160]
  pd.DataFrame(lengths).to_csv(OUT/'generation_lengths.csv',index=False)
 (OUT/'findings.md').write_text('\n'.join(lines)+'\n');print(json.dumps(read(OUT/'completion_status.json')),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','retrieve-base','generate','retrieve','answers-cached','answers','report','resume','access']);a=p.parse_args()
 # A process lock prevents concurrent duplicate paid calls/checkpoint writers.
 import fcntl
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'.runner.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.stage=='prepare':prepare()
  elif a.stage=='retrieve-base':retrieve(False);answers(False);report()
  elif a.stage=='generate':generate()
  elif a.stage=='retrieve':retrieve(True)
  elif a.stage=='answers-cached':answers(False);report()
  elif a.stage=='answers':answers(True);report()
  elif a.stage=='report':report()
  elif a.stage=='access':Store().connect();print('Read-only model access passed')
  elif a.stage=='resume':prepare();generate();retrieve(True);answers(True);report()
