"""Frozen transfer experiment: full BEIR FiQA test and LongBench v2 RAG.

No fitting or model selection on either test set. Credentials never enter artifacts.
Run stages independently; successful API calls and local indexing are resumable.
"""
from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, importlib.metadata
import json, os, re, sqlite3, threading, time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/external_benchmarks'
OUT = ROOT / 'results/person3_cross_dataset/gpt54_focused160_v1'
CACHE = ROOT / 'data/external_benchmarks/cache_v1'
MODEL = 'gpt-5.4-2026-03-05'
ORG = 'org-ya6BfJ99A55qzohO1pvOUaXO'
EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
BUDGET = 30.0
PRICE_IN, PRICE_OUT = 2.5, 15.0
METHODS = ['question_dense', 'bm25', 'standard_hybrid_d25',
    'standard_hybrid_d50', 'focused160_pure_dense', 'focused160_mix_dense',
    'focused160_pure_hybrid_d25', 'focused160_mix_hybrid_d25',
    'focused160_mix_hybrid_d50']
PAIRS = [('focused160_pure_dense','question_dense'),
    ('focused160_mix_dense','question_dense'),
    ('focused160_pure_hybrid_d25','standard_hybrid_d25'),
    ('focused160_mix_hybrid_d25','standard_hybrid_d25'),
    ('focused160_mix_hybrid_d50','standard_hybrid_d50'),
    ('focused160_mix_hybrid_d50','standard_hybrid_d25')]
PROMPTS = {
 'fiqa': 'Write one short hypothetical answer paragraph to help retrieve evidence about finance for the question below. Preserve the key financial issue and distinguish it from similar issues. Describe the relevant financial concept or mechanism in plain language. Do not invent company names, numerical returns, dates, regulations or facts about the questioner. Do not repeat the question or add headings. Aim for 70 to 100 words.\nQuestion: {q}\nHypothetical evidence paragraph:',
 'longbench2': 'Write one short hypothetical evidence paragraph to help retrieve information needed for the task below. Preserve the key entities, terminology and relationships in the task and distinguish it from similar tasks. Describe the type of passage or information that would support an answer. Do not invent names, numerical results, dates, quotations or facts. Do not repeat the question or add headings. Aim for 70 to 100 words.\nTask: {q}\nHypothetical evidence paragraph:'
}
READER = '''Please read the following retrieved text chunks and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)".'''

def dump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2)); t.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def sources():
    s=json.loads((DATA/'sources.json').read_text())
    for x in s: assert sha(DATA/x['name'])==x['sha256'], x['name']
    return s

def load_fiqa():
    corpus=pd.read_parquet(DATA/'fiqa_corpus.parquet').to_dict('records')
    queries=pd.read_parquet(DATA/'fiqa_queries.parquet').to_dict('records')
    rel=pd.read_csv(DATA/'fiqa_test.tsv',sep='\t',dtype={'query-id':str,'corpus-id':str})
    qrels=collections.defaultdict(dict)
    for r in rel.to_dict('records'):
        if r['score']>0:qrels[str(r['query-id'])][str(r['corpus-id'])]=int(r['score'])
    by_id={str(q['_id']):q for q in queries}
    qa=[by_id[qid] for qid in sorted(qrels)]
    assert len(corpus)==57638 and len(qa)==648 and len(rel)==1706
    corpus_ids={str(x['_id']) for x in corpus}
    assert all(d in corpus_ids for rels in qrels.values() for d in rels)
    return corpus,qa,dict(qrels)

def load_lb():
    metadata=CACHE/'longbench2_metadata.json'
    if not metadata.exists():
        import ijson
        qa=[];folder=CACHE/'longbench2_contexts';folder.mkdir(parents=True,exist_ok=True)
        with (DATA/'longbench2.json').open('rb') as f:
            for row in ijson.items(f,'item'):
                context=row.pop('context');ctxhash=hashlib.sha256(context.encode()).hexdigest()
                p=folder/(ctxhash+'.txt')
                if not p.exists():p.write_text(context)
                row['context_sha256']=ctxhash;qa.append(row)
        dump(metadata,qa)
    qa=json.loads(metadata.read_text())
    assert len(qa)==503 and len({q['_id'] for q in qa})==503
    return qa

def context_for(q):
    return (CACHE/'longbench2_contexts'/(q['context_sha256']+'.txt')).read_text()

def protocol():
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    p=dict(model=MODEL,embedding_model=EMBED,embedding_revision=REV,
      embedding_max_seq_length=256,normalized=True,vector_question_weight=.5,
      profiles={'focused160':dict(prompts=PROMPTS,max_output_tokens=160)},
      api=dict(reasoning_effort='none',temperature=0,store=False,retries=0,concurrency=6),
      bm25=dict(k1=1.2,b=.75,binary_query_counts=True),rrf_k=60,depth=100,
      methods=METHODS,paired_comparisons=PAIRS,holm_family='12 comparisons: six per dataset',
      fiqa=dict(split='test',n=648,corpus_size=57638,primary_metric='nDCG@10',
        metrics=['Hit@5','Recall@5','MRR@5','nDCG@5','nDCG@10','Recall@100'],
        document_representation='title + text, first 256 MiniLM tokens; BM25 uses full document'),
      longbench2=dict(n=503,scope='custom top-5 RAG adaptation, not official leaderboard replication',
        primary_metric='multiple-choice answer accuracy',context_scope='each question own full context only',
        chunk_tokens=224,chunk_overlap=32,tokenizer=EMBED,rag_top_k=5,
        retrieval_input='question only, no choices or gold answer',hyde_input='question only',
        reader_model=MODEL,reader_prompt=READER,reader_max_output_tokens=32,
        context_order='retrieval rank',no_context_control=True,
        retrieval_relevance_labels=False,full_context_reader_evaluation=False),
      cost_estimate_usd_per_million=dict(input=PRICE_IN,output=PRICE_OUT),
      budget_guard_usd=BUDGET,dataset_sources=sources(),
      limits=['Focused prompt transferred with domain wording changes before test outcomes.',
        'Original96 omitted because prior run had 94/100 length-capped outputs.',
        'No tuning, train/dev use, or selection on these test sets.',
        'Public benchmarks can overlap model pretraining; not contamination-free evidence.',
        'FiQA and LongBench metrics are not comparable to each other or legal Hit@5.',
        'MiniLM fixed from prior experiment; LongBench chunking is a new task adaptation.'])
    old=OUT/'protocol.json'
    if old.exists():assert json.loads(old.read_text())==json.loads(json.dumps(p)),'Protocol changed: use new run directory'
    else:dump(old,p)
    versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','torch','openai','scipy','scikit-learn','transformers','huggingface-hub','pyarrow','ijson','httpx']}
    for optional in ['sentence-transformers','onnxruntime']:
        try:versions[optional]=importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:versions[optional]='not installed (historical backend only)'
    dump(OUT/'environment.json',dict(versions=versions,python=os.sys.version))

class API:
    def __init__(self):
        from openai import OpenAI
        values={}
        for line in (ROOT/'.env.local').read_text().splitlines():
            if '=' in line:
                k,v=line.split('=',1);values[k.strip()]=v.strip().strip('\"\'')
        self.client=OpenAI(api_key=values['OPENAI_API_KEY'],organization=ORG,max_retries=0,timeout=120)
        self.lock=threading.RLock();self.halt=threading.Event()
        self.db=sqlite3.connect(OUT/'api_checkpoint.sqlite',check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, kind TEXT, status TEXT, reserve REAL, cost REAL, payload TEXT)')
        self.db.commit()
    def cost(self):
        recovery=OUT/'checkpoint_recovery.json'
        allowance=json.loads(recovery.read_text()).get('lost_checkpoint_cost_allowance_usd',0) if recovery.exists() else 0
        return self.db.execute('SELECT COALESCE(SUM(COALESCE(cost,reserve)),0) FROM calls').fetchone()[0]+allowance
    def run(self,jobs):
        errors=[]; n=0
        def one(job):
            key,kind,prompt,cap=job
            reserve=((len(prompt.encode())+256)*PRICE_IN+cap*PRICE_OUT)/1e6
            with self.lock:
                old=self.db.execute('SELECT status,payload FROM calls WHERE key=?',(key,)).fetchone()
                if old:
                    if old[0]=='saved':return
                    self.halt.set();raise RuntimeError('Unresolved call; no automatic retry: '+key)
                if self.halt.is_set():return
                if self.cost()+reserve>BUDGET:
                    self.halt.set();raise RuntimeError('Budget guard reached; checkpoint retained')
                self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?)',(key,kind,'pending',reserve,None,None));self.db.commit()
            started=time.perf_counter()
            try:
                r=self.client.responses.create(model=MODEL,input=prompt,max_output_tokens=cap,
                  reasoning={'effort':'none'},temperature=0,store=False)
                usage=r.usage.model_dump() if r.usage else {}
                cost=(usage.get('input_tokens',0)*PRICE_IN+usage.get('output_tokens',0)*PRICE_OUT)/1e6
                text=r.output_text.strip()
                payload=dict(key=key,kind=kind,prompt=prompt,text=text,model_returned=r.model,
                  response_id=r.id,status=r.status,usage=usage,seconds=time.perf_counter()-started,
                  incomplete_details=r.incomplete_details.model_dump() if r.incomplete_details else None,
                  cost_usd_estimate=cost)
                assert text and r.usage and r.model==MODEL,'Empty output or missing usage/model mismatch'
                with self.lock:
                    self.db.execute('UPDATE calls SET status=?,cost=?,payload=? WHERE key=?',('saved',cost,json.dumps(payload,ensure_ascii=False),key));self.db.commit()
            except Exception as exc:
                with self.lock:
                    safe=dict(error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
                    code=getattr(exc,'code',None)
                    safe['api_code']=code if code in ['insufficient_quota','rate_limit_exceeded','invalid_api_key','model_not_found','permission_denied'] else None
                    chain=[];cause=exc.__cause__
                    while cause is not None:
                        message=str(cause).lower()
                        chain.append(dict(type=type(cause).__name__,signals=[s for s in ['407','proxy','dns','getaddrinfo','connection reset','timed out','certificate'] if s in message]))
                        cause=cause.__cause__
                    safe['transport_cause_chain']=chain
                    self.db.execute('UPDATE calls SET status=?,payload=? WHERE key=?',('failed_or_unknown',json.dumps(safe),key));self.db.commit();self.halt.set()
                raise RuntimeError('API stopped safely: '+type(exc).__name__+' HTTP='+str(getattr(exc,'status_code',None))) from None
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for f in concurrent.futures.as_completed([ex.submit(one,j) for j in jobs]):
                try:f.result()
                except Exception as exc:errors.append(str(exc));self.halt.set()
                n+=1
                if n%50==0:
                    with self.lock:print('API tasks',n,'estimated/reserved USD',round(self.cost(),4),flush=True)
        if errors:raise RuntimeError('; '.join(sorted(set(errors))))
        with self.lock:print('API stage complete; estimated USD',round(self.cost(),6),flush=True)
    def outputs(self):
        return {k:json.loads(p) for k,p in self.db.execute("SELECT key,payload FROM calls WHERE status='saved'")}

def generate():
    _,fq,_=load_fiqa();lb=load_lb();jobs=[]
    for ds,qa in [('fiqa',fq),('longbench2',lb)]:
        for q in qa:
            question=q['text'] if ds=='fiqa' else q['question']
            jobs.append((f'hyde::{ds}::{q["_id"]}','hyde_'+ds,PROMPTS[ds].format(q=question),160))
    api=API();api.run(jobs)
    rows=[api.outputs()[j[0]] for j in jobs]
    with (OUT/'generated_passages.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def encoder():
    # CPU encoding is entirely local. Do not import the runtime whose telemetry
    # connection was rejected; the same pinned weights also run in PyTorch.
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['DO_NOT_TRACK']='1'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,sys
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(8)
    class Encoder:
        def __init__(self):
            self.tokenizer=AutoTokenizer.from_pretrained(EMBED,revision=REV,cache_dir=str(ROOT.parent/'model_cache'),local_files_only=True)
            self.model=AutoModel.from_pretrained(EMBED,revision=REV,cache_dir=str(ROOT.parent/'model_cache'),local_files_only=True).to(device='cpu',dtype=torch.float32).eval()
            assert 'onnxruntime' not in sys.modules,'Offline encoder must not load the telemetry runtime'
            prior=OUT/'execution_backend.json'
            if prior.exists() and json.loads(prior.read_text()).get('backend')!='PyTorch CPU float32 (offline)':
                dump(OUT/'execution_backend_prior.json',json.loads(prior.read_text()))
            dump(prior,dict(backend='PyTorch CPU float32 (offline)',version=torch.__version__,
              embedding_model=EMBED,embedding_revision=REV,threads=8,batch_size=16,quantized=False,
              offline=True,hf_telemetry_disabled=True,onnxruntime_loaded=False,
              note='Continues the same pinned float32 MiniLM model; earlier validated float32 caches are reused. Microsoft runtime is not imported by the offline encoder.'))
            dump(OUT/'telemetry_policy.json',dict(encoder_network_required=False,backend='offline PyTorch',
              rejected_component_removed=True,onnxruntime_imported=False,HF_HUB_DISABLE_TELEMETRY='1',
              DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
              reason='Safety review rejected an unknown Microsoft telemetry connection. Encoding resumes without that runtime, entirely from local weights and tokenizer.'))
        def encode(self,texts,**kwargs):
            order=sorted(range(len(texts)),key=lambda i:len(texts[i]),reverse=True)
            out=np.empty((len(texts),384),dtype=np.float32)
            for start in range(0,len(order),16):
                indices=order[start:start+16]
                tokens=self.tokenizer([texts[i] for i in indices],padding=True,truncation=True,max_length=256,return_tensors='np')
                with torch.inference_mode():
                    y=self.model(**{k:torch.from_numpy(v.astype(np.int64)) for k,v in tokens.items()}).last_hidden_state.numpy()
                mask=tokens['attention_mask'][:,:,None].astype(np.float32)
                vectors=(y*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1e-12)
                out[indices]=normalize(vectors)
            return out
    return Encoder()

def encode(e,texts):
    return e.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def top(s):return np.argsort(-s,kind='stable')[:min(100,len(s))].astype(np.int32)

TOKEN_RE=re.compile(r'[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*')
def tokenize(t):return TOKEN_RE.findall(t.lower())

class BM25:
    def __init__(self,texts):
        self.vec=CountVectorizer(lowercase=True,tokenizer=tokenize,token_pattern=None,dtype=np.float32)
        try:self.mat=self.vec.fit_transform(texts).tocsr().astype(np.float32)
        except ValueError:self.mat=None;self.n=len(texts);return
        dl=np.asarray(self.mat.sum(axis=1)).ravel();avg=max(float(dl.mean()),1e-12)
        df=np.asarray((self.mat>0).sum(axis=0)).ravel()
        idf=np.log1p((len(texts)-df+.5)/(df+.5)).astype(np.float32)
        ids=np.repeat(np.arange(len(texts)),np.diff(self.mat.indptr));tf=self.mat.data
        self.mat.data=(tf*2.2/(tf+1.2*(.25+.75*dl[ids]/avg)))*idf[self.mat.indices]
    def rank(self,q):
        if self.mat is None:return np.arange(min(100,self.n),dtype=np.int32)
        qc=self.vec.transform([q]).tocsr();qc.data[:]=1
        return top((qc@self.mat.T).toarray().ravel())

def fuse(a,b,dense_weight):
    s={}
    for ranks,w in [(a,1-dense_weight),(b,dense_weight)]:
        for r,i in enumerate(ranks,1):s[int(i)]=s.get(int(i),0)+w/(60+r)
    return np.array(sorted(s,key=lambda i:(-s[i],i))[:100],dtype=np.int32)

def ranks(ce,qe,he,br):
    qr=top(qe@ce.T);hr=top(he@ce.T);mr=top(normalize(.5*qe+.5*he)@ce.T)
    return dict(zip(METHODS,[qr,br,fuse(br,qr,.25),fuse(br,qr,.5),hr,mr,
        fuse(br,hr,.25),fuse(br,mr,.25),fuse(br,mr,.5)]))

def metrics(ids,rel,k):
    grades=np.array([rel.get(str(d),0) for d in ids[:k]],dtype=float)
    matches=grades>0;positive=sum(v>0 for v in rel.values())
    hit=float(matches.any());mrr=1/(np.flatnonzero(matches)[0]+1) if hit else 0.
    ideal=np.array(sorted(rel.values(),reverse=True)[:k],dtype=float)
    dcg=(grades/np.log2(np.arange(len(grades))+2)).sum()
    idcg=(ideal/np.log2(np.arange(len(ideal))+2)).sum()
    return dict(hit=hit,recall=float(matches.sum()/positive),precision=float(matches.sum()/k),
      mrr=float(mrr),ndcg=float(dcg/idcg) if idcg else 0.)

def fiqa():
    corpus,qa,qrels=load_fiqa();e=encoder()
    texts=[' '.join([str(x.get('title','')),str(x['text'])]).strip() for x in corpus]
    cache=CACHE/'fiqa_embeddings.npy'
    if cache.exists():ce=np.load(cache)
    else:
        print('FiQA embedding full corpus',len(texts),flush=True)
        ce=encode(e,texts);np.save(cache,ce)
    assert ce.shape==(57638,384)
    qe=encode(e,[q['text'] for q in qa])
    outputs=API().outputs()
    he=encode(e,[outputs['hyde::fiqa::'+str(q['_id'])]['text'] for q in qa])
    np.savez_compressed(OUT/'fiqa_query_embeddings.npz',questions=qe,hyde=he)
    bm=BM25(texts);pids=[str(c['_id']) for c in corpus];allr={m:[] for m in METHODS};rows=[]
    for i,q in enumerate(qa):
        rs=ranks(ce,qe[i],he[i],bm.rank(q['text']));qid=str(q['_id'])
        for m,r in rs.items():
            allr[m].append(r)
            ids=[pids[j] for j in r]
            for k in [1,3,5,10,100]:
                rows.append(dict(method=m,query_id=qid,k=k,**metrics(ids,qrels[qid],k)))
        if (i+1)%100==0:print('FiQA ranked',i+1,'/648',flush=True)
    np.savez_compressed(OUT/'fiqa_rankings.npz',query_ids=np.array([str(q['_id']) for q in qa]),passage_ids=np.array(pids),**{m:np.stack(v) for m,v in allr.items()})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'fiqa_per_query_metrics.csv',index=False)
    summary=frame.groupby(['method','k'],sort=False)[['hit','recall','precision','mrr','ndcg']].mean().reset_index();summary['n']=648
    summary.to_csv(OUT/'fiqa_metrics.csv',index=False)
    with (OUT/'fiqa_queries.jsonl').open('w') as f:
        for q in qa:f.write(json.dumps(dict(query_id=str(q['_id']),question=q['text'],qrels=qrels[str(q['_id'])]),ensure_ascii=False)+'\n')
    print(summary[summary.k.isin([5,10])].to_string(index=False),flush=True)

def chunks_for(e,text):
    # Offsets preserve original case, code and table formatting in reader evidence.
    # WordPiece's whitespace-separated words are independent: bounded input windows
    # ending at whitespace preserve exactly the same tokens/offsets without the
    # fast tokenizer's multi-million-token peak allocation.
    offsets=[];start=0
    while start<len(text):
        end=min(start+60000,len(text))
        while end<len(text) and not text[end].isspace():end+=1
        tok=e.tokenizer(text[start:end],add_special_tokens=False,return_offsets_mapping=True,truncation=False,verbose=False)
        offsets.extend((a+start,b+start) for a,b in tok['offset_mapping']);start=end
    chunks=[];spans=[]
    for start in range(0,len(offsets),192):
        end=min(start+224,len(offsets));a=offsets[start][0];b=offsets[end-1][1]
        chunks.append(text[a:b]);spans.append([int(a),int(b)])
        if end==len(offsets):break
    if not chunks:chunks=[''];spans=[[0,0]]
    return chunks,spans

def cached_embeddings(e,chunks,local):
    """Reuse identical chunks across distinct long contexts; float32, same model throughout."""
    ep=local/'embeddings.npy'
    global_db=sqlite3.connect(CACHE/'chunk_embeddings.sqlite')
    global_db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)')
    keys=[hashlib.sha256(s.encode()).hexdigest() for s in chunks]
    if ep.exists():
        ce=np.load(ep)
    else:
        ce=np.empty((len(chunks),384),dtype=np.float32);missing={}
        for i,key in enumerate(keys):
            row=global_db.execute('SELECT vector FROM embeddings WHERE key=?',(key,)).fetchone()
            if row:ce[i]=np.frombuffer(row[0],dtype=np.float32)
            else:missing.setdefault(key,[]).append(i)
        todo=list(missing)
        print('Encoding',len(todo),'new /',len(chunks),'total chunks',flush=True)
        for start in range(0,len(todo),512):
            ks=todo[start:start+512];vectors=encode(e,[chunks[missing[k][0]] for k in ks])
            for key,v in zip(ks,vectors):
                ce[missing[key]]=v
                global_db.execute('INSERT OR IGNORE INTO embeddings VALUES (?,?)',(key,v.astype(np.float32).tobytes()))
            global_db.commit()
        tmp=local/'embeddings.tmp.npy';np.save(tmp,ce);tmp.replace(ep)
    global_db.executemany('INSERT OR IGNORE INTO embeddings VALUES (?,?)',[(k,v.astype(np.float32).tobytes()) for k,v in zip(keys,ce)])
    global_db.commit();global_db.close();return ce

def longbench():
    qa=load_lb();e=encoder();query_cache=OUT/'longbench2_query_embeddings.npz'
    if query_cache.exists():
        with np.load(query_cache) as saved:qe=saved['questions'];he=saved['hyde']
        assert qe.shape==he.shape==(503,384)
    else:
        outputs=API().outputs()
        qe=encode(e,[q['question'] for q in qa])
        he=encode(e,[outputs['hyde::longbench2::'+q['_id']]['text'] for q in qa])
        np.savez_compressed(query_cache,questions=qe,hyde=he)
    target=OUT/'longbench2_retrieval';target.mkdir(exist_ok=True)
    # Group by identical full context so indexing is reused, but never search other questions' contexts.
    groups=collections.defaultdict(list)
    for i,q in enumerate(qa):groups[q['context_sha256']].append(i)
    completed=0
    for ctxhash,indices in groups.items():
        needed=[i for i in indices if not (target/(qa[i]['_id']+'.json')).exists()]
        if not needed:completed+=len(indices);continue
        t=time.perf_counter();text=context_for(qa[indices[0]])
        local=CACHE/'longbench2'/ctxhash;local.mkdir(parents=True,exist_ok=True)
        cp=local/'chunks.json'
        if cp.exists():
            c=json.loads(cp.read_text());chunks=c['chunks'];spans=c['spans']
        else:
            chunks,spans=chunks_for(e,text);dump(cp,dict(chunks=chunks,spans=spans))
        ce=cached_embeddings(e,chunks,local)
        bm=BM25(chunks)
        for i in needed:
            q=qa[i];rr=ranks(ce,qe[i],he[i],bm.rank(q['question']))
            obj=dict(query_id=q['_id'],context_sha256=ctxhash,n_chunks=len(chunks),
              context_characters=len(text),domain=q['domain'],difficulty=q['difficulty'],length=q['length'],
              methods={m:dict(top100=[int(x) for x in r],
                top5=[dict(chunk_id=int(j),span=spans[j],text=chunks[j]) for j in r[:5]]) for m,r in rr.items()})
            dump(target/(q['_id']+'.json'),obj)
        completed+=len(indices)
        print('LongBench retrieval',completed,'/503;',len(chunks),'chunks;',round(time.perf_counter()-t,2),'seconds',flush=True)

def index_longbench():
    qa=load_lb();e=encoder();seen=set();total_chunks=0
    for i,q in enumerate(qa):
        ctxhash=q['context_sha256']
        if ctxhash in seen:continue
        seen.add(ctxhash);local=CACHE/'longbench2'/ctxhash;local.mkdir(parents=True,exist_ok=True)
        cp=local/'chunks.json';ep=local/'embeddings.npy';t=time.perf_counter()
        if ep.exists():continue
        if cp.exists():c=json.loads(cp.read_text());chunks=c['chunks'];spans=c['spans']
        else:
            chunks,spans=chunks_for(e,context_for(q));dump(cp,dict(chunks=chunks,spans=spans))
        ce=cached_embeddings(e,chunks,local);total_chunks+=len(chunks)
        print('LongBench index',len(seen),'/462 contexts;',len(chunks),'chunks;',round(time.perf_counter()-t,2),'seconds; total new chunks',total_chunks,flush=True)

def tokenize_longbench():
    from transformers import AutoTokenizer
    class TokenizerOnly:pass
    e=TokenizerOnly();e.tokenizer=AutoTokenizer.from_pretrained(EMBED,revision=REV,cache_dir=str(ROOT.parent/'model_cache'),local_files_only=True)
    qa=load_lb();seen=set();unique=set();total=0
    for q in qa:
        key=q['context_sha256']
        if key in seen:continue
        seen.add(key);local=CACHE/'longbench2'/key;local.mkdir(parents=True,exist_ok=True);cp=local/'chunks.json'
        if cp.exists():chunks=json.loads(cp.read_text())['chunks']
        else:
            chunks,spans=chunks_for(e,context_for(q));dump(cp,dict(chunks=chunks,spans=spans))
        total+=len(chunks);unique.update(hashlib.sha256(c.encode()).hexdigest() for c in chunks)
        if len(seen)%25==0:print('Tokenized',len(seen),'/462; total chunks',total,'unique chunks',len(unique),flush=True)
    dump(OUT/'chunk_inventory.json',dict(contexts=len(seen),total_chunks_across_unique_contexts=total,unique_chunk_texts=len(unique)))
    print('Chunk inventory',total,len(unique),flush=True)

def answer_key(prompt):
    return 'reader::'+hashlib.sha256((MODEL+'|32|'+prompt).encode()).hexdigest()

def reader_jobs(qa):
    jobs={};mapping=[]
    for q in qa:
        r=json.loads((OUT/'longbench2_retrieval'/(q['_id']+'.json')).read_text())
        public={k:q[k] for k in ['question','choice_A','choice_B','choice_C','choice_D']}
        for m in METHODS+['no_context']:
            chunks=r['methods'][m]['top5'] if m!='no_context' else []
            context='\n\n'.join(f'Retrieved chunk {i+1}: {c["text"]}' for i,c in enumerate(chunks))
            prompt=READER.format(context=context,**public)
            key=answer_key(prompt)
            jobs[key]=(key,'reader_longbench2',prompt,32)
            mapping.append(dict(query_id=q['_id'],method=m,call_key=key,
              gold=q['answer'],domain=q['domain'],difficulty=q['difficulty'],length=q['length']))
    return list(jobs.values()),mapping

def read_answers():
    qa=load_lb();jobs,mapping=reader_jobs(qa)
    dump(OUT/'longbench2_answer_mapping.json',mapping)
    print('LongBench reader',len(mapping),'method-query pairs;',len(jobs),'unique API calls',flush=True)
    API().run(jobs)

def stream_answers():
    qa=load_lb();api=API();done=set()
    while len(done)<503:
        pause_flag=ROOT.parent/'api_access_private/pause_cross_reader'
        if pause_flag.exists() and pause_flag.read_text().strip()=='pause':
            print('Reader paused cleanly between completed batches.',flush=True);return
        ready=[q for q in qa if q['_id'] not in done and (OUT/'longbench2_retrieval'/(q['_id']+'.json')).exists()]
        if ready:
            jobs,_=reader_jobs(ready);api.run(jobs);done.update(q['_id'] for q in ready)
            print('LongBench reader questions complete',len(done),'/503',flush=True)
        else:time.sleep(10)
    _,mapping=reader_jobs(qa);dump(OUT/'longbench2_answer_mapping.json',mapping)

def parse_answer(s):
    # Same intended answer extraction as official RAG output; malformed answers count wrong.
    patterns=[r'The correct answer is\s*\(?([A-D])\)?(?=[\s.,:;!?)]|$)',r'^\s*\(?([A-D])\)?[.\s]*$']
    for p in patterns:
        m=re.search(p,s,re.I)
        if m:return m.group(1).upper()
    return None

def holm(rows,key='p_raw'):
    order=np.argsort([r[key] for r in rows]);running=0.
    for j,i in enumerate(order):
        running=max(running,min(1.,(len(rows)-j)*rows[i][key]));rows[i]['p_holm']=running
    return rows

def summarize():
    api=API();outputs=api.outputs();mapping=json.loads((OUT/'longbench2_answer_mapping.json').read_text());rows=[]
    for m in mapping:
        o=outputs[m['call_key']];pred=parse_answer(o['text'])
        rows.append(m|dict(pred=pred,correct=int(pred==m['gold']),response_status=o['status']))
    frame=pd.DataFrame(rows);assert len(frame)==5030
    frame.to_csv(OUT/'longbench2_predictions.csv',index=False)
    tables=[]
    for field in ['overall','domain','difficulty','length']:
        ff=frame.assign(overall='all') if field=='overall' else frame
        table=ff.groupby(['method',field],sort=False).agg(n=('correct','size'),accuracy=('correct','mean')).reset_index().rename(columns={field:'group'})
        table['group_by']=field;tables.append(table)
    pd.concat(tables,ignore_index=True).to_csv(OUT/'longbench2_accuracy.csv',index=False)
    fq=pd.read_csv(OUT/'fiqa_per_query_metrics.csv',dtype={'query_id':str})
    tests=[]
    # Planned family: six comparisons per dataset. FiQA primary nDCG@10 paired permutation;
    # LongBench primary binary accuracy exact McNemar. No post-hoc best-method test.
    rng=np.random.default_rng(20261028)
    for ds in ['fiqa','longbench2']:
        for a,b in PAIRS:
            if ds=='longbench2':
                pivot=frame.pivot(index='query_id',columns='method',values='correct')
                av=pivot[a].to_numpy();bv=pivot[b].to_numpy();won=int(((av==1)&(bv==0)).sum());lost=int(((av==0)&(bv==1)).sum())
                p=binomtest(won,won+lost,.5).pvalue if won+lost else 1.
                t='exact paired McNemar';metric='answer_accuracy'
            else:
                pivot=fq[fq.k==10].pivot(index='query_id',columns='method',values='ndcg')
                av=pivot[a].to_numpy();bv=pivot[b].to_numpy();delta=av-bv
                won=int((delta>1e-12).sum());lost=int((delta< -1e-12).sum())
                extreme=0;observed=abs(delta.mean());nperm=100000
                for _ in range(100):
                    signs=rng.choice([-1,1],size=(1000,len(delta)))
                    extreme+=int((np.abs((signs*delta).mean(axis=1))>=observed-1e-15).sum())
                p=(extreme+1)/(nperm+1);t='paired sign-flip permutation, 100000 draws';metric='nDCG@10'
            tests.append(dict(dataset=ds,candidate=a,control=b,metric=metric,n=len(av),
              candidate_mean=float(av.mean()),control_mean=float(bv.mean()),delta=float((av-bv).mean()),
              gained=won,lost=lost,test=t,p_raw=float(p)))
    pd.DataFrame(holm(tests)).to_csv(OUT/'paired_tests.csv',index=False)
    audit=[]
    with (OUT/'api_outputs.jsonl').open('w') as f:
        for key,o in outputs.items():
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
            audit.append(dict(key=key,kind=o['kind'],status=o['status'],words=len(o['text'].split()),
              input_tokens=o['usage']['input_tokens'],output_tokens=o['usage']['output_tokens'],
              cost_usd_estimate=o['cost_usd_estimate'],seconds=o['seconds']))
    ad=pd.DataFrame(audit);ad.to_csv(OUT/'api_audit.csv',index=False)
    ad.groupby('kind').agg(calls=('key','size'),input_tokens=('input_tokens','sum'),output_tokens=('output_tokens','sum'),cost_usd_estimate=('cost_usd_estimate','sum')).to_csv(OUT/'cost_summary.csv')
    unresolved=[dict(key=k,kind=kind,reserved_upper_usd=reserve,metadata=json.loads(payload))
      for k,kind,reserve,payload in api.db.execute("SELECT key,kind,reserve,payload FROM calls WHERE status='unresolved_transport'")]
    dump(OUT/'unresolved_transport_attempts.json',unresolved)
    dump(OUT/'cost_accounting.json',dict(known_successful_request_estimate_usd=float(ad.cost_usd_estimate.sum()),
      unresolved_attempts=len(unresolved),unresolved_upper_allowance_usd=sum(x['reserved_upper_usd'] for x in unresolved),
      checkpoint_recovery=json.loads((OUT/'checkpoint_recovery.json').read_text()) if (OUT/'checkpoint_recovery.json').exists() else None,
      conservative_total_upper_estimate_usd=api.cost(),billing_statement=False))
    print(pd.concat(tables).query("group_by=='overall'").to_string(index=False),flush=True)
    print('Total estimated API cost',api.cost(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['protocol','generate','fiqa','index_longbench','tokenize_longbench','longbench','answers','stream_answers','summarize'])
    args=parser.parse_args();protocol()
    if args.stage!='protocol':globals()[{'answers':'read_answers'}.get(args.stage,args.stage)]()
