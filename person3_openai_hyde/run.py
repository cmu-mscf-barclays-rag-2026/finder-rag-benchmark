"""Run the same fixed OpenAI HyDE experiment locally; no plaintext credential output."""

from pathlib import Path
import json, hashlib, re, time, platform, importlib.metadata, zipfile
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
def display(value):
    print(value.to_string(index=False), flush=True)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/person3_openai_hyde'
BASE.mkdir(parents=True, exist_ok=True)
MODEL = 'gpt-5.4-2026-03-05'
ORGANIZATION = 'org-ya6BfJ99A55qzohO1pvOUaXO'  # Group 004, from the invitation
DATA_REV = 'db0b31dc6d195ce9916897e1ac5e4e6209736c8a'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBED_REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
PRICE_INPUT, PRICE_OUTPUT = 2.50, 15.00  # USD per million tokens; conservative: no cache discount
BUDGET_USD = 1.00
PROFILES = {
    'original96': {
        'max_output_tokens': 96,
        'prompt': 'Generate the type of legal passage that would contain the evidence needed to answer this question about Victorian criminal law. Question: {q} Legal passage:'
    },
    'focused160': {
        'max_output_tokens': 160,
        'prompt': 'Write one short hypothetical textbook paragraph to help retrieve evidence about Victorian criminal law for the question below. Preserve the key legal issue and distinguish it from similar issues. Describe the relevant legal rule in plain language. Do not invent case names, statutory sections, dates, penalties or facts. Do not repeat the question or add headings. Aim for 70 to 100 words.\nQuestion: {q}\nHypothetical evidence paragraph:'
    }
}
HASHES = {
 'corpus.jsonl': '3a3565bc5429f6cead90548e81f87352b449927e4be1cdd804c28d766bb9c246',
 'qa.jsonl': 'e3b869a4e293d081ec5f5b39c2058c8d27b36f611aa9f8275eb1877a7c8b38b0'
}
PROTOCOL = dict(model=MODEL, organization=ORGANIZATION, data_revision=DATA_REV,
    data_hashes=HASHES, embedding_model=EMBED_MODEL, embedding_revision=EMBED_REV,
    embedding_max_seq_length=256, embedding_device='cpu', profiles=PROFILES,
    reasoning_effort='none', temperature=0, store=False, api_retries=0,
    vector_question_weight=0.5, bm25_k1=1.2, bm25_b=0.75, rrf_k=60, depth=100,
    pricing_usd_per_million=dict(input=PRICE_INPUT, output=PRICE_OUTPUT),
    scope='Exploratory: the same previously inspected 100 questions; no final-answer evaluation',
    comparison_limits='OpenAI tokenization/chat formatting/decoding differ from local generators; focused profile changes prompt and token cap together')
fingerprint = hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()[:16]
OUT = BASE / ('gpt54_' + fingerprint)
OUT.mkdir(exist_ok=True)

def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

save_json(OUT/'protocol.json', PROTOCOL)
save_json(OUT/'environment.json', dict(python=platform.python_version(),
    versions={p:importlib.metadata.version(p) for p in ['openai','sentence-transformers','torch','numpy','pandas','scipy','scikit-learn']}))
print('Results folder:', OUT, flush=True)
print('Model:', MODEL, '| Experiment budget protection: $', BUDGET_USD)


import argparse
p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','generate','evaluate']);args=p.parse_args()

DATA=ROOT/'data/legal_rag_bench'
for filename,expected in HASHES.items():
    p=DATA/filename
    assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==expected, filename
corpus_rows = [json.loads(s) for s in (DATA/'corpus.jsonl').read_text().splitlines() if s.strip()]
qa = [json.loads(s) for s in (DATA/'qa.jsonl').read_text().splitlines() if s.strip()]
assert len(qa)==100 and len(corpus_rows)==4876
passage_ids = [str(r['id']) for r in corpus_rows]
id_to_index = {x:i for i,x in enumerate(passage_ids)}
corpus = [' '.join((str(r.get('title','')),str(r['text']))).strip() for r in corpus_rows]
questions = [str(r['question']) for r in qa]
gold = np.array([id_to_index[str(r['relevant_passage_id'])] for r in qa], dtype=np.int32)
pd.DataFrame(qa).to_csv(OUT/'evaluation_queries.csv',index=False)
pd.DataFrame({'corpus_index':range(len(passage_ids)), 'passage_id':passage_ids}).to_csv(OUT/'passage_index.csv',index=False)
print('校验通过：100 questions / 4,876 passages')


if args.stage in ['prepare','evaluate']:
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(2)
    encoder = SentenceTransformer(EMBED_MODEL, revision=EMBED_REV, device='cpu')
    encoder.max_seq_length = 256
    
    def encode(texts):
        return encoder.encode(texts, batch_size=32, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True)
    def normalize(x):
        return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True), 1e-12)
    def top(scores, depth=100):
        return np.argsort(-scores,axis=1,kind='stable')[:,:depth].astype(np.int32)
    def fuse(rankings, weights, depth=100):
        result=[]
        for rows in zip(*rankings):
            scores={}
            for row,w in zip(rows,weights):
                if w==0: continue
                for r,d in enumerate(row,1):
                    d=int(d); scores[d]=scores.get(d,0)+w/(60+r)
            result.append(sorted(scores,key=lambda d:(-scores[d],d))[:depth])
        return np.asarray(result,dtype=np.int32)
    def per_query_metrics(ranks, gold, k=5):
        match=ranks[:,:k]==np.asarray(gold)[:,None]
        hit=match.any(axis=1)
        rank=np.where(hit,match.argmax(axis=1)+1,0)
        rr=np.divide(1.0,rank,out=np.zeros(len(rank)),where=rank>0)
        dcg=np.divide(1.0,np.log2(rank+1),out=np.zeros(len(rank)),where=rank>0)
        return hit,rank,rr,dcg
    
    def metrics(ranks,gold,k=5):
        hit,rank,rr,dcg=per_query_metrics(ranks,gold,k)
        return {'n':len(gold),'k':k,'precision':hit.mean()/k,
                'recall':hit.mean(),'hit':hit.mean(),'mrr':rr.mean(),'ndcg':dcg.mean()}
    
    # Exact BM25 implementation from the prior repository, including binary query term counts.
    TOKEN_RE = re.compile(r'[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*')
    vec=CountVectorizer(lowercase=True,tokenizer=lambda t:TOKEN_RE.findall(t.lower()),
                        token_pattern=None,min_df=1,dtype=np.float32)
    mat=vec.fit_transform(corpus).tocsr().astype(np.float32)
    dl=np.asarray(mat.sum(axis=1)).ravel(); avg=float(dl.mean())
    df=np.asarray((mat>0).sum(axis=0)).ravel()
    idf=np.log1p((mat.shape[0]-df+.5)/(df+.5)).astype(np.float32)
    row_ids=np.repeat(np.arange(mat.shape[0]),np.diff(mat.indptr))
    tf=mat.data
    mat.data=(tf*(1.2+1)/(tf+1.2*(1-.75+.75*dl[row_ids]/avg)))*idf[mat.indices]
    qc=vec.transform(questions).tocsr();qc.data[:]=1
    br=top((qc@mat.T).toarray())
    cache=OUT/'embeddings.npz'
    if cache.exists():
        with np.load(cache) as saved:
            ce=saved['corpus']; qe=saved['questions']
    else:
        ce=encode(corpus);qe=encode(questions)
        np.savez_compressed(cache,corpus=ce,questions=qe)
    assert ce.shape==(4876,384) and qe.shape==(100,384)
    qr=top(qe@ce.T)
    BASELINES={'question_dense':qr,'bm25':br,
     'standard_hybrid_d25':fuse([br,qr],[.75,.25]),
     'standard_hybrid_d50':fuse([br,qr],[.5,.5])}
    baseline_table=pd.DataFrame([{'method':name,**metrics(r,gold)} for name,r in BASELINES.items()])
    display(baseline_table)
    for name,expected in [('question_dense',.29),('bm25',.31),('standard_hybrid_d25',.36)]:
        actual=metrics(BASELINES[name],gold)['hit']
        assert abs(actual-expected)<1e-8, f'{name} baseline differs ({actual}); stop and send this table back before API generation.'
    np.savez_compressed(OUT/'baseline_rankings.npz', gold=gold, **BASELINES)
    baseline_table.to_csv(OUT/'baseline_metrics.csv',index=False)
    print('Baseline 校验通过；可以开始 5 题 pilot。')

if args.stage in ['pilot','generate']:
    from openai import OpenAI
    try:
        client = OpenAI(api_key=(ROOT/'.env.local').read_text().split('=',1)[1].strip(), organization=ORGANIZATION,
                        max_retries=0, timeout=90)
    except Exception:
        raise RuntimeError('Missing locally configured credential.') from None
    try:
        available_models = {m.id for m in client.models.list().data}
    except Exception as exc:
        status = getattr(exc,'status_code',None)
        raise RuntimeError(f'模型权限检查失败，HTTP={status}。检查 key、Group 004 权限和网络；未发起生成。') from None
    if MODEL not in available_models:
        raise RuntimeError('Group 004 当前未列出所选固定模型。请把这个提示发回给我；不要自行切换模型。')
    print('Secret 已读取；Group 004 模型列表包含所选模型。尚未生成段落。')
    
    CHECKPOINT = OUT/'generation_checkpoint.json'
    state=json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {'fingerprint':fingerprint,'attempts':{},'outputs':{}}
    assert state['fingerprint']==fingerprint
    
    def budget_charge():
        # Actual known cost, plus conservative reservations for unresolved/failed attempts.
        return sum(a.get('cost_usd_estimate',a['reserved_usd']) for a in state['attempts'].values())
    
    def generate(indices):
        for profile,settings in PROFILES.items():
            for i in indices:
                qid=str(qa[i]['id']); key=profile+'::'+qid
                if key in state['outputs']: continue
                if key in state['attempts']:
                    raise RuntimeError('发现一次未完成或失败的历史请求。先下载检查点并发回分析；程序不会自动重复收费。')
                prompt=settings['prompt'].format(q=questions[i])
                # UTF-8 bytes + 256 is a conservative input-token allowance for these short plain-text requests.
                reserve=((len(prompt.encode('utf-8'))+256)*PRICE_INPUT + settings['max_output_tokens']*PRICE_OUTPUT)/1e6
                if budget_charge()+reserve > BUDGET_USD:
                    raise RuntimeError('达到本次实验预算保护，已停止。现有输出已保存。')
                state['attempts'][key]={'query_id':qid,'profile':profile,'reserved_usd':reserve,'status':'pending'}
                save_json(CHECKPOINT,state)
                started=time.perf_counter()
                try:
                    response=client.responses.create(model=MODEL,input=prompt,
                        max_output_tokens=settings['max_output_tokens'],
                        reasoning={'effort':'none'},temperature=0,store=False)
                except Exception as exc:
                    state['attempts'][key].update(status='failed_or_unknown',
                        error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
                    save_json(CHECKPOINT,state)
                    raise RuntimeError(f'API 请求停止，HTTP={getattr(exc,"status_code",None)}。已安全保存检查点；请发回不含 key 的错误提示。') from None
                seconds=time.perf_counter()-started
                usage=response.usage.model_dump() if response.usage else {}
                input_tokens=int(usage.get('input_tokens',0));output_tokens=int(usage.get('output_tokens',0))
                cost=(input_tokens*PRICE_INPUT+output_tokens*PRICE_OUTPUT)/1e6
                attempt=state['attempts'][key]
                attempt.update(status=response.status,response_id=response.id,usage=usage)
                if response.usage: attempt['cost_usd_estimate']=cost
                text=response.output_text.strip()
                output={'query_id':qid,'query_index':int(i),'profile':profile,'prompt':prompt,
                    'text':text,'model_returned':response.model,'response_id':response.id,
                    'status':response.status,'incomplete_details':response.incomplete_details.model_dump() if response.incomplete_details else None,
                    'seconds':seconds,'usage':usage,'cost_usd_estimate':cost if response.usage else None,
                    'words':len(text.split()),'reached_output_cap':output_tokens>=settings['max_output_tokens']}
                if text: state['outputs'][key]=output
                save_json(CHECKPOINT,state)
                if not text: raise RuntimeError('模型返回空文字，已停止；检查点已保留，不自动重发。')
                print(profile, qid, '|',len(state['outputs']),'/200 saved | estimated charged/reserved $',round(budget_charge(),4))
    
    
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock=threading.Lock()
    halt=threading.Event()
    
    def generate_one(profile,i):
        settings=PROFILES[profile]
        qid=str(qa[i]['id']);key=profile+'::'+qid
        prompt=settings['prompt'].format(q=questions[i])
        reserve=((len(prompt.encode('utf-8'))+256)*PRICE_INPUT+settings['max_output_tokens']*PRICE_OUTPUT)/1e6
        with lock:
            if halt.is_set():return
            if key in state['outputs']:return
            if key in state['attempts']:
                halt.set()
                raise RuntimeError('Unresolved attempt in checkpoint: '+key+'; no automatic retry.')
            if budget_charge()+reserve>BUDGET_USD:
                halt.set()
                raise RuntimeError('Run budget reached; all recorded outputs retained.')
            state['attempts'][key]={'query_id':qid,'profile':profile,'reserved_usd':reserve,'status':'pending'}
            save_json(CHECKPOINT,state)
        t=time.perf_counter()
        try:
            r=client.responses.create(model=MODEL,input=prompt,
              max_output_tokens=settings['max_output_tokens'],reasoning={'effort':'none'},temperature=0,store=False)
        except Exception as exc:
            with lock:
                state['attempts'][key].update(status='failed_or_unknown',error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
                save_json(CHECKPOINT,state);halt.set()
            raise RuntimeError('API stopped; HTTP='+str(getattr(exc,'status_code',None))) from None
        usage=r.usage.model_dump() if r.usage else {}
        input_tokens=int(usage.get('input_tokens',0));output_tokens=int(usage.get('output_tokens',0))
        cost=(input_tokens*PRICE_INPUT+output_tokens*PRICE_OUTPUT)/1e6
        text=r.output_text.strip()
        o={'query_id':qid,'query_index':i,'profile':profile,'prompt':prompt,'text':text,
           'model_returned':r.model,'response_id':r.id,'status':r.status,
           'incomplete_details':r.incomplete_details.model_dump() if r.incomplete_details else None,
           'seconds':time.perf_counter()-t,'usage':usage,'cost_usd_estimate':cost if r.usage else None,
           'words':len(text.split()),'reached_output_cap':output_tokens>=settings['max_output_tokens']}
        with lock:
            state['attempts'][key].update(status=r.status,response_id=r.id,usage=usage)
            if r.usage:state['attempts'][key]['cost_usd_estimate']=cost
            if text:state['outputs'][key]=o
            save_json(CHECKPOINT,state)
            n=len(state['outputs'])
            print(profile,qid,n,'/200; estimated/reserved USD',round(budget_charge(),4),flush=True)
            if not text:halt.set();raise RuntimeError('Empty output; no automatic retry.')
    
    def generate_parallel(indices):
        halt.clear()
        errors=[]
        tasks=[(profile,i) for profile in PROFILES for i in indices]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures=[ex.submit(generate_one,*t) for t in tasks]
            for f in as_completed(futures):
                try:f.result()
                except Exception as e:errors.append(str(e));halt.set()
        if errors:raise RuntimeError('; '.join(sorted(set(errors))))
    
    pilot_indices=sorted(range(len(qa)),key=lambda i:hashlib.sha256(str(qa[i]['id']).encode()).hexdigest())[:5]
    generate_parallel(pilot_indices if args.stage=='pilot' else list(range(100)))
    print('Generation stage done.',len(state['outputs']),flush=True)

if args.stage=='evaluate':
    state=json.loads((OUT/'generation_checkpoint.json').read_text())
    assert len(state['outputs'])==200, '尚未完成全部生成；请继续步骤 7，或先下载已有结果。'
    rankings=dict(BASELINES); audit=[]
    for profile in PROFILES:
        outputs=[state['outputs'][profile+'::'+str(q['id'])] for q in qa]
        texts=[x['text'] for x in outputs]
        he=encode(texts)
        pure=top(he@ce.T)
        mixed=top(normalize(.5*qe+.5*he)@ce.T)
        rankings[profile+'_pure_dense']=pure
        rankings[profile+'_mix_dense']=mixed
        rankings[profile+'_pure_hybrid_d25']=fuse([br,pure],[.75,.25])
        rankings[profile+'_mix_hybrid_d25']=fuse([br,mixed],[.75,.25])
        rankings[profile+'_mix_hybrid_d50']=fuse([br,mixed],[.5,.5])
        for o in outputs:
            audit.append({k:o[k] for k in ['query_id','profile','words','status','reached_output_cap','seconds','cost_usd_estimate']} |
              {'minilm_truncated':len(encoder.tokenizer(o['text'],truncation=False)['input_ids'])>256})
    summary=pd.DataFrame([{'method':name,**metrics(r,gold,k)} for name,r in rankings.items() for k in [1,3,5,10]])
    summary.to_csv(OUT/'retrieval_metrics.csv',index=False)
    summary[summary.k==5].to_csv(OUT/'comparison_k5.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'generation_audit.csv',index=False)
    pd.DataFrame(state['outputs'].values()).to_json(OUT/'generated_passages.jsonl',orient='records',lines=True,force_ascii=False)
    np.savez_compressed(OUT/'saved_rankings.npz',gold=gold,**rankings)
    per_rows=[]
    for method,r in rankings.items():
        hit,rank,rr,dcg=per_query_metrics(r,gold)
        for i,q in enumerate(qa):
            per_rows.append({'method':method,'query_id':q['id'],'hit5':bool(hit[i]),
              'gold_rank_top5':int(rank[i]),'reciprocal_rank5':float(rr[i]),'ndcg5':float(dcg[i]),
              'top5_passage_ids':json.dumps([passage_ids[j] for j in r[i,:5]])})
    pd.DataFrame(per_rows).to_csv(OUT/'per_query_metrics.csv',index=False)
    tests=[]
    for profile in PROFILES:
        pairs=[(profile+'_pure_dense','question_dense'),(profile+'_mix_dense','question_dense'),
          (profile+'_pure_hybrid_d25','standard_hybrid_d25'),
          (profile+'_mix_hybrid_d25','standard_hybrid_d25'),
          (profile+'_mix_hybrid_d50','standard_hybrid_d50'),
          (profile+'_mix_hybrid_d50','standard_hybrid_d25')]
        for candidate,control in pairs:
            a=per_query_metrics(rankings[candidate],gold)[0];b=per_query_metrics(rankings[control],gold)[0]
            won=int((a & ~b).sum());lost=int((~a & b).sum())
            p=binomtest(won,won+lost,.5).pvalue if won+lost else 1.0
            tests.append({'candidate':candidate,'control':control,'gained':won,'lost':lost,
              'delta_hit5':float(a.mean()-b.mean()),'mcnemar_p':p})
    # Holm step-down adjustment across all 12 prelisted comparisons.
    order=np.argsort([t['mcnemar_p'] for t in tests]); running=0
    for position,index in enumerate(order):
        running=max(running,min(1,(len(tests)-position)*tests[index]['mcnemar_p']))
        tests[index]['holm_p']=float(running)
    pd.DataFrame(tests).to_csv(OUT/'paired_tests.csv',index=False)
    display(summary[summary.k==5].reset_index(drop=True))
    display(pd.DataFrame(tests))
    cap_rate=pd.DataFrame(audit).groupby('profile')[['reached_output_cap','minilm_truncated']].mean()
    display(cap_rate)
    report=['# OpenAI HyDE retrieval results','',PROTOCOL['scope'],'',
      'All current metrics are recomputed from saved rankings. This run does not evaluate final-answer correctness.',
      'The 96-token and 160-token profiles differ in both prompt and length. OpenAI and local model decoding/tokenization differ.',
      'Historical SmolLM3 37% dense / 40% hybrid figures are recovered aggregate observations, not verified raw-data controls in this run.',
      'Compare hybrid against same-weight controls as well as the historical standard d25 control.',
      'Consult comparison_k5.csv, paired_tests.csv, generation_audit.csv, and per_query_metrics.csv.',
      'Token prices are estimates; organization billing is authoritative.',
      '', '```', summary[summary.k==5].to_string(index=False),'```']
    (OUT/'findings.md').write_text('\n'.join(report),encoding='utf-8')
    print('已保存全部指标、逐题排名、统计检验和生成审计。')