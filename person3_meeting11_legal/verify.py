"""Independent offline audit of Meeting11 scores, saved routes and budget packing.

Uses a separately constructed BM25 matrix and verifies each generated route.
No model calls or encoder reexecution. Does not import the experimental runner.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime,timezone

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_meeting11_legal/v1'
BASE=ROOT.parent


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rrf(lists,weights,cut):
    counts=Counter()
    for ids,w in zip(lists,weights):
        for position,doc in enumerate(ids,1):counts[int(doc)]+=w/(60+position)
    return sorted(counts,key=lambda j:(-counts[j],j))[:cut]
def score(ids,gold):
    try:r=list(ids).index(gold)+1
    except ValueError:r=0
    return [float(r>0),float(r>0),1/r if r else 0.,1/np.log2(r+1) if r else 0.]


def main():
    protocol=read(OUT/'protocol.json');split=read(OUT/'selection.json')
    for p,h in read(OUT/'source_hashes.json').items():assert digest(ROOT/p)==h,p
    assert digest(OUT.parent.parent.parent/'person3_meeting11_legal/run.py')==read(OUT/'implementation.json')['run_sha256']
    for p,h in read(OUT/'freeze_receipt.json')['hashes'].items():assert digest(OUT/p)==h,p
    qa=[json.loads(s) for s in (ROOT/'data/legal_rag_bench/qa.jsonl').read_text().splitlines()]
    pages=[json.loads(s) for s in (ROOT/'data/legal_rag_bench/corpus.jsonl').read_text().splitlines()]
    byqid={str(q['id']):q for q in qa};pids=[p['id'] for p in pages]
    assert len(qa)==100 and len(pages)==4876
    groups=[{byqid[q]['relevant_passage_id'] for q in split[key]} for key in ['train_ids','val_ids','test_ids']]
    assert list(map(len,[split[k] for k in ['train_ids','val_ids','test_ids']]))==[10,10,80]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    token_re=re.compile(r'[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*')
    texts=[(p['title']+' '+p['text']).strip() for p in pages]
    counters=[Counter(token_re.findall(t.lower())) for t in texts]
    vocab=sorted(set().union(*(set(c) for c in counters)));vi={t:i for i,t in enumerate(vocab)}
    lengths=np.array([sum(c.values()) for c in counters],dtype=np.float32)
    df=Counter(t for c in counters for t in c)
    av=float(lengths.mean());indptr=[0];indices=[];values=[]
    for i,c in enumerate(counters):
        for t in sorted(c):
            f=np.float32(c[t]);freq=df[t];idf=np.float32(np.log1p((len(pages)-freq+.5)/(freq+.5)))
            value=f*np.float32(2.2)/(f+np.float32(1.2)*(np.float32(.25)+np.float32(.75)*lengths[i]/np.float32(av)))*idf
            indices.append(vi[t]);values.append(value)
        indptr.append(len(indices))
    matrix=csr_matrix((np.array(values,np.float32),np.array(indices),np.array(indptr)),shape=(len(pages),len(vocab)))
    def bm25(query):
        cols=sorted({vi[t] for t in token_re.findall(query.lower()) if t in vi})
        q=csr_matrix((np.ones(len(cols),np.float32),(np.zeros(len(cols),int),cols)),shape=(1,len(vocab)))
        values=(q@matrix.T).toarray()[0]
        return np.argsort(-values,kind='stable')[:100].tolist()
    z=np.load(ROOT/'results/person3_openai_hyde/embeddings.npz');corpus=z['corpus']
    qidx={str(q['id']):i for i,q in enumerate(qa)}
    routes={v:read(OUT/f'{v}_routes.json') for v in ['default','legal']}
    rank_checks=dict(bm25=0,dense=0,hybrid=0,final=0,fixed100=0)
    independent_rankings={q:{} for q in byqid}
    independent_fixed={q:{} for q in byqid}
    for variant in ['default','legal']:
        vz=np.load(OUT/f'{variant}_query_vectors.npz');vectors=dict(zip(vz['texts'].tolist(),vz['vectors']))
        assert vz['vectors'].shape[1]==384 and np.isfinite(vz['vectors']).all()
        assert np.max(abs(np.linalg.norm(vz['vectors'],axis=1)-1))<1e-5
        queries=read((OUT if variant=='legal' else ROOT/'results/person3_meeting10_legal/v1')/('legal_queries.json' if variant=='legal' else 'multiquery.json'))
        for qid,q in byqid.items():
            record=routes[variant][qid];gold=pids.index(q['relevant_passage_id'])
            assert record['gold']==gold and record['routes'][-1]['query']==q['question']
            assert [r['query'] for r in record['routes']]==queries[qid]['queries']
            for item in record['routes']:
                b=bm25(item['query']);assert b==item['bm25'],(variant,qid,'BM25')
                v=z['questions'][qidx[qid]] if item['query']==q['question'] else vectors[item['query']]
                d=np.argsort(-(v@corpus.T),kind='stable')[:100].tolist()
                assert d==item['dense'];assert rrf([b,d],[.5,.5],100)==item['hybrid']
                for key in ['bm25','dense','hybrid']:rank_checks[key]+=1
            for backend in ['bm25','dense','hybrid']:
                for k in [5,8,10]:
                    lists=[r[backend] for r in record['routes']]
                    independent_rankings[qid].setdefault('question_'+backend,{})[str(k)]=lists[-1][:k]
                    independent_rankings[qid].setdefault(variant+'_'+backend,{})[str(k)]=rrf([x[:k] for x in lists],[1.]*len(lists),k)
                    independent_fixed[qid].setdefault(variant+'_'+backend,{})[str(k)]=rrf(lists,[1.]*len(lists),100)[:k]
    saved=read(OUT/'rankings.json');fixed=read(OUT/'fixed100_rankings.json')
    assert saved==independent_rankings;assert fixed==independent_fixed
    rank_checks.update(final=2700,fixed100=1800)
    metrics=pd.read_csv(OUT/'retrieval_per_query.csv',dtype={'query_id':str});assert len(metrics)==2700
    for row in metrics.itertuples():
        gold=pids.index(byqid[row.query_id]['relevant_passage_id'])
        values=score(saved[row.query_id][row.method][str(row.k)],gold)
        assert np.allclose(values,[row.hit,row.recall,row.mrr,row.ndcg],atol=1e-14,rtol=0)
        assert row.cohort==('test80' if row.query_id in split['test_ids'] else 'dev20')
    summary=pd.read_csv(OUT/'retrieval_summary.csv').set_index(['cohort','method','k'])
    for key,part in metrics.groupby(['cohort','method','k']):
        assert len(part)==summary.loc[key,'n']
        assert np.allclose(part[['hit','recall','mrr','ndcg']].mean(),summary.loc[key,['hit','recall','mrr','ndcg']].to_numpy(float),atol=1e-14)
    tests=pd.read_csv(OUT/'primary_tests.csv');assert len(tests)==9
    primary=metrics[(metrics.cohort=='test80')&(metrics.k==5)]
    for t in tests.itertuples():
        c=primary[primary.method==t.candidate].set_index('query_id').hit
        b=primary[primary.method==t.control].set_index('query_id').hit
        delta=c-b;wins=int((delta>0).sum());losses=int((delta<0).sum())
        assert (wins,losses)==(t.wins,t.losses)
        p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.
        assert abs(p-t.p_raw)<1e-14 and abs(delta.mean()-t.mean_delta)<1e-14
    order=np.argsort(tests.p_raw,kind='stable');adj=np.empty(9)
    adj[order]=np.minimum(1,np.maximum.accumulate(tests.p_raw.to_numpy()[order]*np.arange(9,0,-1)))
    assert np.allclose(adj,tests.p_holm9,atol=1e-14)
    os.environ['TIKTOKEN_CACHE_DIR']=str(BASE/'work/meeting11_20261123/tiktoken_cache')
    import tiktoken
    enc=tiktoken.get_encoding('o200k_base')
    records=read(OUT/'delivery_records.json');delivery=pd.read_csv(OUT/'delivery_per_query.csv',dtype={'query_id':str})
    assert len(records)==len(delivery)==8100
    lookup=delivery.set_index(['query_id','method','k','budget']);contexts={}
    for rec in records:
        qid=rec['query_id'];method=rec['method'];k=rec['k'];budget=rec['budget'];key=(qid,method,k,budget)
        ranked=saved[qid][method][str(k)];full=rec['full_ids'];partial=rec['partial_id']
        assert full==ranked[:len(full)]
        if partial is not None:assert partial==ranked[len(full)]
        content=[f'Passage {i+1}:\n{texts[pid]}' for i,pid in enumerate(full)]
        if partial is not None:
            fragment=enc.decode(enc.encode(texts[partial],disallowed_special=())[:rec['partial_tokens']],errors='ignore')
            assert texts[partial].startswith(fragment)
            content.append(f'Passage {len(full)+1}:\n{fragment}')
        context='\n\n'.join(content);n=len(enc.encode(context,disallowed_special=()))
        assert n==rec['context_tokens'] and n<=budget
        assert hashlib.sha256(context.encode()).hexdigest()==rec['context_sha256']
        touched=full+([partial] if partial is not None else []);gold=pids.index(byqid[qid]['relevant_passage_id'])
        row=lookup.loc[key];assert len(full)==row.full_passages and len(touched)==row.delivered_passages
        assert int(gold in full)==row.full_hit and int(gold in touched)==row.touched_hit
        assert row.excluded_seed_passages==k-len(touched)
        if len(full)<k:
            candidate='\n\n'.join([f'Passage {i+1}:\n{texts[pid]}' for i,pid in enumerate(ranked[:len(full)+1])])
            assert len(enc.encode(candidate,disallowed_special=()))>budget
        contexts[key]=context
    api=read(OUT/'api/cost_summary.json');assert api['states']=={'saved':100} and api['unresolved_reserved_usd']==0
    assert api['known_actual_cost_usd']<=.25 and api['cumulative_estimate_including_reserves_usd']<=7
    c=sqlite3.connect(f"file:{OUT/'api/api_checkpoint.sqlite'}?mode=ro",uri=True)
    db={r[0]:(json.loads(r[1]),json.loads(r[2]),r[3]) for r in c.execute('select key,request_json,response_json,cost from calls')}
    qcache=read(OUT/'legal_queries.json');ids=[];cost=0.
    for qid,q in qcache.items():
        req=q['request'];assert req['messages']==[{'role':'user','content':protocol['legal_prompt'].format(question=byqid[qid]['question'])}]
        assert req['model']==protocol['model'] and req['temperature']==0 and req['max_completion_tokens']==256
        canonical=json.dumps(req,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        key=hashlib.sha256(canonical.encode()).hexdigest();request,response,spent=db[key]
        assert request==req and response['id']==q['response_id'] and response['model']==protocol['model']
        assert response['choices'][0]['finish_reason']=='stop' and q['generated_query_count']==3
        assert q['generated_queries']==list(filter(None,response['choices'][0]['message']['content'].strip().split('\n')))
        assert q['queries']==q['generated_queries']+[byqid[qid]['question']]
        u=response['usage'];expected=(u['prompt_tokens']*.15+u['completion_tokens']*.60)/1e6
        assert abs(expected-spent)<1e-14;cost+=spent;ids.append(q['response_id'])
    assert len(set(ids))==100 and abs(cost-api['known_actual_cost_usd'])<1e-14
    c.close()
    result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),source_hashes_preserved=True,split_gold_disjoint=True,
        original_queries_reused=100,new_legal_outputs=100,rank_checks=rank_checks,metric_rows_checked=2700,
        exact_primary_tests_checked=9,holm_family=9,delivery_rows_repacked=8100,api_ledger_checked=True,
        new_api_estimate_usd=cost,cumulative_estimate_usd=api['cumulative_estimate_including_reserves_usd'],
        minimum_grid_budget_suffices_for_all=bool(delivery[delivery.budget==5000].all_k_fit.eq(1).all()),
        max_context_tokens=int(delivery.context_tokens.max()),api_calls_in_audit=0)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
